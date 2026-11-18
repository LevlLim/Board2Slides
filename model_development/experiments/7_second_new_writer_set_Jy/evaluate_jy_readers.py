#!/usr/bin/env python3
"""Evaluate reviewed, ORIGINAL Jy crops. No training and no source-file edits.

Each recogniser runs in its own process. Full-photo geometry is deliberately
not scored here: this is the reader-only experiment on supplied crop labels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ASSIGNMENT = PROJECT.parent
DEFAULT_OUT = PROJECT / 'outputs/jy_ocr_evaluation'
DEFAULT_MODELS = ['S0', 'S0s2', 'S1', 'S1s2', 'G0', 'G0s2']
SCHEMA = 1
BOOTSTRAPS = 5000
FIELDS = ['model', 'file', 'parent', 'photo', 'label', 'pred', 'score', 'seconds',
          'errors', 'characters', 'cer', 'exact_case', 'word_group', 'length_group']


def normalise(text):
    return ' '.join(str(text).lower().split())


def distance(a, b):
    """Unit-cost Levenshtein distance; no CER cap or punctuation removal."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def write_csv(path, rows, fields=FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.pending')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def inventory(limit=0):
    source = ASSIGNMENT / 'codex_jy_dataset_review_20261108/original_inventory.csv'
    root = ASSIGNMENT / 'Jy Dataset'
    with source.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 298 and len({r['path'] for r in rows}) == 298
    assert all(r['augmentation'] == 'original' for r in rows)
    selected = rows[:limit] if limit else rows
    entries = []
    for r in selected:
        path = (root / r['path']).resolve()
        assert path.is_relative_to(root.resolve())
        raw = path.read_bytes()
        if hashlib.md5(raw).hexdigest() != r['bytes_md5']:
            raise ValueError(f'Image changed since inventory: {path}')
        assert r['label'].strip(), f'Empty ground truth: {path}'
        entries.append(dict(file=r['path'], label=r['label'], parent=r['parent'],
                            photo=str(r['photo']), image_sha256=hashlib.sha256(raw).hexdigest()))
    identity = dict(schema=SCHEMA, source_manifest_sha256=sha(source),
                    selected_entries=entries, full_originals=298, limit=limit,
                    evaluator_sha256=sha(Path(__file__)), bootstraps=BOOTSTRAPS,
                    label_review='User stated on 8 Nov 2026 that teammate reviewed Jy labels.',
                    scoring='lowercase; collapse whitespace; preserve punctuation; uncapped CER',
                    primary='G0 vs S0, per-crop mean across matched seed 1/2: photo-cluster 95% CI below zero AND both seed changes negative',
                    secondary='S1 vs S0; word/length/photo subgroups; blank outputs',
                    scope='Reader-only on provided crops; not full-photo transcription accuracy.')
    return root, entries, identity


def bootstrap_difference(after, before):
    assert set(after) == set(before)
    groups = defaultdict(list)
    for file in after:
        a, b = after[file], before[file]
        assert a['label'] == b['label'] and str(a['photo']) == str(b['photo'])
        groups[str(a['photo'])].append(float(a['cer']) - float(b['cer']))
    parts = list(groups.values())
    mean = lambda gs: sum(map(sum, gs)) / sum(map(len, gs))
    rng = random.Random(0)
    values = sorted(mean([parts[rng.randrange(len(parts))] for _ in parts]) for _ in range(BOOTSTRAPS))
    return dict(difference=mean(parts), low=values[int(.025 * BOOTSTRAPS)],
                high=values[int(.975 * BOOTSTRAPS) - 1], photo_clusters=len(parts),
                better=sum(float(after[f]['cer']) < float(before[f]['cer']) - 1e-12 for f in after),
                worse=sum(float(after[f]['cer']) > float(before[f]['cer']) + 1e-12 for f in after))


def aggregate(rows):
    n = len(rows)
    return dict(crops=n, mean_line_CER=sum(float(r['cer']) for r in rows) / n,
                corpus_CER=sum(int(r['errors']) for r in rows) / max(1, sum(int(r['characters']) for r in rows)),
                exact_case=sum(int(r['exact_case']) for r in rows) / n,
                blank_readings=sum(not str(r['pred']).strip() for r in rows),
                inference_seconds=sum(float(r['seconds']) for r in rows))


def summary(rows, protocol, model_meta):
    by_model = defaultdict(list)
    for r in rows:
        by_model[r['model']].append(r)
    result = dict(protocol=protocol, models={m: aggregate(rs) for m, rs in by_model.items()},
                  model_identity=model_meta, subgroups={}, comparisons={})
    for field in ['word_group', 'length_group', 'photo']:
        groups = defaultdict(list)
        for r in rows:
            groups[(r['model'], str(r[field]))].append(r)
        result['subgroups'][field] = {f'{m} | {group}': aggregate(rs) for (m, group), rs in groups.items()}
    maps = {m: {r['file']: r for r in rs} for m, rs in by_model.items()}
    for base in ['G0', 'S1']:
        if all(m in maps for m in ['S0', 'S0s2', base, base + 's2']):
            pairs = [bootstrap_difference(maps[base + suffix], maps['S0' + suffix]) for suffix in ['', 's2']]
            averaged = {}
            for b in ['S0', base]:
                averaged[b] = {file: dict(r, cer=(float(r['cer']) + float(maps[b + 's2'][file]['cer'])) / 2)
                               for file, r in maps[b].items()}
            mean = bootstrap_difference(averaged[base], averaged['S0'])
            helps = mean['high'] < 0 and all(p['difference'] < 0 for p in pairs)
            hurts = mean['low'] > 0 and all(p['difference'] > 0 for p in pairs)
            decision = 'helps' if helps else ('hurts' if hurts else 'improvement not established')
            if protocol['limit']:
                decision = 'smoke sample — no formal conclusion'
            result['comparisons'][base + ' vs S0'] = dict(seed1=pairs[0], seed2=pairs[1], mean=mean,
                                                         role='primary confirmation' if base == 'G0' else 'secondary', decision=decision)
    return result


def report(result):
    protocol = result['protocol']
    lines = ['# Jy original-crop recognition evaluation', '',
             f"Generated {datetime.now(timezone.utc).isoformat()} (UTC). **Reader-only evaluation.** No training.", '',
             f"Selected {len(protocol['selected_entries'])} of 298 original crops; synthetic variants are excluded. Labels were reported reviewed by a teammate.", '',
             'The supplied train/test folder split is ignored: all originals remain evaluation data for these already-trained readers.', '',
             '| Model | Crops | Mean line CER | Corpus CER | Exact lines (case-sensitive) | Empty readings |',
             '|---|---:|---:|---:|---:|---:|']
    for m, x in result['models'].items():
        lines.append(f"| {m} | {x['crops']} | {x['mean_line_CER']:.2%} | {x['corpus_CER']:.2%} | {x['exact_case']:.2%} | {x['blank_readings']} |")
    lines += ['', '## Matched two-seed comparisons', '',
              'Difference is after minus S0: negative means lower error. Intervals resample source photos with the trained seeds held fixed.', '',
              '| Comparison | Role | Seed 1 change (pp) | Seed 2 change (pp) | Mean change (pp) | 95% interval (pp) | Decision |',
              '|---|---|---:|---:|---:|---|---|']
    for name, c in result['comparisons'].items():
        q = c['mean']
        lines.append(f"| {name} | {c['role']} | {100*c['seed1']['difference']:+.2f} | {100*c['seed2']['difference']:+.2f} | {100*q['difference']:+.2f} | {100*q['low']:+.2f} .. {100*q['high']:+.2f} | {c['decision']} |")
    lines += ['', '## Interpretation limits', '',
              '- This fresh-collection confirmation does not change the earlier whiteboard-validation primary verdict.',
              '- The 298 crops include 241 individual words and 57 multi-word crops; report length/word groups separately.',
              '- Writer identity/provenance is user supplied. This collection does not establish population-wide handwriting performance.',
              '- Crop-level scores do not include missed detections, merged lines, extra page text or reading-order errors.',
              '- Two-seed averages are statistics, not a deployable ensemble. Do not choose a seed by this test.',
              '- If these mistakes inform a new training or preprocessing change, Jy becomes development evidence for that change.',
              '- Proposed full-photo box positions and completeness require review before claiming full-photo CER.',
              '- Seconds are backend-reported inference time, excluding startup and image decoding; compare only on the same runtime.',
              '- Source files, labels and models were not edited. See protocol.json and per-model metadata for fingerprints.', '']
    return '\n'.join(lines)


def worker(args, root, entries, protocol):
    sys.path.insert(0, str(PROJECT / 'ocr_prototype'))
    from ocr_backend import RecognitionModel, load_image
    model = RecognitionModel(args.worker, threads=args.threads)
    model_identity = model.identity()
    cache = args.out / 'cache' / args.worker
    fingerprint = json.loads(json.dumps(dict(protocol=protocol, model=model_identity, threads=args.threads)))
    meta_path, rows_path = cache / 'identity.json', cache / 'predictions.csv'
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != fingerprint:
            raise SystemExit(f'Cache identity changed for {args.worker}; choose a new --out folder. Existing results preserved.')
        if rows_path.exists():
            print(f'{args.worker}: verified cached predictions retained.', flush=True)
            return
    results = []
    started = time.perf_counter()
    for i in range(0, len(entries), 8):
        chunk = entries[i:i + 8]
        readings = model.read([load_image(root / r['file']) for r in chunk])
        assert len(readings) == len(chunk)
        for r, reading in zip(chunk, readings):
            pred = str(reading['text'])
            chars = max(1, len(normalise(r['label'])))
            errors = distance(normalise(r['label']), normalise(pred))
            results.append(dict(model=args.worker, file=r['file'], parent=r['parent'], photo=r['photo'],
                                label=r['label'], pred=pred, score=float(reading['score']),
                                seconds=float(reading.get('seconds', 0)), errors=errors, characters=chars,
                                cer=errors / chars, exact_case=int(pred == r['label']),
                                word_group='one word' if len(r['label'].split()) == 1 else 'multiple words',
                                length_group='1-20' if len(r['label']) <= 20 else ('21-40' if len(r['label']) <= 40 else '>40')))
        print(f"{args.worker}: {len(results)}/{len(entries)} crops ({time.perf_counter()-started:.1f}s)", flush=True)
    assert len(results) == len(entries)
    write_csv(rows_path, results)
    write_json(meta_path, fingerprint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default=','.join(DEFAULT_MODELS))
    parser.add_argument('--limit', type=int, default=0, help='Smoke sample; separate default output and no formal conclusion.')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--check-only', action='store_true', help='Validate inventory and protocol without loading models.')
    args = parser.parse_args()
    if args.limit < 0 or args.limit > 298 or args.threads < 1:
        parser.error('limit must be 0..298 and threads positive')
    args.out = (args.out or (DEFAULT_OUT / f'smoke_{args.limit}' if args.limit else DEFAULT_OUT)).resolve()
    if args.limit and args.out == DEFAULT_OUT.resolve():
        parser.error('A limited smoke sample may not write into the full evaluation directory.')
    root, entries, protocol = inventory(args.limit)
    if args.check_only:
        print(json.dumps(dict(crops=len(entries), manifest=protocol['source_manifest_sha256'], protocol=protocol['primary']), indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    p = args.out / 'protocol.json'
    if p.exists() and json.loads(p.read_text()) != protocol:
        raise SystemExit('Evaluation protocol/data changed: choose a new --out directory; existing results preserved.')
    write_json(p, protocol)  # written BEFORE any model prediction
    if args.worker:
        worker(args, root, entries, protocol)
        return
    models = list(dict.fromkeys(args.models.split(',')))
    if not models or any(not re.fullmatch(r'[A-Za-z0-9_]+', m) for m in models):
        parser.error('Invalid model IDs')
    rows, identities = [], {}
    for model in models:
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', model,
                   '--limit', str(args.limit), '--threads', str(args.threads), '--out', str(args.out)]
        subprocess.run(command, check=True)
        cache = args.out / 'cache' / model
        identities[model] = json.loads((cache / 'identity.json').read_text())['model']
        with (cache / 'predictions.csv').open(newline='', encoding='utf-8') as f:
            current = list(csv.DictReader(f))
        assert len(current) == len(entries) and {r['file'] for r in current} == {r['file'] for r in entries}
        rows.extend(current)
        write_csv(args.out / 'predictions.csv', rows)
        result = summary(rows, protocol, identities)
        write_json(args.out / 'summary.json', result)
        (args.out / 'report.md').write_text(report(result))
    print(f'FINISHED: {args.out / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
