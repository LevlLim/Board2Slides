#!/usr/bin/env python3
"""Check the fresh-photo set and freeze it BEFORE any prediction is made.

    python fresh_photo_confirmation/tools/freeze.py            check only (safe to repeat)
    python fresh_photo_confirmation/tools/freeze.py --freeze   check, then write FROZEN.json (once)

Runs no OCR. Writes nothing unless --freeze is given and every check passes.
"""
from __future__ import annotations

import csv, datetime, hashlib, json, sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
PKG = KIT.parent
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
REQUIRED = ['photo_id', 'file', 'writer_id', 'writer_new_to_project', 'consent_for_evaluation', 'captured_on', 'device',
            'distance_m', 'angle', 'lighting', 'glare', 'board_type', 'layout', 'in_any_training_or_dev_set',
            'transcriber', 'reviewer', 'transcription_status']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_rows():
    with open(KIT / 'photos_manifest.csv', newline='', encoding='utf-8') as f:
        return [r for r in csv.DictReader(f) if any((v or '').strip() for v in r.values())]


def written_lines(pid):
    """Non-empty transcription lines, numbered from 1 in this order (quads refer to these numbers)."""
    lines = [l.strip() for l in (KIT / 'transcriptions' / f'{pid}.txt').read_text(encoding='utf-8').splitlines()]
    return [l for l in lines if l]


def reference_lines(pid):
    return [l for l in written_lines(pid) if l != '[illegible]']


def existing_project_images(sizes):
    """Paths under the Assignment folder (outside this kit) whose byte size equals a fresh photo's size."""
    root = PKG.parent; hits = {}
    for p in root.rglob('*'):
        try:
            if KIT in p.parents or p.suffix.lower() not in IMAGE_EXT or not p.is_file(): continue
            s = p.stat().st_size
        except OSError:
            continue
        if s in sizes: hits.setdefault(s, []).append(p)
    return hits


def check():
    problems = []; prot = json.loads((KIT / 'protocol.json').read_text())
    if prot.get('status') != 'agreed' or not prot.get('agreed_by') or not prot.get('agreed_on'):
        problems.append('protocol.json: set status "agreed", agreed_by and agreed_on before freezing')
    rows = load_rows(); S = prot['sample']
    if not S['photos_min'] <= len(rows) <= S['photos_max']:
        problems.append(f"{len(rows)} photos listed; protocol needs {S['photos_min']}-{S['photos_max']}")
    writers = {}
    for r in rows: writers.setdefault(r['writer_id'].strip(), []).append(r['photo_id'])
    if not S['writers_min'] <= len(writers) <= S['writers_max']:
        problems.append(f"{len(writers)} writers; protocol needs {S['writers_min']}-{S['writers_max']}")
    for w, ps in writers.items():
        if len(ps) > S['max_photos_per_writer']: problems.append(f'writer {w}: {len(ps)} photos > {S["max_photos_per_writer"]}')
        if w.lower() in {x.lower() for x in S['excluded_writers']}: problems.append(f'writer {w} is an excluded (project) writer')
    ids = [r['photo_id'] for r in rows]
    if len(set(ids)) != len(ids): problems.append('duplicate photo_id')
    hashes, sizes = {}, set()
    for r in rows:
        pid = r['photo_id']
        for k in REQUIRED:
            if not (r.get(k) or '').strip(): problems.append(f'{pid}: column {k} is empty')
        if r.get('writer_new_to_project', '').strip().lower() != 'yes': problems.append(f'{pid}: writer_new_to_project must be yes')
        if r.get('consent_for_evaluation', '').strip().lower() != 'yes': problems.append(f'{pid}: consent_for_evaluation must be yes')
        if r.get('in_any_training_or_dev_set', '').strip().lower() != 'no': problems.append(f'{pid}: in_any_training_or_dev_set must be no')
        if r.get('transcription_status', '').strip().lower() != 'reviewed': problems.append(f'{pid}: transcription_status must be reviewed')
        if r.get('transcriber', '').strip().lower() == r.get('reviewer', '').strip().lower(): problems.append(f'{pid}: reviewer must differ from transcriber')
        photo = KIT / 'photos' / r.get('file', '')
        if not r.get('file') or not photo.is_file(): problems.append(f'{pid}: photo file photos/{r.get("file")} not found'); continue
        h = sha(photo)
        if h in hashes: problems.append(f'{pid}: same image bytes as {hashes[h]}')
        hashes[h] = pid; sizes.add(photo.stat().st_size)
        t = KIT / 'transcriptions' / f'{pid}.txt'
        if not t.is_file() or not reference_lines(pid): problems.append(f'{pid}: transcriptions/{pid}.txt missing or empty')
        q = KIT / 'transcriptions' / f'{pid}.quads.json'
        if q.is_file():
            quads = json.loads(q.read_text()); n = len(written_lines(pid))
            bad = [e for e in quads if not (1 <= int(e['line']) <= n and len(e['quad']) == 4)]
            if bad or sorted(int(e['line']) for e in quads) != list(range(1, n + 1)):
                problems.append(f'{pid}: quads file must give one 4-point quad for each written line 1..{n} (including [illegible] lines)')
    if sizes and prot['sample']['photo_must_not_match_any_existing_project_image']:
        for s, paths in existing_project_images(sizes).items():
            for p in paths:
                if sha(p) in hashes: problems.append(f'{hashes[sha(p)]}: byte-identical to existing project image {p.relative_to(PKG.parent)}')
    return problems, rows, prot


def main():
    problems, rows, prot = check()
    for p in problems: print('PROBLEM:', p)
    if problems:
        print(f'{len(problems)} problem(s): not frozen.'); sys.exit(1)
    print(f'checks passed: {len(rows)} photos, {len({r["writer_id"] for r in rows})} writers')
    if '--freeze' not in sys.argv:
        print('dry run only; add --freeze to freeze'); return
    out = KIT / 'FROZEN.json'
    if out.exists():
        print('FROZEN.json already exists; the set is frozen and cannot be re-frozen.'); sys.exit(1)
    files = {'protocol.json': sha(KIT / 'protocol.json'), 'PROTOCOL.md': sha(KIT / 'PROTOCOL.md'),
             'photos_manifest.csv': sha(KIT / 'photos_manifest.csv'), 'package MANIFEST.json': sha(PKG / 'MANIFEST.json'),
             'tools/scoring.py': sha(KIT / 'tools/scoring.py'), 'tools/run_and_score.py': sha(KIT / 'tools/run_and_score.py')}
    for r in rows:
        files[f'photos/{r["file"]}'] = sha(KIT / 'photos' / r['file'])
        for suf in ('.txt', '.quads.json'):
            p = KIT / 'transcriptions' / f'{r["photo_id"]}{suf}'
            if p.is_file(): files[f'transcriptions/{p.name}'] = sha(p)
    out.write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                                   agreed_by=prot['agreed_by'], predictions_made_before_freeze=False, files=files), indent=1) + '\n')
    print(f'FROZEN: {len(files)} files hashed -> {out.name}. Next: tools/run_and_score.py')


if __name__ == '__main__':
    main()
