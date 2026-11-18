#!/usr/bin/env python3
"""Run the unchanged package once on the FROZEN fresh-photo set and score it with the frozen rules.

    python fresh_photo_confirmation/tools/run_and_score.py

Refuses to run if FROZEN.json is missing, if any frozen file has changed, if the package fails its integrity check,
or if results already exist (single run). Writes results/predictions/*, results/RESULTS.json and results/RESULTS.md.
"""
from __future__ import annotations

import datetime, json, subprocess, sys, tempfile
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
PKG = KIT.parent
sys.path.insert(0, str(KIT / 'tools')); sys.path.insert(0, str(PKG / 'b2n_ocr'))
from freeze import load_rows, sha, written_lines        # noqa: E402
from scoring import bootstrap, evaluate_photo, norm, page_score, reordered_page   # noqa: E402

OUT = KIT / 'results'


def preflight(args_test=False):
    frozen = KIT / 'FROZEN.json'
    if not frozen.is_file(): sys.exit('Not frozen: run tools/freeze.py --freeze first (before any prediction).')
    f = json.loads(frozen.read_text())
    paths = {'package MANIFEST.json': PKG / 'MANIFEST.json'}
    changed = [k for k, h in f['files'].items() if sha(paths.get(k, KIT / k)) != h]
    if changed: sys.exit(f'Frozen files changed since freezing: {changed}. Do not edit frozen inputs; see PROTOCOL.md.')
    r = subprocess.run([sys.executable, str(PKG / 'tools/verify_package.py')], capture_output=True, text=True)
    if r.returncode != 0: sys.exit('Package integrity check failed:\n' + r.stdout[-2000:])
    if (OUT / 'RESULTS.json').exists(): sys.exit('results/RESULTS.json exists: the confirmation run is single-shot.')
    return f


def predict(rows):
    from ocr_backend import OCRService
    from student_flow import decode_photo
    svc = OCRService('S1'); preds = {}
    (OUT / 'predictions').mkdir(parents=True, exist_ok=True)
    for r in rows:
        pid = r['photo_id']
        try:
            image = decode_photo((KIT / 'photos' / r['file']).read_bytes())
        except ValueError as exc:
            preds[pid] = dict(status='rejected', message=str(exc), lines=[], text='')
        else:
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / 'whiteboard.png'; image.save(p, format='PNG'); res = svc.recognise(str(p))
            preds[pid] = dict(status='ok', **res)
        (OUT / 'predictions' / f'{pid}.json').write_text(json.dumps(preds[pid], ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        (OUT / 'predictions' / f'{pid}.txt').write_text(preds[pid]['text'] + '\n', encoding='utf-8')
        print(f"{pid}: {preds[pid]['status']}, {len(preds[pid]['lines'])} regions")
    return preds, svc.reader.identity()


def score_photo(r, pred):
    pid = r['photo_id']; written = written_lines(pid)
    reference = '\n'.join(l for l in written if l != '[illegible]')
    hyp = ' '.join(l['text'] for l in pred['lines'])
    errors, chars = page_score(reference, hyp)
    out = dict(photo_id=pid, writer_id=r['writer_id'], layout=r['layout'], status=pred['status'], regions=len(pred['lines']),
               errors=errors, chars=chars, page_CER=errors / chars, reference=norm(reference), prediction=norm(hyp))
    q = KIT / 'transcriptions' / f'{pid}.quads.json'
    if q.is_file() and pred['lines']:
        quad = {int(e['line']): e['quad'] for e in json.loads(q.read_text())}
        gt = [dict(id=f'{pid}.{i}', text='' if t == '[illegible]' else t, quad=quad[i], proposed_reading_order=i,
                   illegible=t == '[illegible]') for i, t in enumerate(written, 1)]
        ill = {g['id'] for g in gt if g['illegible']}
        _, comps = evaluate_photo(gt, [dict(index=l['index'], box=l['box'], text=l['text']) for l in pred['lines']])
        for c in comps:
            if c['gt_ids'] and set(c['gt_ids']) <= ill: c['kind'] = 'illegible_region'
        kinds = ('one_to_one', 'merge_candidate', 'split_candidate', 'many_to_many_candidate', 'missed_target',
                 'unmatched_prediction', 'illegible_region')
        scored = [c for c in comps if c['gt_ids'] and c['kind'] != 'illegible_region']
        one = [c for c in comps if c['kind'] == 'one_to_one']
        out['regions_diagnostic'] = {k: sum(c['kind'] == k for c in comps) for k in kinds}
        out['reordered_errors'] = reordered_page(gt, comps, reference)
        out['matched_line'] = dict(errors=sum(c['errors'] for c in one), chars=sum(c['reference_characters'] for c in one), lines=len(one))
        out['annotated_target'] = dict(errors=sum(c['errors'] for c in scored), chars=sum(c['reference_characters'] for c in scored))
        out['components'] = comps
    return out


def pooled(ps):
    e = sum(p['errors'] for p in ps); c = sum(p['chars'] for p in ps); return dict(errors=e, chars=c, page_CER=e / c, photos=len(ps))


def report(per, prot, frozen, identity):
    allp = list(per.values()); res = dict(main=pooled(allp))
    bs = prot['secondary']['photo_bootstrap_interval']
    lo, hi = bootstrap(per, B=bs['resamples'], seed=bs['seed'], level=bs['level'])
    res['main']['photo_bootstrap_95'] = [lo, hi]
    res['per_writer'] = {w: pooled([p for p in allp if p['writer_id'] == w]) for w in sorted({p['writer_id'] for p in allp})}
    res['per_layout'] = {w: pooled([p for p in allp if p['layout'] == w]) for w in sorted({p['layout'] for p in allp})}
    q = [p for p in allp if 'components' in p]
    if q:
        c = sum(p['chars'] for p in q)
        res['with_quads'] = dict(photos=len(q), page_CER_app_order=sum(p['errors'] for p in q) / c,
                                 page_CER_reference_order=sum(p['reordered_errors'] for p in q) / c,
                                 matched_line_CER=sum(p['matched_line']['errors'] for p in q) / max(sum(p['matched_line']['chars'] for p in q), 1),
                                 matched_lines=sum(p['matched_line']['lines'] for p in q),
                                 regions={k: sum(p['regions_diagnostic'][k] for p in q) for k in q[0]['regions_diagnostic']})
    d = prot['decision_rule']
    res['decision'] = ('descriptive (no pass/fail)' if d['type'] == 'descriptive' else
                       f"threshold {d['threshold_page_CER']:.2%}: {'MET' if res['main']['page_CER'] <= d['threshold_page_CER'] else 'NOT MET'}")
    meta = dict(run_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), frozen_at=frozen['frozen_at'],
                reader=identity, package_manifest_sha256=sha(PKG / 'MANIFEST.json'))
    (OUT / 'RESULTS.json').write_text(json.dumps(dict(meta=meta, summary=res, photos=per), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    m = res['main']; L = [f"# Fresh-photo confirmation results\n\nRun {meta['run_at']} on the set frozen {meta['frozen_at']}. Rules: PROTOCOL.md / protocol.json.\n",
         f"**Main: pooled page CER (app reading order) {m['page_CER']:.2%}** ({m['errors']} edits / {m['chars']} reference characters, {m['photos']} photos); "
         f"photo bootstrap 95% {lo:.2%}–{hi:.2%} (writers fixed). Decision: {res['decision']}.\n",
         '| writer | photos | page CER |', '|---|---|---|'] + [f"| {w} | {v['photos']} | {v['page_CER']:.2%} |" for w, v in res['per_writer'].items()]
    L += ['', '| layout | photos | page CER |', '|---|---|---|'] + [f"| {w} | {v['photos']} | {v['page_CER']:.2%} |" for w, v in res['per_layout'].items()]
    if q:
        w = res['with_quads']
        L += ['', f"With reference geometry ({w['photos']} photos): page CER in reference order {w['page_CER_reference_order']:.2%} "
              f"(app order {w['page_CER_app_order']:.2%}; the difference is the reading-order cost, deferred work). "
              f"Matched-line CER {w['matched_line_CER']:.2%} on {w['matched_lines']} one-to-one lines. Regions: {w['regions']}."]
    L += ['', '| photo | writer | layout | regions | edits / chars | page CER |', '|---|---|---|---|---|---|'] + \
         [f"| {p['photo_id']} | {p['writer_id']} | {p['layout']} | {p['regions']} | {p['errors']}/{p['chars']} | {p['page_CER']:.2%} |" for p in allp]
    L += ['', 'Small sample: 2–3 writers cannot establish general accuracy for new writers; report as confirmation evidence only.']
    (OUT / 'RESULTS.md').write_text('\n'.join(L) + '\n', encoding='utf-8'); print('\n'.join(L[:3]))


def main():
    frozen = preflight(); prot = json.loads((KIT / 'protocol.json').read_text()); rows = load_rows()
    preds, identity = predict(rows)
    per = {r['photo_id']: score_photo(r, preds[r['photo_id']]) for r in rows}
    report(per, prot, frozen, identity)


if __name__ == '__main__':
    main()
