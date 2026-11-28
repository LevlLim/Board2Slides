#!/usr/bin/env python3
"""Board2Notes scalability check: handwritten paper notes (18 Nov 2026).

What it does, per photo:
  1. EXIF-orient the photo (as the Board2Slides upload does).
  2. Two conditions:
       original      - the upright photo as taken
       app_prepared  - Board2Slides' own "Automatic preparation" (app/prepare_infer.py --mode auto),
                       run unchanged as a subprocess; its decisions (perspective, crop, rotation,
                       lighting) are recorded
  3. Detect text lines ONCE per condition with the 9 Nov OCR package's detector settings
     (PP-OCRv5_mobile_det, quick pass + small-text retry, line grouping, reading order, quad crop).
  4. Give the SAME crops to every reader (so detector differences cannot look like reader differences).
  5. Score two ways against the typed reference text:
       reading     - mean line CER on reference lines that the detector found (crops matched to lines)
       end to end  - whole-page CER: missed lines count as fully wrong, extra boxes as insertions;
                     reported in the app's reading order and order-free
  6. Write overlays, crops, CSV/JSON, summary.md and review.html.

Commands
  python3 run_paper_notes_eval.py run    [--photos DIR] [--out DIR] [--readers a,b] [--max-seconds N]
  python3 run_paper_notes_eval.py report [--out DIR]
  python3 run_paper_notes_eval.py check  [--n 30] [--out DIR]   # plumbing: re-read crops with recorded predictions

`run` caches every step (detection, preparation, each reader) and can be stopped and re-run: it
continues where it stopped. --max-seconds makes it exit cleanly before a time limit.

Photo names must contain page1, page2 or page3 (e.g. page1_flat.jpg, page1_angle.jpg).
Scoring: case-insensitive, runs of whitespace collapsed (the same normalisation as the training reports).
Nothing here trains or changes a model, the package or the app.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent                                   # Assignment/scalability_paper_notes_20261118
ASSIGNMENT = WORK.parent                             # Assignment/
PKG = ASSIGNMENT / 'board2notes_ocr_package'         # the 9 Nov OCR package (detector + reader runtime)
APP = ASSIGNMENT.parent / 'B2N' / 'Board2Slides'     # the team's application repo (preparation step)
APP_DETECTOR = APP / 'models' / 'ocr' / 'detector'

M = ASSIGNMENT
READERS = {   # key: (label, exported model folder[, input shape override]). All exist; nothing is trained here.
    # The downloaded reader is read at 48 px with crops padded to 1024 px, exactly as every experiment scored it
    # (reproduces the recorded predictions). PaddleOCR's default resize (minimum width 320) reads long lines worse:
    # use 'downloaded_default' to see that.
    'downloaded': ('Downloaded PP-OCRv5 server reader (no fine-tuning), 48 px, padded to 1024 px as in our evaluation',
                   M / 'models' / 'PP-OCRv5_server_rec_official_hf', [3, 48, 1024]),
    'whiteboard': ('Whiteboard reader: our whiteboard lines, 64x1024, width pooling',
                   M / 'results_versions_2x2/results/models/PP-OCRv5_server_rec_board2notes_V2'),
    'paper':      ('Paper reader: IAM paper lines, same training settings as the whiteboard reader',
                   M / 'results_versions_2x2/results/models/PP-OCRv5_server_rec_board2notes_V1b'),
    'final':      ('Final model: stage-wise LR, teammate dataset',
                   M / 'results_server_his_20261110/results/models/PP-OCRv5_server_rec'),
    'stagewise':  ('Stage-wise LR reader: our whiteboard lines',
                   M / 'results_strategies_s0s3/results/models/PP-OCRv5_server_rec_board2notes_S1'),
    'downloaded_default': ('Downloaded reader with PaddleOCR default resize (48 px, minimum width 320)',
                           M / 'models' / 'PP-OCRv5_server_rec_official_hf', [3, 48, 320]),
}
DEFAULT_READERS = ('downloaded', 'whiteboard', 'paper', 'final', 'stagewise')


def load_reader(key, threads=4):
    import ocr_backend as ob
    ob.MODELS[key] = READERS[key][1]
    model = ob.RecognitionModel(key, threads=threads)
    if len(READERS[key]) > 2:
        model.shape = list(READERS[key][2])
        model._identity['image_shape'] = model.shape
    return model
CONDITIONS = ('original', 'app_prepared')
LONG_LINE = 60          # characters; lines longer than this are reported separately
MATCH_MIN = 65          # minimum mean partial-match score (0-100) for a crop to belong to a reference line

if platform.system() == 'Linux' and platform.machine() == 'aarch64':
    os.environ.setdefault('B2N_DET_CPU_THREADS', '1')   # package note: multithreaded detection crashes here
os.environ['B2N_PACKAGE_ROOT'] = str(PKG)
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
sys.path.insert(0, str(PKG / 'b2n_ocr'))


# ----------------------------------------------------------------------------- text and scoring
def norm(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or '').replace('　', ' ')).strip().lower()


def edits(a: str, b: str) -> int:
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(a, b)


def cer(pred: str, ref: str) -> float:
    r = norm(ref)
    return edits(norm(pred), r) / max(1, len(r))


def cer_nospace(pred: str, ref: str) -> float:
    r = norm(ref).replace(' ', '')
    return edits(norm(pred).replace(' ', ''), r) / max(1, len(r))


def page_of(name: str):
    m = re.search(r'page\s*_?\s*([123])', name.lower())
    return int(m.group(1)) if m else None


def load_refs():
    refs = {}
    for p in (1, 2, 3):
        f = WORK / 'reference_text' / f'page{p}.txt'
        if f.is_file():
            refs[p] = [l.strip() for l in f.read_text(encoding='utf-8').splitlines() if l.strip()]
    return refs


# ----------------------------------------------------------------------------- caching helpers
def jload(p: Path, default=None):
    return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else default


def jsave(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')
    tmp.replace(p)


class Budget:
    def __init__(self, seconds):
        self.t0, self.seconds = time.time(), seconds

    def left(self):
        return self.seconds is None or time.time() - self.t0 < self.seconds


# ----------------------------------------------------------------------------- pipeline steps
def photos_in(folder: Path):
    exts = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp'}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts and not p.name.startswith('.'))


def upright(photo: Path, out: Path) -> Path:
    """EXIF orientation applied, saved as JPEG (what the app's upload step stores)."""
    dst = out / 'upright' / (photo.stem + '.jpg')
    if dst.is_file():
        return dst
    from PIL import Image, ImageOps
    if photo.suffix.lower() in ('.heic', '.heif'):
        import pillow_heif
        pillow_heif.register_heif_opener()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(photo) as im:
        ImageOps.exif_transpose(im).convert('RGB').save(dst, quality=95)
    return dst


def app_prepare(up: Path, out: Path):
    """Run Board2Slides' Automatic preparation exactly as the app does (subprocess, its own code)."""
    img = out / 'prepared' / (up.stem + '.jpg')
    log = out / 'prepared' / (up.stem + '.json')
    if img.is_file() and log.is_file():
        return img, jload(log)
    img.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, BOARD2SLIDES_OCR_DET_DIR=str(APP_DETECTOR))
    t0 = time.time()
    # app_prepare_wrapper.py runs app/prepare_infer.py unchanged (only a thread-count workaround on Linux aarch64)
    r = subprocess.run([sys.executable, str(HERE / 'app_prepare_wrapper.py'), str(APP / 'app'), str(up), str(img),
                        str(log), '--mode', 'auto'], cwd=str(APP / 'app'), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'App preparation failed for {up.name}:\n{r.stderr[-2000:]}')
    info = jload(log)
    info['wall_seconds'] = round(time.time() - t0, 2)
    jsave(log, info)
    return img, info


_DETECTOR = None


def detect(image_path: Path, key: str, out: Path):
    """Package detection once; crops saved; result cached."""
    cache = out / 'cache' / f'{key}__det.json'
    hit = jload(cache)
    if hit:
        return hit
    global _DETECTOR
    import cv2
    import numpy as np
    import ocr_backend as ob
    from preprocessing import text_lines as tl
    load = 0.0
    if _DETECTOR is None:
        t = time.time()
        _DETECTOR = tl.Detector(ob.DETECTOR_NAME, det_args={'thresh': .20, 'box_thresh': .40, 'unclip_ratio': 1.2},
                                model_dir=ob.DETECTOR_DIR, runtime=ob.DETECTOR_RUNTIME)
        load = time.time() - t
    t = time.time()
    image = ob.load_image(image_path)
    boxes, info = _DETECTOR.detect(image, unclip=1.2)
    boxes, _ = tl.group_into_lines(boxes)
    boxes = [boxes[i] for i in ob.spatial_order(boxes)]
    crops_dir = out / 'crops' / key
    crops_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i, p in enumerate(boxes):
        c = tl.quad_crop(image, p)
        n = f'crop_{i + 1:02d}.png'
        cv2.imwrite(str(crops_dir / n), c)
        names.append(n)
    res = dict(image=str(image_path), width=int(image.shape[1]), height=int(image.shape[0]),
               boxes=[np.asarray(p).astype(float).round(1).tolist() for p in boxes], crops=names,
               detector_info={k: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in info.items()},
               detect_seconds=round(time.time() - t, 2), detector_load_seconds=round(load, 2))
    jsave(cache, res)
    return res


def read_all(reader_key: str, jobs, out: Path, budget: Budget):
    """Read every pending (key, det) job with one reader; one model in memory at a time."""
    pending = [(k, d) for k, d in jobs if not (out / 'cache' / f'{k}__{reader_key}.json').is_file()]
    if not pending:
        return True
    import cv2
    t = time.time()
    model = load_reader(reader_key)
    load = time.time() - t
    for k, d in pending:
        if not budget.left():
            return False
        crops = [cv2.imread(str(out / 'crops' / k / n)) for n in d['crops']]
        t = time.time()
        rows = model.read(crops) if crops else []
        jsave(out / 'cache' / f'{k}__{reader_key}.json',
              dict(reader=reader_key, model=model.identity(), model_load_seconds=round(load, 2),
                   read_seconds=round(time.time() - t, 2),
                   lines=[dict(crop=n, text=r['text'], score=round(r['score'], 4)) for n, r in zip(d['crops'], rows)]))
        load = 0.0
    del model
    return True


# ----------------------------------------------------------------------------- matching and scoring
def match(det, readings, refs):
    """Assign each crop to the reference line it best matches, using the MEAN over all readers
    (reader-neutral), so every reader is scored on the same crop-to-line assignment."""
    from rapidfuzz import fuzz
    n = len(det['crops'])
    assign, scores = [], []
    for i in range(n):
        preds = [norm(readings[r]['lines'][i]['text']) for r in readings]
        best, best_j, second = -1.0, None, -1.0
        for j, ref in enumerate(refs):
            r = norm(ref)
            # partial_ratio lets a split piece match its whole line; the length factor stops a long crop from
            # matching a short line that happens to be contained in it (e.g. "A filter slide" in the next line)
            vals = [fuzz.partial_ratio(p, r) * min(1.0, len(r) / len(p)) if len(p) >= 1 else 0.0 for p in preds]
            s = sum(vals) / len(vals)
            if s > best:
                second, best, best_j = best, s, j
            elif s > second:
                second = s
        ok = best >= MATCH_MIN and max(len(p) for p in preds) >= 2
        assign.append(best_j if ok else None)
        scores.append(dict(best=round(best, 1), second=round(max(second, 0), 1)))
    return assign, scores


def score_page(det, readings, refs, assign, scores):
    xs = [min(pt[0] for pt in b) for b in det['boxes']]
    per_reader = {}
    for r, rd in readings.items():
        preds = [l['text'] for l in rd['lines']]
        lines = []
        for j, ref in enumerate(refs):
            idx = sorted((i for i, a in enumerate(assign) if a == j), key=lambda i: xs[i])
            pred = ' '.join(preds[i] for i in idx)
            lines.append(dict(line=j + 1, ref=ref, pred=pred, crops=[det['crops'][i] for i in idx],
                              found=bool(idx), split=len(idx) > 1,
                              cer=round(cer(pred, ref), 4) if idx else None,
                              cer_nospace=round(cer_nospace(pred, ref), 4) if idx else None,
                              edits=edits(norm(pred), norm(ref)) if idx else len(norm(ref))))
        found = [l for l in lines if l['found']]
        extra = [i for i, a in enumerate(assign) if a is None]
        ref_chars = sum(len(norm(x)) for x in refs)
        e2e_order = edits(norm(' '.join(preds)), norm(' '.join(refs))) / max(1, ref_chars)
        e2e_free = (sum(l['edits'] for l in lines) + sum(len(norm(preds[i])) for i in extra)) / max(1, ref_chars)
        per_reader[r] = dict(
            lines=lines,
            reading_mean_line_cer=round(sum(l['cer'] for l in found) / len(found), 4) if found else None,
            reading_mean_line_cer_nospace=round(sum(l['cer_nospace'] for l in found) / len(found), 4) if found else None,
            reading_long_lines=round(sum(l['cer'] for l in found if len(l['ref']) > LONG_LINE) /
                                     max(1, sum(1 for l in found if len(l['ref']) > LONG_LINE)), 4),
            e2e_page_cer_app_order=round(e2e_order, 4),
            e2e_page_cer_order_free=round(e2e_free, 4),
            found_lines=len(found), missed_lines=[l['line'] for l in lines if not l['found']],
            split_lines=[l['line'] for l in lines if l['split']],
            extra_boxes=[det['crops'][i] for i in extra],
            read_seconds=rd['read_seconds'], model_load_seconds=rd['model_load_seconds'])
    merged = [det['crops'][i] for i, s in enumerate(scores) if assign[i] is not None and s['second'] >= 80]
    return per_reader, merged


# ----------------------------------------------------------------------------- outputs
def overlay(det, assign, out_file: Path):
    import cv2
    import numpy as np
    img = cv2.imread(det['image'])
    for i, b in enumerate(det['boxes']):
        pts = np.array(b, dtype=np.int32)
        colour = (40, 160, 40) if assign[i] is not None else (30, 30, 220)
        cv2.polylines(img, [pts], True, colour, max(2, img.shape[1] // 600))
        x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
        label = f'{i + 1}' + (f'>L{assign[i] + 1}' if assign[i] is not None else '>?')
        cv2.putText(img, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, img.shape[1] / 2400),
                    colour, max(1, img.shape[1] // 900), cv2.LINE_AA)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    scale = min(1.0, 1800 / max(img.shape[:2]))
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_file), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def fmt(x):
    return '–' if x is None else f'{100 * x:.1f}'


def report(out: Path):
    refs = load_refs()
    index = jload(out / 'index.json', {})
    if not index:
        print('Nothing to report: run first.')
        return
    results, rows = {}, []
    for key, meta in index.items():
        det = jload(out / 'cache' / f'{key}__det.json')
        readings = {r: jload(out / 'cache' / f'{key}__{r}.json') for r in meta['readers']}
        if not det or any(v is None for v in readings.values()):
            print(f'incomplete, skipped: {key}')
            continue
        page_refs = refs[meta['page']]
        assign, scores = match(det, readings, page_refs)
        per_reader, merged = score_page(det, readings, page_refs, assign, scores)
        overlay(det, assign, out / 'overlays' / f'{key}.jpg')
        results[key] = dict(meta, detect_seconds=det['detect_seconds'], n_boxes=len(det['boxes']),
                            detector_info=det['detector_info'], possibly_merged_boxes=merged,
                            assignment=[None if a is None else a + 1 for a in assign], match_scores=scores,
                            readers=per_reader)
        for r, pr in per_reader.items():
            for l in pr['lines']:
                rows.append(dict(photo=meta['photo'], condition=meta['condition'], page=meta['page'], reader=r,
                                 line=l['line'], ref=l['ref'], pred=l['pred'], found=l['found'],
                                 split=l['split'], cer='' if l['cer'] is None else l['cer'],
                                 crops=' '.join(l['crops'])))
    jsave(out / 'results.json', results)
    with (out / 'lines.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['photo'])
        w.writeheader()
        w.writerows(rows)
    write_summary(out, results)
    write_html(out, results)
    print(f'report: {len(results)} photo-conditions -> {out / "summary.md"}, {out / "review.html"}')


def write_summary(out: Path, results):
    L = ['# Paper-notes scalability check: results', '',
         'Mean line CER (%), case-insensitive, whitespace collapsed. **Reading** = mean over reference lines the '
         'detector found (identical crops for every reader). **End to end** = whole page, missed lines fully '
         'wrong, extra boxes as insertions; "app order" uses the package reading order, "order-free" matches '
         'crops to lines first.', '',
         'Detector and crops: 9 Nov OCR package settings. `app_prepared` = Board2Slides Automatic preparation '
         'run unchanged before detection. This is the OCR package, not the final deployed app.', '']
    for key, res in results.items():
        prep = res.get('preparation')
        L += [f"## {res['photo']} · {res['condition']} (page {res['page']})", '',
              f"{res['n_boxes']} boxes · detection {res['detect_seconds']} s · detector path: "
              f"{res['detector_info'].get('path', '')}"]
        if prep:
            L.append('Preparation: ' + '; '.join(f"{d['name']} {'applied' if d['applied'] else 'skipped'} "
                                                 f"({d['reason']})" for d in prep.get('decisions', [])))
        L += ['', '| reader | reading | reading, spaces ignored | reading, lines >60 chars | end to end (app order) | '
                  'end to end (order-free) | found / lines | missed | split | extra boxes | read s |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
        for r, pr in res['readers'].items():
            n = len(pr['lines'])
            L.append(f"| {r} | {fmt(pr['reading_mean_line_cer'])} | {fmt(pr['reading_mean_line_cer_nospace'])} | "
                     f"{fmt(pr['reading_long_lines'])} | {fmt(pr['e2e_page_cer_app_order'])} | "
                     f"{fmt(pr['e2e_page_cer_order_free'])} | {pr['found_lines']} / {n} | "
                     f"{pr['missed_lines'] or '–'} | {pr['split_lines'] or '–'} | {len(pr['extra_boxes'])} | "
                     f"{pr['read_seconds']} |")
        longs = [l['line'] for l in next(iter(res['readers'].values()))['lines'] if len(l['ref']) > LONG_LINE]
        if longs:
            L += ['', f'Long lines (>{LONG_LINE} characters): ' + '; '.join(
                f"line {j}: " + ', '.join(f"{r} {fmt(pr['lines'][j - 1]['cer'])}" for r, pr in res['readers'].items())
                for j in longs)]
        if res['page'] == 3 and {'downloaded', 'final'} <= set(res['readers']):
            L += ['', 'Page 3, line by line (before = downloaded, after = final; stage-wise shown if read):', '',
                  '| line | reference | downloaded | final | change | stage-wise |', '|---|---|---|---|---|---|']
            d, fi = res['readers']['downloaded']['lines'], res['readers']['final']['lines']
            sw = res['readers'].get('stagewise', {}).get('lines')
            for a, b, c in zip(d, fi, sw or [None] * len(d)):
                if a['cer'] is None:
                    change = 'not found'
                else:
                    change = 'improved' if b['cer'] < a['cer'] else 'regressed' if b['cer'] > a['cer'] else 'unchanged'
                L.append(f"| {a['line']} | {a['ref']} | {a['pred']} ({fmt(a['cer'])}) | {b['pred']} ({fmt(b['cer'])}) | "
                         f"{change} | {'' if c is None else c['pred'] + ' (' + fmt(c['cer']) + ')'} |")
        L.append('')
    (out / 'summary.md').write_text('\n'.join(L), encoding='utf-8')


def write_html(out: Path, results):
    e = html.escape
    parts = ['<!doctype html><meta charset="utf-8"><title>Paper-notes check</title><style>'
             'body{font:15px system-ui,sans-serif;margin:24px;max-width:1500px}img{max-width:100%;border:1px solid #ccc}'
             'table{border-collapse:collapse;margin:8px 0 28px}td,th{border:1px solid #ddd;padding:4px 8px;vertical-align:top}'
             '.bad{background:#fdecea}.ok{background:#eaf6ea}.miss{color:#a00}</style>',
             '<h1>Paper-notes scalability check</h1><p>Green boxes matched a reference line, red boxes did not. '
             'Same crops for every reader.</p>']
    for key, res in results.items():
        readers = list(res['readers'])
        parts.append(f"<h2>{e(res['photo'])} · {e(res['condition'])} · page {res['page']}</h2>"
                     f"<img src='overlays/{e(key)}.jpg'>")
        parts.append('<table><tr><th>#</th><th>reference</th>' + ''.join(f'<th>{e(r)}</th>' for r in readers) + '</tr>')
        for j, l0 in enumerate(res['readers'][readers[0]]['lines']):
            cells = []
            for r in readers:
                l = res['readers'][r]['lines'][j]
                if not l['found']:
                    cells.append("<td class='miss'>not detected</td>")
                else:
                    cls = 'ok' if l['cer'] == 0 else 'bad' if l['cer'] > 0.2 else ''
                    cells.append(f"<td class='{cls}'>{e(l['pred'])}<br><small>{fmt(l['cer'])}%</small></td>")
            parts.append(f"<tr><td>{j + 1}</td><td>{e(l0['ref'])}</td>{''.join(cells)}</tr>")
        parts.append('</table>')
    (out / 'review.html').write_text('\n'.join(parts), encoding='utf-8')


# ----------------------------------------------------------------------------- commands
def cmd_run(a):
    out = Path(a.out).resolve()
    refs = load_refs()
    readers = [r.strip() for r in a.readers.split(',') if r.strip()]
    for r in readers:
        if r not in READERS or not (READERS[r][1] / 'inference.pdiparams').is_file():
            sys.exit(f'Reader {r!r} missing: {READERS[r][1] if r in READERS else "unknown name"}')
    photos = [p for p in photos_in(Path(a.photos).resolve()) if page_of(p.name) in refs]
    if not photos:
        sys.exit(f'No photos named page1/page2/page3 in {a.photos}')
    budget = Budget(a.max_seconds)
    index = jload(out / 'index.json', {})
    jobs = []
    for p in photos:
        up = upright(p, out)
        conds = {'original': (up, None)}
        if 'app_prepared' in a.conditions:
            if not budget.left():
                break
            img, info = app_prepare(up, out)
            conds['app_prepared'] = (img, info)
        for c, (img, info) in conds.items():
            if c not in a.conditions:
                continue
            if not budget.left():
                break
            key = f'{p.stem}__{c}'
            det = detect(img, key, out)
            index[key] = dict(photo=p.name, condition=c, page=page_of(p.name), readers=readers, preparation=info)
            jobs.append((key, det))
    jsave(out / 'index.json', index)
    done = True
    for r in readers:
        if not budget.left() or not read_all(r, jobs, out, budget):
            done = False
            break
    missing = [k for k in index for r in readers if not (out / 'cache' / f'{k}__{r}.json').is_file()]
    if done and not missing and len(jobs) == len(photos) * len(a.conditions):
        report(out)
        print('DONE')
    else:
        print(f'PARTIAL: {len(missing)} reader passes left; run the same command again to continue.')


def recorded_file(name: str):
    if name.startswith('reader__'):
        return ASSIGNMENT / 'board2notes_reader_dataset_20261105' / name[len('reader__'):].replace('__', '/')
    if name.startswith('iam__'):
        return ASSIGNMENT / 'iam_line_paddle' / name[len('iam__'):].replace('__', '/')
    if name.startswith('his__'):
        return ASSIGNMENT / 'whiteboard-notes-v1_75619db2' / name[len('his__'):].replace('__', '/')
    return None


def cmd_check(a):
    """Plumbing: re-read crops that have recorded RunPod/CPU predictions and compare text."""
    out = Path(a.out)
    wb_iam = ('whiteboard val', 'IAM val')
    sources = {'downloaded': ('results_versions_2x2', 'pretrained', wb_iam),
               'whiteboard': ('results_versions_2x2', 'V2', wb_iam),
               'paper': ('results_versions_2x2', 'V1b', wb_iam),
               'stagewise': ('results_strategies_s0s3', 'S1', wb_iam),
               'final': ('results_server_his_20261110', 'S1his', ('his val', 'his test'))}
    budget = Budget(a.max_seconds)
    cache = out / 'check'
    cache.mkdir(parents=True, exist_ok=True)
    import cv2
    import ocr_backend as ob
    summary = jload(cache / 'summary.json', {})
    for reader, (exp, model_name, sets) in sources.items():
        if reader in summary or not budget.left():
            continue
        rows = [r for r in csv.DictReader(open(ASSIGNMENT / exp / 'results' / 'predictions.csv', encoding='utf-8'))
                if r['model'] == model_name and r['set'] in sets]
        picked = []
        for s in sets:
            picked += sorted((r for r in rows if r['set'] == s), key=lambda r: r['file'])[:a.n]
        model = load_reader(reader)
        same, diffs, n = 0, [], 0
        for r in picked:
            f = recorded_file(r['file'])
            if not f or not f.is_file():
                continue
            img = ob.load_image(f)
            text = model.read([img])[0]['text']
            n += 1
            same += text == r['pred']
            diffs.append(cer(text, r['label']) - float(r['cer']))
            if not budget.left():
                break
        summary[reader] = dict(model=model_name, experiment=exp, sets=list(sets), crops=n, identical_text=same,
                               mean_cer_difference=round(sum(diffs) / max(1, len(diffs)), 5),
                               paddle=model.identity()['paddle'], platform=f'{platform.system()} {platform.machine()}')
        jsave(cache / 'summary.json', summary)
        print(reader, summary[reader])
        del model
    left = [r for r in sources if r not in summary]
    print('CHECK DONE' if not left else f'CHECK PARTIAL, left: {left}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--photos', default=str(WORK / 'photos'))
    r.add_argument('--out', default=str(WORK / 'outputs'))
    r.add_argument('--readers', default=','.join(DEFAULT_READERS))
    r.add_argument('--conditions', default=','.join(CONDITIONS), type=lambda s: s.split(','))
    r.add_argument('--max-seconds', type=float, default=None)
    p = sub.add_parser('report')
    p.add_argument('--out', default=str(WORK / 'outputs'))
    c = sub.add_parser('check')
    c.add_argument('--n', type=int, default=20)
    c.add_argument('--out', default=str(WORK / 'plumbing_check'))
    c.add_argument('--max-seconds', type=float, default=None)
    a = ap.parse_args()
    {'run': cmd_run, 'report': lambda a: report(Path(a.out).resolve()), 'check': cmd_check}[a.cmd](a)


if __name__ == '__main__':
    main()
