"""JH full photos: unchanged app pipeline vs dataset crops. Reads saved outputs only (no inference).
Run from the Assignment folder:  python3 claude_jh_fullphoto_20261109/analyse_jh.py"""
import csv, json, math, random, collections, hashlib
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

A = Path(__file__).resolve().parent.parent; O = A / 'claude_jh_fullphoto_20261109'; RUN = A / 'whiteboard-ocr/outputs/jh_full_photos_app'
V = O / 'visuals'; V.mkdir(exist_ok=True)
P = json.loads((O / 'PROTOCOL.json').read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(O / 'dataset_line_quads.json') == P['hashes']['line_quads'] and sha(O / 'dataset_crop_placement.json') == P['hashes']['placement']
run = json.loads((RUN / 'run.json').read_text())
assert run['labels_used'] is False and run['no_training'] and not run['app_modified']
LQ = json.loads((O / 'dataset_line_quads.json').read_text())
DSR = json.loads((RUN / 'dataset_crop_readings.json').read_text())
LAB = {r['id']: r['label'] for r in csv.DictReader(open(A / 'board2notes_newwriter_test_20261107/manifest.csv'))}
photos = {int(p.stem.split('_')[1]): json.loads(p.read_text()) for p in RUN.glob('Photo_*.json')}
norm = lambda s: ' '.join(str(s).lower().split())
def font(size, mono=False):
    candidates = (
        ['/System/Library/Fonts/Menlo.ttc', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf']
        if mono else
        ['/System/Library/Fonts/Supplemental/Arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)

F = lambda size: font(size)
FM = lambda size: font(size, mono=True)


def ops(a, b):
    a, b = norm(a), norm(b); n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): D[i][0] = i
    for j in range(m + 1): D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i-1][j] + 1, D[i][j-1] + 1, D[i-1][j-1] + (a[i-1] != b[j-1]))
    i, j, S, Dl, I = n, m, 0, 0, 0
    while i or j:
        if i and j and D[i][j] == D[i-1][j-1] + (a[i-1] != b[j-1]): S += a[i-1] != b[j-1]; i -= 1; j -= 1
        elif i and D[i][j] == D[i-1][j] + 1: Dl += 1; i -= 1
        else: I += 1; j -= 1
    return dict(cer=D[n][m] / max(1, n), S=S, D=Dl, I=I, errors=D[n][m], chars=n)


def inter(a, b):
    a = cv2.convexHull(np.float32(a)); b = cv2.convexHull(np.float32(b)); i, _ = cv2.intersectConvexConvex(a, b)
    return float(i), float(cv2.contourArea(a)), float(cv2.contourArea(b))


def components(lines, regions):
    """lines: [(id, core_quad)], regions: [(index, box)] -> list of components (protocol link rule)"""
    n, m = len(lines), len(regions); nb = {k: set() for k in range(n + m)}; cover = {}
    for i, (lid, core) in enumerate(lines):
        for j, (rid, box) in enumerate(regions):
            it, ac, ar = inter(core, box)
            cover[(i, j)] = it / max(ac, 1e-9)
            if it / max(ac, 1e-9) >= 0.2 or it / max(ar, 1e-9) >= 0.5:
                nb[i].add(n + j); nb[n + j].add(i)
    seen, out = set(), []
    for k in range(n + m):
        if k in seen: continue
        st, comp = [k], set()
        while st:
            x = st.pop()
            if x in comp: continue
            comp.add(x); st.extend(nb[x] - comp)
        seen |= comp
        li = sorted(x for x in comp if x < n); ri = sorted(x - n for x in comp if x >= n)
        kind = ('extra_region' if not li else 'missed_line' if not ri else 'one_to_one' if len(li) == len(ri) == 1 else
                'merge' if len(ri) == 1 else 'split' if len(li) == 1 else 'many_to_many')
        out.append(dict(kind=kind, lines=[lines[i][0] for i in li], regions=[regions[j][0] for j in ri],
                        core_cover=[round(sum(cover[(i, j)] for j in ri), 3) for i in li]))
    return out


def ink_mask(gray):
    h, w = gray.shape; k = max(15, (min(h, w) // 3) | 1); k = min(k, 255) | 1
    diff = cv2.subtract(cv2.medianBlur(gray, k), gray); thr = max(10.0, 0.20 * float(np.percentile(diff, 99.5)))
    m = cv2.morphologyEx((diff > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8); keep = [i for i in range(1, n) if st[i, 4] >= max(6, 0.00005 * h * w)]
    return np.isin(lab, keep)


def profile_angle(gray):
    ys, xs = np.nonzero(ink_mask(gray))
    if len(xs) < 50: return None
    x = xs - xs.mean(); y = ys - ys.mean(); best, ba = -1, 0.0
    for a in np.arange(-12, 12.0001, 0.1):
        t = math.radians(a); yr = -x * math.sin(t) + y * math.cos(t); h = np.bincount((yr - yr.min()).astype(int))
        s = float((h.astype(float) ** 2).sum())
        if s > best: best, ba = s, float(a)
    return round(ba, 1)


lines_by_photo = collections.defaultdict(list)
for l in LQ: lines_by_photo[l['photo']].append((l['id'], l['core_quad']))
res = dict(per_photo={}, one_to_one=[], other=[], extra=[])
for pid in sorted(photos):
    rec = photos[pid]; L = lines_by_photo[pid]
    g = components(L, [(r['index'], r['box']) for r in rec['grouped']])
    raw = components(L, [(b['raw_id'], b['box']) for b in rec['raw']])
    cnt = lambda cs: dict(collections.Counter(c['kind'] for c in cs))
    res['per_photo'][pid] = dict(lines=len(L), raw_boxes=len(rec['raw']), regions=len(rec['grouped']), grouped=cnt(g), raw=cnt(raw),
                                 detect_path=rec['detect_info']['path'], detect_seconds=rec['detect_seconds'])
    regs = {r['index']: r for r in rec['grouped']}
    for c in g:
        if c['kind'] == 'one_to_one':
            lid = c['lines'][0]; r = regs[c['regions'][0]]
            crop = cv2.imread(str(RUN / 'crops' / r['files']['crop']), cv2.IMREAD_GRAYSCALE)
            row = dict(photo=pid, line=lid, label=LAB[lid], region=r['index'], app_text=r['text'], app=ops(LAB[lid], r['text']),
                       app_width64=math.ceil(64 * r['crop_w'] / r['crop_h']), app_crop_wh=[r['crop_w'], r['crop_h']],
                       app_residual_slope=profile_angle(crop), core_cover=c['core_cover'][0], app_files=r['files'], raw_members=r['raw_members'])
            for k in ('original', 'tight', 'straightened'):
                d = DSR[lid][k]; row[k] = dict(text=d['text'], file=d['file'], width64=math.ceil(64 * d['wh'][0] / d['wh'][1]), **ops(LAB[lid], d['text']))
            res['one_to_one'].append(row)
        elif c['kind'] == 'extra_region':
            for i in c['regions']:
                context = ({(2, 5): 'Black board eraser; the saved OCR reading is empty.',
                            (5, 11): 'Projected Windows taskbar: magnifier icon and Search, not handwritten notes.'}
                           .get((pid, i), 'Context not classified.'))
                res['extra'].append(dict(photo=pid, region=i, text=regs[i]['text'], files=regs[i]['files'],
                                         visual_context=context))
        else:
            joined_lab = ' '.join(LAB[x] for x in c['lines']); joined_app = ' '.join(regs[i]['text'] for i in c['regions'])
            res['other'].append(dict(photo=pid, **c, labels=[LAB[x] for x in c['lines']], app_texts=[regs[i]['text'] for i in c['regions']],
                                     joined=ops(joined_lab, joined_app) if c['regions'] else None,
                                     raw_kind=[k['kind'] for k in raw if set(c['lines']) & set(k['lines'])]))
OO = res['one_to_one']


def boot(rows, f, B=4000, seed=0):
    cl = collections.defaultdict(list)
    for r in rows: cl[r['photo']].append(r)
    ks = sorted(cl); rnd = random.Random(seed); v = []
    for _ in range(B):
        s = [r for k in (rnd.choice(ks) for _ in ks) for r in cl[k]]; v.append(sum(f(r) for r in s) / len(s))
    v.sort(); return [v[int(.025 * B)], v[int(.975 * B) - 1]]


summ = dict(n_one_to_one=len(OO), lines=len(LQ))
for k in ('app', 'original', 'tight', 'straightened'):
    summ[f'cer_{k}'] = sum(r[k]['cer'] for r in OO) / len(OO); summ[f'D_{k}'] = sum(r[k]['D'] for r in OO)
    summ[f'persist15_{k}'] = sum(r[k]['cer'] >= .15 for r in OO)
for k in ('original', 'tight', 'straightened'):
    d = lambda r, k=k: r['app']['cer'] - r[k]['cer']
    summ[f'app_minus_{k}'] = dict(diff=sum(d(r) for r in OO) / len(OO), ci_photo_bootstrap=boot(OO, d),
                                  app_better=sum(d(r) < -1e-12 for r in OO), app_worse=sum(d(r) > 1e-12 for r in OO))
summ['median_width64'] = {k: float(np.median([r[k]['width64'] if k != 'app' else r['app_width64'] for r in OO])) for k in ('app', 'original', 'tight', 'straightened')}
summ['app_residual_slope_abs_median'] = float(np.median([abs(r['app_residual_slope']) for r in OO if r['app_residual_slope'] is not None]))
res['summary'] = summ
tot = collections.Counter(); tot_raw = collections.Counter()
for v in res['per_photo'].values(): tot.update(v['grouped']); tot_raw.update(v['raw'])
res['totals'] = dict(grouped=dict(tot), raw=dict(tot_raw))


# ---------------- visuals
COL = {'one_to_one': (22, 163, 74), 'merge': (147, 51, 234), 'split': (37, 99, 235), 'many_to_many': (234, 88, 12), 'extra_region': (220, 38, 38)}


def overlay(pid):
    rec = photos[pid]; L = lines_by_photo[pid]
    base = Image.open(A / 'JH_datasetNov5Sentence' / rec['image']).convert('RGB')
    panels = []
    for name, regs, comps in (('raw detector boxes', [(b['raw_id'], b['box']) for b in rec['raw']], None),
                              ('after grouping (what S1 reads)', [(r['index'], r['box']) for r in rec['grouped']], None)):
        comps = components(L, regs); kind = {r: c['kind'] for c in comps for r in c['regions']}
        missed = {l for c in comps if c['kind'] == 'missed_line' for l in c['lines']}
        im = base.copy(); d = ImageDraw.Draw(im)
        for lid, q in L:
            q = [tuple(v) for v in q]; d.line(q + [q[0]], fill=(220, 38, 38) if lid in missed else (150, 150, 150), width=10 if lid in missed else 3)
        for idx, box in regs:
            q = [tuple(v) for v in box]; c = COL.get(kind.get(idx, 'extra_region'))
            d.line(q + [q[0]], fill=c, width=8); d.text((min(p[0] for p in q), min(p[1] for p in q) - 64), str(idx), fill=c, font=F(56), stroke_width=4, stroke_fill='white')
        im.thumbnail((760, 1100)); cv_ = Image.new('RGB', (im.width, im.height + 30), 'white'); cv_.paste(im, (0, 30))
        ImageDraw.Draw(cv_).text((6, 6), f'Photo {pid}: {name} ({len(regs)})', fill='black', font=F(18)); panels.append(cv_)
    sh = Image.new('RGB', (sum(p.width for p in panels) + 10, max(p.height for p in panels)), 'white'); x = 0
    for p in panels: sh.paste(p, (x, 0)); x += p.width + 10
    f = V / f'overlay_Photo_{pid}.jpg'; sh.save(f, quality=72); return f.name


def strip(pid):
    rec = photos[pid]; L = lines_by_photo[pid]
    comps = components(L, [(r['index'], r['box']) for r in rec['grouped']]); kind = {r: c for c in comps for r in c['regions']}
    W, rh = 1500, 64 + 64
    sh = Image.new('RGB', (W, 34 + rh * len(rec['grouped'])), 'white'); d = ImageDraw.Draw(sh)
    d.text((8, 8), f'Photo {pid}: app regions in reading order - crop as cut (left) and the exact S1 input (right; 64 px high, grey = padding)', fill='black', font=F(18))
    for k, r in enumerate(rec['grouped']):
        y = 34 + k * rh; c = kind.get(r['index']); kd = c['kind'] if c else 'extra_region'
        cr = Image.open(RUN / 'crops' / r['files']['crop']).convert('RGB'); s = 56 / cr.height; cr = cr.resize((max(1, min(430, int(cr.width * s))), 56))
        sh.paste(cr, (40, y)); rv = Image.open(RUN / 'crops' / r['files']['reader']).convert('RGB')
        if rv.width > W - 500: rv = rv.resize((W - 500, max(1, round(64 * (W - 500) / rv.width))))
        sh.paste(rv, (490, y)); d.text((6, y + 18), str(r['index']), fill=COL.get(kd, (0, 0, 0)), font=F(18))
        ref = ' | '.join(LAB[x] for x in c['lines']) if c and c['lines'] else '(no dataset line)'
        cer = ops(ref, r['text'])['cer'] if kd == 'one_to_one' else None
        d.text((40, y + 70), f"S1: {r['text']!r}   dataset label: {ref!r}   [{kd}{', CER %.2f' % cer if cer is not None else ''}]", fill=(30, 30, 30), font=FM(13))
    f = V / f'strip_Photo_{pid}.jpg'; sh.save(f, quality=70); return f.name


def compare_sheets():
    rows = sorted(OO, key=lambda r: r['app']['cer'] - r['tight']['cer']); names = []
    COLW, IH = 370, 50
    for s0 in range(0, len(rows), 12):
        part = rows[s0:s0 + 12]; ph = 20 + IH + 4 + 2 * 16 + 8
        sh = Image.new('RGB', (4 * COLW, 26 + len(part) * (ph + 4)), (225, 225, 225)); d = ImageDraw.Draw(sh)
        d.text((6, 6), 'Same line, four crops read by the same S1 object: dataset original | dataset tightened | dataset straightened | APP (mobile_det -> group -> quad crop)', fill='black', font=F(14))
        for k, r in enumerate(part):
            y = 26 + k * (ph + 4); t = Image.new('RGB', (4 * COLW, ph), 'white'); dd = ImageDraw.Draw(t)
            dd.text((4, 2), f"{r['line']}  label: {r['label']}", fill='black', font=F(13))
            for j, key in enumerate(('original', 'tight', 'straightened', 'app')):
                p = (A / r[key]['file']) if key != 'app' else (RUN / 'crops' / r['app_files']['crop'])
                im = Image.open(p).convert('RGB'); s_ = min(IH / im.height, (COLW - 8) / im.width); im = im.resize((max(1, int(im.width * s_)), max(1, int(im.height * s_))))
                t.paste(im, (j * COLW + 4, 20)); m = r[key]; w64 = r['app_width64'] if key == 'app' else r[key]['width64']
                dd.text((j * COLW + 4, 24 + IH), f"CER {m['cer']:.2f} D {m['D']}  w64 {w64}", fill=(185, 28, 28) if m['cer'] >= .15 else (21, 128, 61), font=FM(12))
                txt = r['app_text'] if key == 'app' else r[key]['text']
                dd.text((j * COLW + 4, 40 + IH), txt[:46], fill=(30, 30, 30), font=FM(11))
            sh.paste(t, (0, y))
        f = V / f'compare_{s0 // 12 + 1}.jpg'; sh.save(f, quality=72); names.append(f.name)
    return names


res['visuals'] = dict(overlays=[overlay(p) for p in sorted(photos)], strips=[strip(p) for p in sorted(photos)], compare=compare_sheets())
json.dump(res, open(O / 'RESULTS.json', 'w'), indent=1, ensure_ascii=False)
s = res['summary']
print('totals grouped', res['totals']['grouped'], '| raw', res['totals']['raw'])
print(f"one-to-one lines {s['n_one_to_one']}/{s['lines']}: CER app {s['cer_app']:.3f} | original {s['cer_original']:.3f} | tight {s['cer_tight']:.3f} | straightened {s['cer_straightened']:.3f}")
for k in ('original', 'tight', 'straightened'):
    x = s[f'app_minus_{k}']; print(f"  app - {k}: {100*x['diff']:+.2f} pts [{100*x['ci_photo_bootstrap'][0]:+.1f},{100*x['ci_photo_bootstrap'][1]:+.1f}] app better {x['app_better']} worse {x['app_worse']}")
print('deletions', {k: s[f'D_{k}'] for k in ('app', 'original', 'tight', 'straightened')}, '| >=15% CER', {k: s[f'persist15_{k}'] for k in ('app', 'original', 'tight', 'straightened')})
print('median width64', s['median_width64'], '| app residual |slope| median', s['app_residual_slope_abs_median'])
for o in res['other']: print('OTHER', o['photo'], o['kind'], o['lines'], o['regions'], o['app_texts'], '| raw:', o['raw_kind'])
for e in res['extra']: print('EXTRA', e['photo'], e['region'], repr(e['text']))
