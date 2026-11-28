import json, collections, time
from common import *
U = Path("/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment")
READER = os.environ.get('READER', 'S1')
ob.MODELS.update({'S1': U/'results_strategies_s0s3/results/models/PP-OCRv5_server_rec_board2notes_S1',
                  'S1his': U/'results_server_his_20261110/results/models/PP-OCRv5_server_rec'})
CFGS = os.environ.get('CFGS', 'app,640,960,1280,1600,2400,full').split(',')
refs = json.load(open('refs.json')); det = json.load(open(os.environ.get('DET_FILE', 'det.json')))
byp = collections.defaultdict(list)
for r in refs: byp[r['pid']].append(r)
norm = lambda s: ' '.join(s.lower().split())
def lev(a, b):
    p = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        c = [i]
        for j, cb in enumerate(b, 1): c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (ca != cb)))
        p = c
    return p[-1]
def sub_lev(a, b):   # best match of a anywhere inside b (extra text around it is not counted)
    p = [0] * (len(b) + 1)
    for i, ca in enumerate(a, 1):
        c = [i]
        for j, cb in enumerate(b, 1): c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (ca != cb)))
        p = c
    return min(p)
def cfg_boxes(rec, c):
    if c == 'full': return rec['cfg'].get('full', rec['cfg'].get('4000'))
    return rec['cfg'].get(c, cfg_boxes(rec, 'full') if c != 'app' else None)
def det_scale(rec, c):
    L = rec['long']
    if c not in ('app', 'full') and not c.isdigit():
        return 1.0
    if c == 'app':
        return min(1, 1600 / L) if 'kept' in rec['info']['path'] else min(1, 4000 / L)
    return min(1, (4000 if c == 'full' else int(c)) / L)
rcache = json.load(open(f'reads_{READER}.json')) if os.path.exists(f'reads_{READER}.json') else {}
rec_model = ob.RecognitionModel(READER, threads=2)
rows = []; t0 = time.time()
for vid, rec in list(det.items())[:int(os.environ.get('LIMIT_V', 10**6))]:
    pid, k = vid.rsplit('@', 1); k = int(k)
    img = None
    R = []
    for r in byp[pid]:
        q = np.asarray(r['quad'], float) * rec['scale'] + np.asarray(rec['off'], float)
        R.append((r, q))
    L = rec['long']; S = 1100 / L
    h0, w0 = load(R[0][0]['photo']).shape[:2]; H, W = round(h0 * S) + 2, round(w0 * S) + 2
    def mask(q):
        m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.round(np.asarray(q) * S).astype(np.int32)], 1); return m
    rm = [mask(q) for _, q in R]
    yr = [(q[:, 1].min(), q[:, 1].max()) for _, q in R]
    for c in CFGS:
        boxes = cfg_boxes(rec, c)
        assign = collections.defaultdict(list); stacked = set(); extras = 0
        for bi, b in enumerate(boxes):
            bm = mask(b); a = bm.sum()
            if a == 0: continue
            ov = [(bm & m).sum() / a for m in rm]; cov = [(bm & m).sum() / max(m.sum(), 1) for m in rm]
            hits = [i for i in range(len(R)) if ov[i] > .25 or cov[i] > .25]
            if not hits: extras += 1; continue
            best = max(hits, key=lambda i: ov[i]); assign[best].append(bi)
            for i in hits:
                for j in hits:
                    if i < j:
                        lo = max(yr[i][0], yr[j][0]); hi = min(yr[i][1], yr[j][1])
                        if hi - lo < .5 * min(yr[i][1] - yr[i][0], yr[j][1] - yr[j][0]): stacked.update((i, j))
        need = sorted({bi for v in assign.values() for bi in v})
        keys = {bi: f"{vid}|" + ','.join(f'{v:.0f}' for v in np.asarray(boxes[bi]).ravel()) for bi in need}
        todo = [bi for bi in need if keys[bi] not in rcache]
        if todo:
            if img is None:
                img, _, _ = variant(load(R[0][0]['photo']), k)
            crops = [tl.quad_crop(img, np.asarray(boxes[bi], np.float32)) for bi in todo]
            for bi, x in zip(todo, rec_model.read(crops)): rcache[keys[bi]] = x['text']
        for i, (r, q) in enumerate(R):
            bs = sorted(assign.get(i, []), key=lambda bi: np.mean(np.asarray(boxes[bi])[:, 0]))
            txt = ' '.join(rcache[keys[bi]] for bi in bs); a = norm(r['text'])
            rows.append(dict(set=r['set'], group=r['group'] + (f' (zoomed out x{k})' if k > 1 else ''), vid=vid, zoom=k, cfg=c, id=r['id'],
                             writing=r['writing'], line_h_photo=line_h(q), line_h_det=line_h(q) * det_scale(rec, c),
                             kind='missed' if not bs else ('one box' if len(bs) == 1 else 'split'), stacked_merge=i in stacked,
                             n_boxes=len(bs), pred=txt, cer=sub_lev(a, norm(txt)) / len(a), cer_strict=lev(a, norm(txt)) / len(a),
                             extras=extras if i == 0 else 0, app_path=rec['info']['path']))
    json.dump(rcache, open(f'reads_{READER}.json', 'w'), ensure_ascii=False)
    print(vid, f'{time.time() - t0:.0f}s', flush=True)
json.dump(rows, open(os.environ.get('ROWS_OUT', f'rows_{READER}.json'), 'w'), ensure_ascii=False)
print('SCORE DONE', flush=True)
