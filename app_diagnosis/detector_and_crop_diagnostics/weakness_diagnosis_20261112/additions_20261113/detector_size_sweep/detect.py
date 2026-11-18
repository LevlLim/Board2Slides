import json, time, collections
from common import *
refs = json.load(open('refs.json'))
photos = collections.OrderedDict()
FRESH_DE = {r['pid'] for r in refs if r['group'] == 'fresh sentences'}
for r in refs: photos.setdefault(r['pid'], (r['set'], r['photo']))
det = tl.Detector(ob.DETECTOR_NAME, det_args={'thresh': .20, 'box_thresh': .40, 'unclip_ratio': 1.2}, model_dir=ob.DETECTOR_DIR, runtime=ob.DETECTOR_RUNTIME)
out = json.load(open('det.json')) if os.path.exists('det.json') else {}
for pid, (s, path) in photos.items():
    zs = ZOOM[s] if not (s == 'fresh' and pid not in FRESH_DE) else []
    for k in [1] + zs:
        vid = f'{pid}@{k}'
        if vid in out: continue
        img, sc, off = variant(load(path), k)
        L = max(img.shape[:2]); rec = dict(long=L, scale=sc, off=off, cfg={}, info=None)
        t = time.time()
        b, info = det.detect(img, unclip=1.2); b, _ = tl.group_into_lines(b)
        rec['cfg']['app'] = [np.asarray(p).astype(float).tolist() for p in b]; rec['info'] = info
        for S in SIZES:
            key = str(S) if S < L else 'full'
            if key in rec['cfg']: continue
            b = det.boxes(img, limit=S if S < L else None, unclip=1.2); b, _ = tl.group_into_lines(b)
            rec['cfg'][key] = [np.asarray(p).astype(float).tolist() for p in b]
        out[vid] = rec
        json.dump(out, open('det.json', 'w'))
        print(vid, L, info['path'][:40], {c: len(v) for c, v in rec['cfg'].items()}, f'{time.time()-t:.0f}s', flush=True)
print('DETECT DONE', flush=True)
