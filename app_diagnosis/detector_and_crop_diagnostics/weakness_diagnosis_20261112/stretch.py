import csv, math, os, sys, collections
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, '/home/claude/fresh/pkg/b2n_ocr'); os.environ['B2N_PACKAGE_ROOT'] = '/home/claude/fresh/pkg'
import ocr_backend as ob
U = Path("/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment")
MODELS = {m: U / f'results_strategies_s0s3/results/models/PP-OCRv5_server_rec_board2notes_{m}' for m in ('S0', 'S0s2')}
MODELS.update({m: U / f'results_guide_g0/results/models/PP-OCRv5_server_rec_board2notes_{m}' for m in ('G0', 'G0s2')})
ob.MODELS.update(MODELS)
man = list(csv.DictReader(open('nw/b2n_data/newwriter/manifest.csv', encoding='utf-8')))
saved = {}
for f in ('results_strategies_s0s3', 'results_guide_g0'):
    for r in csv.DictReader(open(f'/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment/{f}/results/predictions.csv', encoding='utf-8')) if Path(f'/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment/{f}/results/predictions.csv').exists() else []:
        if r['set'] == 'new writers': saved[(r['model'], r['file'])] = r['pred']
norm = lambda s: ' '.join(s.lower().split())
def lev(a, b):
    p = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        c = [i]
        for j, cb in enumerate(b, 1): c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (ca != cb)))
        p = c
    return p[-1]
def need(t):
    t = ' '.join(t.split()); return len(t) + sum(1 for a, b in zip(t, t[1:]) if a == b)
imgs = [cv2.imread(f"nw/b2n_data/newwriter/{r['path']}") for r in man]
fit = [math.ceil(math.ceil(64 * im.shape[1] / im.shape[0]) / 8) / need(r['label']) for r, im in zip(man, imgs)]
rows = []
for m in MODELS:
    rec = ob.RecognitionModel(m, threads=8)
    for s in (1.0, 1.5, 2.0):
        ims = [cv2.resize(im, (int(round(im.shape[1] * s)), im.shape[0]), interpolation=cv2.INTER_CUBIC) if s != 1 else im for im in imgs]
        for r, x, fr in zip(man, rec.read(ims), fit):
            a = norm(r['label']); rows.append(dict(model=m, s=s, file=r['path'], writer=r['writer'], fit=fr, cer=lev(a, norm(x['text'])) / len(a), pred=x['text']))
    print(m, 'done', flush=True)
import json; json.dump(rows, open('stretch_rows.json', 'w'))
