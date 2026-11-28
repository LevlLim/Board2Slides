import sys, json, os
from pathlib import Path
import numpy as np, cv2
src = open('refine4.py').read()
head = src[:src.index('allq = {}')]
exec(head)
S = Path('/tmp/claude-0/-home-claude/02f3df2f-1dbb-533d-b187-71169cf31ef0/scratchpad/unf')
q3 = json.load(open(S/'quads_auto3.json')); q4 = json.load(open(S/'quads_auto4.json'))
OUTF = S/'fixA'; OUTF.mkdir(exist_ok=True)
for f in OUTF.glob('*'): f.unlink()
def Q(src, ph, i): return np.float32((q3 if src == 3 else q4)[ph][i - 1])
jobs = {  # photo: (food, weights) sources, and which lines to predict
 'IMG_5100': [((4, 4), (4, 5), 'ghosts')], 'IMG_5101': [((4, 4), (4, 5), 'ghosts')],
 'IMG_5103': [((4, 4), (4, 5), 'ghosts')], 'IMG_5104': [((4, 3), (4, 4), 'ghosts')],
 'IMG_5097': [((3, 3), (3, 4), 'weights'), ((3, 3), (3, 4), 'ghosts')], 'IMG_5102': [((4, 4), (4, 5), 'ghosts')]}
out = {}
for ph, js in jobs.items():
    img = ob.load_image(PH / f'{ph}.HEIC')
    for (sa, sb, name) in js:
        a, b = Q(*sa[:1], ph, sa[1]) if False else Q(sa[0], ph, sa[1]), Q(sb[0], ph, sb[1])
        step = b.mean(0) - a.mean(0)
        k = 1 if name == 'ghosts' and ph != 'IMG_5097' else (1 if name == 'weights' else 2)
        base = b if ph != 'IMG_5097' else b
        pred = base + step * k
        c = pred.mean(0); pred = (pred - c) * np.float32([1.4, 1.1]) + c
        rs = refine(img, pred)
        rs = sorted(rs, key=lambda q: np.linalg.norm(q.mean(0) - pred.mean(0)))
        if rs:
            out[f'{ph}:{name}'] = rs[0].tolist()
            cv2.imwrite(str(OUTF / f'{ph}_{name}.png'), tl.quad_crop(img, np.float32(rs[0])))
        print(ph, name, len(rs))
json.dump(out, open(S/'quads_fixA.json', 'w'))
