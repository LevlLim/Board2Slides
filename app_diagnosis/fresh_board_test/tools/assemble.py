import json, csv, os, sys, hashlib
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path('pkg/b2n_ocr').resolve())); os.environ['B2N_PACKAGE_ROOT'] = str(Path('pkg').resolve())
import ocr_backend as ob
from preprocessing import text_lines as tl
S = Path('/tmp/claude-0/-home-claude/02f3df2f-1dbb-533d-b187-71169cf31ef0/scratchpad/unf')
PH = Path("/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment/1111_unfamiliar")
q3 = json.load(open(S/'quads_auto3.json')); q4 = json.load(open(S/'quads_auto4.json')); qf = json.load(open(S/'quads_fixA.json'))
det = json.load(open(S/'det2/boxes.json'))
LAB = {'A': ['bookkeeper', 'parallel', 'access', 'food', 'weights', 'ghosts'],
       'B': ['book keeper', 'letter', 'gradient'], 'C': ['earth', 'generalization'],
       'D': ['The model should preserve repeated letters in words such as Aiken and Xavier.',
             'The white board contains a mixture of short words, long phrases and numbers',
             'A careful crop keeps every hand written character inside the recognition region'],
       'E': ['Different writers use different spacing, pressure, height on the same board.',
             'Please leave enough space between lines, so the detector does not merge neighbouring sentences']}
BOARD = {**{f'IMG_{n}': 'A' for n in range(5095, 5109)}, **{f'IMG_{n}': 'B' for n in range(5109, 5113)},
         **{f'IMG_{n}': 'C' for n in range(5113, 5116)}, **{f'IMG_{n}': 'D' for n in range(5116, 5125)},
         **{f'IMG_{n}': 'E' for n in range(5125, 5139)}}
v3, v4 = lambda i: ('v3', i), lambda i: ('v4', i)
SEL = {  # photo -> one entry per reference line: (source, index) or None (excluded, reason in EXCL)
 'IMG_5095': [v4(1), v3(2), v3(3), v3(4), v3(5), v3(6)],
 'IMG_5096': [v4(1), v4(2), v4(3), v4(4), v4(5), v4(6)],
 'IMG_5097': [v3(1), v3(2), v3(3), v3(4), None, None],
 'IMG_5098': [v3(i) for i in range(1, 7)], 'IMG_5099': [v3(i) for i in range(1, 7)],
 'IMG_5100': [v4(1), v4(2), v4(3), v4(4), v4(5), ('fix', 'ghosts')],
 'IMG_5101': [v4(1), v4(2), v4(3), v4(4), v4(5), ('fix', 'ghosts')],
 'IMG_5102': [v4(i) for i in range(1, 6)] + [('fix', 'ghosts')],
 'IMG_5103': [v4(1), v4(2), v3(3), v4(4), v4(5), ('fix', 'ghosts')],
 'IMG_5104': [v4(1), v4(2), v3(3), v4(3), v4(4), ('det', 0)],
 'IMG_5105': [v4(i) for i in range(1, 7)], 'IMG_5106': [v4(i) for i in range(1, 7)],
 'IMG_5107': [v3(i) for i in range(1, 7)], 'IMG_5108': [v3(i) for i in range(1, 7)],
 'IMG_5109': [v4(1), v4(2), v4(3)],
 **{f'IMG_{n}': [v3(1), v3(2), v3(3)] for n in (5110, 5111, 5112)},
 **{f'IMG_{n}': [v3(1), v3(2)] for n in (5113, 5114, 5115)},
 **{f'IMG_{n}': [v3(1), v3(2), v3(3)] for n in (5116, 5117, 5118, 5122, 5123, 5124)},
 **{f'IMG_{n}': [v3(1), v3(2), None] for n in (5119, 5120, 5121)},
 **{f'IMG_{n}': [v3(1), v3(2)] for n in range(5125, 5139)}, 'IMG_5132': [v3(1), v3(3)],
}
EXCL = {('IMG_5097', 5): 'weights and ghosts overlap in this tilted photo; no single-line crop possible',
        ('IMG_5097', 6): 'weights and ghosts overlap in this tilted photo; no single-line crop possible',
        **{(f'IMG_{n}', 3): 'line runs off the photo edge (ends mid-word)' for n in (5119, 5120, 5121)}}
EXTEND_LEFT = {(f'IMG_{n}', 3): 0.6 for n in (5122, 5123, 5124)}
OUT = S / 'board2notes_freshwriter_test_20261111'; (OUT / 'images').mkdir(parents=True, exist_ok=True)
rows = []
for ph, sel in SEL.items():
    b = BOARD[ph]; assert len(sel) == len(LAB[b]), ph
    img = ob.load_image(PH / f'{ph}.HEIC')
    src_sha = hashlib.sha256((PH / f'{ph}.HEIC').read_bytes()).hexdigest()
    for li, (s, label) in enumerate(zip(sel, LAB[b]), 1):
        if s is None:
            rows.append(dict(id=f'{b}_{ph[4:]}_{li}', file='', board=b, photo=ph, line=li, writing=f'{b}{li}', label=label,
                             included='no', reason=EXCL[(ph, li)], quad='', source='', photo_sha256=src_sha)); continue
        if s[0] == 'v3': q = q3[ph][s[1] - 1]
        elif s[0] == 'v4': q = q4[ph][s[1] - 1]
        elif s[0] == 'fix': q = qf[f'{ph}:{s[1]}']
        else:
            cands = det[ph][0]; q = max(cands, key=lambda p: np.mean(np.asarray(p)[:, 1]))   # lowest detector box
        q = tl.order_tl_tr_br_bl(np.float32(q))
        if (ph, li) in EXTEND_LEFT:   # faint first letter missed by the ink check; checked by eye
            ux = (q[1] - q[0]) / np.linalg.norm(q[1] - q[0]); d = EXTEND_LEFT[(ph, li)] * tl.line_height(q)
            q[0] -= ux * d; q[3] -= ux * d
        if (b, li) in (('D', 1), ('E', 1)):   # these lines end with a full stop that the ink check tends to drop
            ux = (q[1] - q[0]) / np.linalg.norm(q[1] - q[0]); d = 0.6 * tl.line_height(q)
            q[1] += ux * d; q[2] += ux * d
        crop = tl.quad_crop(img, q)
        fid = f'{b}_{ph[4:]}_{li}'; cv2.imwrite(str(OUT / 'images' / f'{fid}.png'), crop)
        rows.append(dict(id=fid, file=f'images/{fid}.png', board=b, photo=ph, line=li, writing=f'{b}{li}', label=label,
                         included='yes', reason='', quad=json.dumps(np.round(q, 1).tolist()), source=s[0], photo_sha256=src_sha,
                         width=crop.shape[1], height=crop.shape[0]))
with open(OUT / 'manifest.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=['id', 'file', 'board', 'photo', 'line', 'writing', 'label', 'included', 'reason', 'width', 'height', 'source', 'quad', 'photo_sha256'])
    w.writeheader(); [w.writerow(r) for r in rows]
inc = [r for r in rows if r['included'] == 'yes']
print('crops', len(inc), 'excluded', len(rows) - len(inc))
from collections import Counter; print(Counter(r['board'] for r in inc)); print(Counter(r['writing'] for r in inc))
