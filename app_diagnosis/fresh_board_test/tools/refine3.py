import sys, json, os
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path('pkg/b2n_ocr').resolve()))
os.environ['B2N_PACKAGE_ROOT'] = str(Path('pkg').resolve())
import ocr_backend as ob
from preprocessing import text_lines as tl
PH = Path("/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment/1111_unfamiliar")
S = Path('/tmp/claude-0/-home-claude/02f3df2f-1dbb-533d-b187-71169cf31ef0/scratchpad/unf')
boxes = json.load(open(S / 'det2/boxes.json'))
OUT = S / 'crops3'; OUT.mkdir(exist_ok=True)
for f in OUT.glob('*'): f.unlink()
THR = 0.86
def unit(p):
    ux = (p[1]-p[0]) / max(np.linalg.norm(p[1]-p[0]), 1e-6); uy = (p[3]-p[0]) / max(np.linalg.norm(p[3]-p[0]), 1e-6); return ux, uy
def refine(img, quad):
    H, W = img.shape[:2]
    p0 = tl.order_tl_tr_br_bl(np.asarray(quad, np.float32)); lh = tl.line_height(p0); ux, uy = unit(p0)
    gx, gy = 1.0 * lh, 0.45 * lh
    p = (p0 + np.float32([-ux*gx - uy*gy, ux*gx - uy*gy, ux*gx + uy*gy, -ux*gx + uy*gy])).astype(np.float32)
    w = int(round(max(np.linalg.norm(p[0]-p[1]), np.linalg.norm(p[3]-p[2])))); h = int(round(max(np.linalg.norm(p[0]-p[3]), np.linalg.norm(p[1]-p[2]))))
    if w < 8 or h < 8: return []
    M = cv2.getPerspectiveTransform(p, np.float32([[0,0],[w,0],[w,h],[0,h]]))
    try: Mi = np.linalg.inv(M)
    except np.linalg.LinAlgError: return []
    crop = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    o = cv2.perspectiveTransform(p0[None], M)[0]; ox0, ox1 = o[:, 0].min(), o[:, 0].max(); oy0, oy1 = o[:, 1].min(), o[:, 1].max()
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), max(3.0, h / 2)); ink = (np.clip(g / np.maximum(bg, 1), 0, 1.5) < THR).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros_like(ink)
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        if a >= max(4, 0.00003 * w * h) and not (y == 0 or y + hh >= h):
            keep[lab == i] = 1
    prof = keep.mean(1)
    if prof.max() <= 0: return []
    k = max(1, h // 25); prof = np.convolve(prof, np.ones(k)/k, 'same')
    on = list(prof > 0.12 * prof.max()) + [False]; bs = []; s0 = None
    for y, v in enumerate(on):
        if v and s0 is None: s0 = y
        if not v and s0 is not None: bs.append([s0, y]); s0 = None
    bs = [b for b in bs if b[1] - b[0] >= 0.12 * (oy1 - oy0)]
    merged = []
    for b in bs:
        if merged and b[0] - merged[-1][1] < 0.06 * h: merged[-1][1] = b[1]
        else: merged.append(b)
    if not merged: return []
    cy = (oy0 + oy1) / 2
    main = min(merged, key=lambda b: abs((b[0] + b[1]) / 2 - cy))
    sel = [main] + [b for b in merged if b is not main and oy0 <= (b[0]+b[1])/2 <= oy1 and (b[1]-b[0]) >= 0.6*(main[1]-main[0]) and (oy1-oy0) > 1.6*(main[1]-main[0])]
    res = []
    comps = [(st[i], i) for i in range(1, n) if keep[lab == i].any()] if False else None
    cst = [st[i] for i in range(1, n) if st[i][4] >= max(4, 0.00003 * w * h) and not (st[i][1] == 0 or st[i][1] + st[i][3] >= h)]
    for y0, y1 in sel:
        bh = y1 - y0
        mine = [c for c in cst if c[1] < y1 and c[1] + c[3] > y0 and min(c[1] + c[3], y1) - max(c[1], y0) >= 0.4 * min(c[3], bh) and c[3] < 3 * bh]
        if not mine: continue
        # horizontal clusters
        mine.sort(key=lambda c: c[0]); cl = []
        th = bh
        for c in mine:
            if cl and c[0] - cl[-1][1] <= 2.5 * th: cl[-1][1] = max(cl[-1][1], c[0] + c[2]); cl[-1][2].append(c)
            else: cl.append([c[0], c[0] + c[2], [c]])
        cl = [c for c in cl if c[1] >= ox0 - 1.5 * th and c[0] <= ox1 + 1.5 * th]
        if not cl: continue
        # keep the run of clusters that contains the box centre, joined while gaps are small
        x0, x1 = cl[0][0], cl[-1][1]
        cs = [c for k in cl for c in k[2]]
        yy0 = min(c[1] for c in cs); yy1 = max(c[1] + c[3] for c in cs)
        th = max(1, yy1 - yy0)
        if ox1 - x1 > 1.5 * bh: x1 = ox1
        mx, my = 0.25 * bh, 0.12 * bh
        q = np.float32([[max(0, x0-mx), max(0, yy0-my)], [min(w, x1+mx), max(0, yy0-my)], [min(w, x1+mx), min(h, yy1+my)], [max(0, x0-mx), min(h, yy1+my)]])
        res.append(cv2.perspectiveTransform(q[None], Mi)[0])
    return res
allq = {}
for f in sorted(PH.glob('*.HEIC')):
    img = ob.load_image(f); qs = []
    for quad in boxes[f.stem][0]:
        qs += refine(img, quad)
    qs = [q for q in qs if tl.line_width(q) > 1.2 * tl.line_height(q)]
    def ov(a, b):
        A = np.zeros(img.shape[:2], np.uint8); B = A.copy()
        cv2.fillPoly(A, [a.astype(np.int32)], 1); cv2.fillPoly(B, [b.astype(np.int32)], 1)
        return (A & B).sum() / max(1, min(A.sum(), B.sum()))
    ded = []
    for q in sorted(qs, key=lambda q: -cv2.contourArea(q)):
        if all(ov(q, d) < 0.5 for d in ded): ded.append(q)
    ded.sort(key=lambda q: float(q[:, 1].mean()))
    allq[f.stem] = [q.tolist() for q in ded]
    for i, q in enumerate(ded):
        cv2.imwrite(str(OUT / f'{f.stem}_{i+1:02d}.png'), tl.quad_crop(img, np.float32(q)))
    print(f.stem, len(ded), flush=True)
json.dump(allq, open(S / 'quads_auto3.json', 'w'))
