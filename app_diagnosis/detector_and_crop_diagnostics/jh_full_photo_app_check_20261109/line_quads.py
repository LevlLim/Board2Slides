"""Oriented line quads in photo coordinates for the 59 JH dataset lines, from images only:
straightened tight box (straighten/crops_straighten.json) -> back through the deskew rotation -> placement affine."""
import json, os, math
from pathlib import Path
import cv2, numpy as np
A = Path(os.environ['A']); O = A/'claude_jh_fullphoto_20261109'; D = A/'claude_margin_diagnostic_20261109'
P = {x['id']: x for x in json.load(open(O/'dataset_crop_placement.json'))}
B = {x['name']: x for x in json.load(open(D/'boxes_final.json'))}
S = {x['name']: x for x in json.load(open(D/'straighten/crops_straighten.json'))}
out = []
for n, p in P.items():
    b, s = B[n], S[n]
    assert b['sha256'] == p['crop_sha256'], n     # the straightened source crop is the same file that was placed
    src = cv2.imread(str(A/b['src'])); SH, SW = src.shape[:2]
    if s['processed']:
        bx0, by0, bx1, by1 = b['box']; M = cv2.getRotationMatrix2D(((bx0+bx1)/2, (by0+by1)/2), s['angle_deg'], 1.0)
        corners = np.array([[0,0,1],[SW,0,1],[SW,SH,1],[0,SH,1]], float) @ M.T; M[:, 2] -= corners.min(0)
        x0, y0, x1, y1 = s['R_box_in_rotated_src']; Minv = cv2.invertAffineTransform(M)
        q_src = cv2.transform(np.float32([[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]]), Minv)[0]
    else:
        x0, y0, x1, y1 = b['box']; q_src = np.float32([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    Mp = np.array(p_aff := None) if False else None
    # placement affine from the stored quad: crop corners -> photo quad (similarity)
    w, h = p['crop_wh']; Mpl, _ = cv2.estimateAffinePartial2D(np.float32([[0,0],[w,0],[w,h],[0,h]]), np.float32(p['quad']))
    q_ph = cv2.transform(q_src[None], Mpl)[0]
    out.append(dict(id=n, photo=p['photo'], line_quad=q_ph.round(1).tolist(), upright_crop_quad=p['quad'],
                    deskew_angle=s.get('angle_deg', 0.0), source='straightened tight box (images only)' if s['processed'] else 'tight box (not processed)'))
json.dump(out, open(O/'dataset_line_quads.json', 'w'), indent=1)
# overlap between neighbouring line quads vs upright crop boxes
def inter(a, b):
    a = cv2.convexHull(np.float32(a)); b = cv2.convexHull(np.float32(b)); i, _ = cv2.intersectConvexConvex(a, b)
    return i / max(min(cv2.contourArea(a), cv2.contourArea(b)), 1e-9)
for key in ('upright_crop_quad', 'line_quad'):
    v = [inter(a[key], b[key]) for i, a in enumerate(out) for b in out[i+1:] if a['photo'] == b['photo']]
    print(key, 'pairs with intersection/min-area >= 0.5:', sum(x >= .5 for x in v), '| >= 0.2:', sum(x >= .2 for x in v), '| max', round(max(v), 2))

# ---- core quads: the dense text band of each line (contiguous band of rows around the ink-profile peak (>= 25% of peak) in the straightened crop)
def ink_mask(gray):
    h, w = gray.shape; k = max(15, (min(h, w) // 3) | 1); k = min(k, 255) | 1
    diff = cv2.subtract(cv2.medianBlur(gray, k), gray); thr = max(10.0, 0.20 * float(np.percentile(diff, 99.5)))
    m = cv2.morphologyEx((diff > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8); keep = [i for i in range(1, n) if st[i, 4] >= max(6, 0.00005 * h * w)]
    return np.isin(lab, keep)
for rec in out:
    n = rec['id']; s = S[n]; b = B[n]
    R = cv2.imread(str(D/s['R_file']), cv2.IMREAD_GRAYSCALE); ys, xs = np.nonzero(ink_mask(R))
    prof = np.bincount(ys, minlength=R.shape[0]).astype(float); prof = np.convolve(prof, np.ones(5) / 5, 'same'); pk = int(prof.argmax())
    y10 = pk; y90 = pk                                  # contiguous band around the densest row where the profile stays >= 25% of the peak
    while y10 > 0 and prof[y10 - 1] >= 0.25 * prof[pk]: y10 -= 1
    while y90 < len(prof) - 1 and prof[y90 + 1] >= 0.25 * prof[pk]: y90 += 1
    band = (ys >= y10) & (ys <= y90); x05, x95 = np.percentile(xs[band], [1, 99])
    src = cv2.imread(str(A/b['src'])); SH, SW = src.shape[:2]
    if s['processed']:
        bx0, by0, bx1, by1 = b['box']; M = cv2.getRotationMatrix2D(((bx0+bx1)/2, (by0+by1)/2), s['angle_deg'], 1.0)
        corners = np.array([[0,0,1],[SW,0,1],[SW,SH,1],[0,SH,1]], float) @ M.T; M[:, 2] -= corners.min(0)
        ox, oy = s['R_box_in_rotated_src'][:2]; Minv = cv2.invertAffineTransform(M)
    else:
        ox, oy = b['box'][:2]; Minv = np.float32([[1, 0, 0], [0, 1, 0]])
    q = np.float32([[[x05+ox, y10+oy], [x95+ox, y10+oy], [x95+ox, y90+oy], [x05+ox, y90+oy]]])
    q_src = cv2.transform(q, Minv)[0]
    p = P[n]; w, h = p['crop_wh']; Mpl, _ = cv2.estimateAffinePartial2D(np.float32([[0,0],[w,0],[w,h],[0,h]]), np.float32(p['quad']))
    rec['core_quad'] = cv2.transform(q_src[None], Mpl)[0].round(1).tolist()
json.dump(out, open(O/'dataset_line_quads.json', 'w'), indent=1)
v = [inter(a['core_quad'], b['core_quad']) for i, a in enumerate(out) for b in out[i+1:] if a['photo'] == b['photo']]
print('core_quad pairs with intersection/min-area >= 0.2:', sum(x >= .2 for x in v), '| max', round(max(v), 3))
