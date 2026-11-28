import sys, os, json
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, '/home/claude/fresh/pkg/b2n_ocr'); os.environ['B2N_PACKAGE_ROOT'] = '/home/claude/fresh/pkg'
import ocr_backend as ob
from preprocessing import text_lines as tl
SIZES = [640, 960, 1280, 1600, 2400, 4000]
ZOOM = {'fresh': [3], 'jh': [4], 'jy': [2]}
FRESH_ZOOM_BOARDS = 'DE'   # zoom out only the sentence boards (small writing); big words zoomed out = medium, already covered
_cache = {}
def load(path):
    if path not in _cache:
        _cache.clear(); _cache[path] = ob.load_image(Path(path))
    return _cache[path]
def variant(img, k):
    if k == 1: return img, 1.0, (0.0, 0.0)
    h, w = img.shape[:2]
    small = cv2.resize(img, (round(w / k), round(h / k)), interpolation=cv2.INTER_AREA)
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    can = np.empty_like(img); can[:] = np.median(border, 0).astype(np.uint8)
    oy, ox = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    can[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
    return can, 1.0 / k, (ox, oy)
def line_h(q):
    q = np.asarray(q, float); p = tl.order_tl_tr_br_bl(q)
    return float((np.linalg.norm(p[0] - p[3]) + np.linalg.norm(p[1] - p[2])) / 2)
