import sys, json, os
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path('pkg/b2n_ocr').resolve()))
os.environ['B2N_PACKAGE_ROOT'] = str(Path('pkg').resolve())
import ocr_backend as ob
from preprocessing import text_lines as tl
PH = Path("/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment/1111_unfamiliar")
OUT = Path('/tmp/claude-0/-home-claude/02f3df2f-1dbb-533d-b187-71169cf31ef0/scratchpad/unf/det2'); OUT.mkdir(parents=True, exist_ok=True)
det = tl.Detector(ob.DETECTOR_NAME, det_args={'thresh': .20, 'box_thresh': .40, 'unclip_ratio': 2.0}, model_dir=ob.DETECTOR_DIR, runtime=ob.DETECTOR_RUNTIME)
res = {}
for f in sorted(PH.glob('*.HEIC')):
    img = ob.load_image(f)
    out = []
    for lim in (960, 1600, 2400):
        b = det.boxes(img, limit=lim, unclip=2.0)
        b, _ = tl.group_into_lines(b, gap_factor=2.5)
        out.append([np.asarray(p).astype(float).tolist() for p in b])
    res[f.stem] = out
    vis = img.copy()
    for k, col in zip(range(3), [(0,0,255),(0,160,0),(255,0,0)]):
        for p in out[k]:
            cv2.polylines(vis, [np.asarray(p).astype(np.int32)], True, col, 4 + 4 * (2 - k))
    s = 900 / max(vis.shape[:2]); cv2.imwrite(str(OUT / f'{f.stem}.jpg'), cv2.resize(vis, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 70])
    print(f.stem, [len(o) for o in out], flush=True)
json.dump(res, open(OUT / 'boxes.json', 'w'))
