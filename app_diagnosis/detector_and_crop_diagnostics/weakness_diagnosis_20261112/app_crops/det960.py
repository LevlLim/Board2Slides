import sys, os, json
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, '/home/claude/fresh/pkg/b2n_ocr'); os.environ['B2N_PACKAGE_ROOT'] = '/home/claude/fresh/pkg'
import ocr_backend as ob
from preprocessing import text_lines as tl
PH = Path("/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment/1111_unfamiliar")
OUT = Path('crops960'); OUT.mkdir(exist_ok=True)
det = tl.Detector(ob.DETECTOR_NAME, det_args={'thresh': .20, 'box_thresh': .40, 'unclip_ratio': 1.2}, model_dir=ob.DETECTOR_DIR, runtime=ob.DETECTOR_RUNTIME)
res = {}
for f in sorted(PH.glob('*.HEIC')):
    img = ob.load_image(f)
    b = det.boxes(img, limit=960, unclip=1.2)
    b, _ = tl.group_into_lines(b)
    b = [b[i] for i in ob.spatial_order(b)]
    res[f.stem] = [np.asarray(p).astype(float).tolist() for p in b]
    for k, p in enumerate(b): cv2.imwrite(str(OUT / f'{f.stem}_{k:02d}.png'), tl.quad_crop(img, p))
    print(f.stem, len(b), flush=True)
json.dump(res, open('app_boxes960.json', 'w'))
