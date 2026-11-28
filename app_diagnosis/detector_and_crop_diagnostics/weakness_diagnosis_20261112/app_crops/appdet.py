import sys, os, json, csv
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, '/home/claude/fresh/pkg/b2n_ocr'); os.environ['B2N_PACKAGE_ROOT'] = '/home/claude/fresh/pkg'
import ocr_backend as ob
from preprocessing import text_lines as tl
PH = Path("/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment/1111_unfamiliar")
OUT = Path('/home/claude/appcrop/crops'); OUT.mkdir(exist_ok=True)
det = tl.Detector(ob.DETECTOR_NAME, det_args={'thresh': .20, 'box_thresh': .40, 'unclip_ratio': 1.2}, model_dir=ob.DETECTOR_DIR, runtime=ob.DETECTOR_RUNTIME)
res = {}
for f in sorted(PH.glob('*.HEIC')):
    img = ob.load_image(f)
    boxes, info = det.detect(img, unclip=1.2)
    boxes, _ = tl.group_into_lines(boxes)
    boxes = [boxes[i] for i in ob.spatial_order(boxes)]
    res[f.stem] = [np.asarray(p).astype(float).tolist() for p in boxes]
    for k, p in enumerate(boxes):
        cv2.imwrite(str(OUT / f'{f.stem}_{k:02d}.png'), tl.quad_crop(img, p))
    print(f.stem, len(boxes), img.shape, flush=True)
json.dump(res, open('app_boxes.json', 'w'))
