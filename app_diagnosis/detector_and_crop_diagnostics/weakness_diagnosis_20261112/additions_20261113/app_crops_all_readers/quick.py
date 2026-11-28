import sys, os
from pathlib import Path
sys.path.insert(0, '/home/claude/fresh/pkg/b2n_ocr'); os.environ['B2N_PACKAGE_ROOT'] = '/home/claude/fresh/pkg'
import ocr_backend as ob
from preprocessing import text_lines as tl
PH = Path("/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment/1111_unfamiliar")
det = tl.Detector(ob.DETECTOR_NAME, det_args={'thresh': .20, 'box_thresh': .40, 'unclip_ratio': 1.2}, model_dir=ob.DETECTOR_DIR, runtime=ob.DETECTOR_RUNTIME)
for ph in ('IMG_5095','IMG_5109','IMG_5113','IMG_5122'):
    img = ob.load_image(PH/f'{ph}.HEIC')
    b, info = det.detect(img, unclip=1.2)
    out=[ph, info['path'], len(tl.group_into_lines(b)[0])]
    for lim, un, gf in ((960,1.2,1.2),(960,2.0,1.2),(960,2.0,2.5),(2400,2.0,2.5)):
        bb = det.boxes(img, limit=lim, unclip=un); out.append((lim,un,gf,len(tl.group_into_lines(bb, gap_factor=gf)[0])))
    print(*out, flush=True)
