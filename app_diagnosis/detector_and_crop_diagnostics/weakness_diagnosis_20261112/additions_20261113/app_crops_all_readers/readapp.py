import sys, os, json, csv, math
from pathlib import Path
import cv2
sys.path.insert(0, '/home/claude/fresh/pkg/b2n_ocr'); os.environ['B2N_PACKAGE_ROOT'] = '/home/claude/fresh/pkg'
import ocr_backend as ob
U = Path("/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment")
MODELS = {'S0': U/'results_strategies_s0s3/results/models/PP-OCRv5_server_rec_board2notes_S0', 'S0s2': U/'results_strategies_s0s3/results/models/PP-OCRv5_server_rec_board2notes_S0s2',
          'G0': U/'results_guide_g0/results/models/PP-OCRv5_server_rec_board2notes_G0', 'G0s2': U/'results_guide_g0/results/models/PP-OCRv5_server_rec_board2notes_G0s2',
          'S1his': U/'results_server_his_20261110/results/models/PP-OCRv5_server_rec'}
ob.MODELS.update(MODELS)
match = json.load(open('match.json'))
files = sorted({f"{i.rsplit('_',2)[0]}" for i in []})
need = sorted({(k.split('_')[1], b) for k, v in match.items() if v['included'] == 'yes' for b in v['boxes']})
paths = [f"crops/IMG_{p}_{b:02d}.png" for p, b in need]
imgs = [cv2.imread(p) for p in paths]
out = {}
for m in MODELS:
    rec = ob.RecognitionModel(m, threads=8)
    out[m] = {p: x['text'] for p, x in zip(paths, rec.read(imgs))}
    print(m, 'done', flush=True)
json.dump(out, open('app_reads.json', 'w'), ensure_ascii=False)
