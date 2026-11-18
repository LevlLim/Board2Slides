import csv, json, os, sys, time
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path('pkg/b2n_ocr').resolve())); os.environ['B2N_PACKAGE_ROOT'] = str(Path('pkg').resolve())
import ocr_backend as ob
U = Path("/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment")
MODELS = {'S0': U/'results_strategies_s0s3/results/models/PP-OCRv5_server_rec_board2notes_S0',
          'S0s2': U/'results_strategies_s0s3/results/models/PP-OCRv5_server_rec_board2notes_S0s2',
          'S1': U/'results_strategies_s0s3/results/models/PP-OCRv5_server_rec_board2notes_S1',
          'S1s2': U/'results_strategies_s0s3/results/models/PP-OCRv5_server_rec_board2notes_S1s2',
          'G0': U/'results_guide_g0/results/models/PP-OCRv5_server_rec_board2notes_G0',
          'G0s2': U/'results_guide_g0/results/models/PP-OCRv5_server_rec_board2notes_G0s2',
          'V2': U/'results_versions_2x2/results/models/PP-OCRv5_server_rec_board2notes_V2',
          'S1his': U/'results_server_his_20261110/results/models/PP-OCRv5_server_rec'}
ob.MODELS.update(MODELS)
D = Path('/tmp/claude-0/-home-claude/02f3df2f-1dbb-533d-b187-71169cf31ef0/scratchpad/unf/board2notes_freshwriter_test_20261111')
R = [r for r in csv.DictReader(open(D/'manifest.csv', encoding='utf-8')) if r['included'] == 'yes']
crops = [cv2.imread(str(D/r['file'])) for r in R]
norm = lambda s: ' '.join(str(s).lower().split())
def lev(a, b):
    p = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        c = [i]
        for j, cb in enumerate(b, 1): c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (ca != cb)))
        p = c
    return p[-1]
out = []
for m in MODELS:
    t = time.time(); rec = ob.RecognitionModel(m, threads=8); ident = rec.identity()
    reads = rec.read(crops)
    for r, x in zip(R, reads):
        e = lev(norm(r['label']), norm(x['text'])); n = len(norm(r['label']))
        out.append(dict(model=m, id=r['id'], board=r['board'], photo=r['photo'], writing=r['writing'], label=r['label'], pred=x['text'], err=e, chars=n, cer=e / max(1, n)))
    print(m, f"{np.mean([o['cer'] for o in out if o['model'] == m]):.4f}", f"{time.time()-t:.0f}s", ident['params_sha256'][:12], flush=True)
with open(D/'scores_server_models.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
