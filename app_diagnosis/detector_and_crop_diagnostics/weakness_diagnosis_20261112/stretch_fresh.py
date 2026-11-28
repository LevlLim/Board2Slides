import csv, math, os, sys, collections, json
from pathlib import Path
import cv2
sys.path.insert(0, '/home/claude/fresh/pkg/b2n_ocr'); os.environ['B2N_PACKAGE_ROOT'] = '/home/claude/fresh/pkg'
import ocr_backend as ob
U = Path("/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment")
MODELS = {'S0': U/'results_strategies_s0s3/results/models/PP-OCRv5_server_rec_board2notes_S0', 'S0s2': U/'results_strategies_s0s3/results/models/PP-OCRv5_server_rec_board2notes_S0s2',
          'G0': U/'results_guide_g0/results/models/PP-OCRv5_server_rec_board2notes_G0', 'G0s2': U/'results_guide_g0/results/models/PP-OCRv5_server_rec_board2notes_G0s2',
          'S1his': U/'results_server_his_20261110/results/models/PP-OCRv5_server_rec'}
ob.MODELS.update(MODELS)
D = Path('/tmp/claude-0/-home-claude/02f3df2f-1dbb-533d-b187-71169cf31ef0/scratchpad/unf/board2notes_freshwriter_test_20261111')
man = [r for r in csv.DictReader(open(D/'manifest.csv', encoding='utf-8')) if r['included'] == 'yes']
imgs = [cv2.imread(str(D/r['file'])) for r in man]
norm = lambda s: ' '.join(s.lower().split())
def lev(a, b):
    p = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        c = [i]
        for j, cb in enumerate(b, 1): c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (ca != cb)))
        p = c
    return p[-1]
need = lambda t: len(' '.join(t.split())) + sum(1 for a, b in zip(' '.join(t.split()), ' '.join(t.split())[1:]) if a == b)
rows = []
for m in MODELS:
    rec = ob.RecognitionModel(m, threads=8)
    for s in (1.0, 1.5, 2.0):
        ims = [cv2.resize(im, (int(round(im.shape[1] * s)), im.shape[0]), interpolation=cv2.INTER_CUBIC) if s != 1 else im for im in imgs]
        for r, x, im in zip(man, rec.read(ims), imgs):
            a = norm(r['label']); fit = math.ceil(math.ceil(64 * im.shape[1] / im.shape[0]) / 8) / need(r['label'])
            rows.append(dict(model=m, s=s, id=r['id'], board=r['board'], writing=r['writing'], fit=fit, cer=lev(a, norm(x['text'])) / len(a)))
    print(m, 'done', flush=True)
json.dump(rows, open('stretch_fresh_rows.json', 'w'))
