import sys, os, json, csv
from pathlib import Path
import cv2
sys.path.insert(0, '/home/claude/fresh/pkg/b2n_ocr'); os.environ['B2N_PACKAGE_ROOT'] = '/home/claude/fresh/pkg'
import ocr_backend as ob
U = Path("/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment")
R = U / 'results_strategies_s0s3/results/models'
MODELS = {'pretrained': Path('/root/.paddlex/official_models/PP-OCRv5_server_rec'),
          'S0': R/'PP-OCRv5_server_rec_board2notes_S0', 'S0s2': R/'PP-OCRv5_server_rec_board2notes_S0s2',
          'S1': R/'PP-OCRv5_server_rec_board2notes_S1', 'S1s2': R/'PP-OCRv5_server_rec_board2notes_S1s2',
          'G0': U/'results_guide_g0/results/models/PP-OCRv5_server_rec_board2notes_G0', 'G0s2': U/'results_guide_g0/results/models/PP-OCRv5_server_rec_board2notes_G0s2',
          'V2': U/'results_versions_2x2/results/models/PP-OCRv5_server_rec_board2notes_V2', 'S1his': U/'results_server_his_20261110/results/models/PP-OCRv5_server_rec'}
ob.MODELS.update(MODELS)
FR = Path('/tmp/claude-0/-home-claude/02f3df2f-1dbb-533d-b187-71169cf31ef0/scratchpad/unf/board2notes_freshwriter_test_20261111')
man = [r for r in csv.DictReader(open(FR/'manifest.csv', encoding='utf-8')) if r['included'] == 'yes']
def app_paths(mfile, cdir):
    m = json.load(open(mfile))
    return sorted({f"{cdir}/{byid['photo']}_{b:02d}.png" for k, v in m.items() if v['included'] == 'yes' for byid in [next(r for r in man if r['id']==k)] for b in v['boxes']})
jobs = {'hand': [str(FR/r['file']) for r in man], 'app': app_paths('match.json','crops'), 'app960': app_paths('match960.json','crops960')}
have = {'app': json.load(open('app_reads.json')), 'app960': json.load(open('app_reads960.json'))}
out = json.load(open('all_reads.json')) if os.path.exists('all_reads.json') else {}
for m in MODELS:
    rec = None
    for src, paths in jobs.items():
        key = f'{m}|{src}'
        if key in out: continue
        if src != 'hand' and m in have[src]:
            out[key] = {p: have[src][m][p] for p in paths}; continue
        if src == 'hand' and m != 'pretrained': continue          # saved in scores_server_models.csv
        rec = rec or ob.RecognitionModel(m, threads=8)
        imgs = [cv2.imread(p) for p in paths]
        out[key] = {p: x['text'] for p, x in zip(paths, rec.read(imgs))}
        json.dump(out, open('all_reads.json', 'w'), ensure_ascii=False)
        print(key, len(paths), flush=True)
json.dump(out, open('all_reads.json', 'w'), ensure_ascii=False)
print('ALL DONE', flush=True)
