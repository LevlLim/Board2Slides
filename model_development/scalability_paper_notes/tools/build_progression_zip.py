#!/usr/bin/env python3
"""Build board2notes_reader_progression_20261118.zip: every exported reader of the model-development story,
renamed so the folder says what changed. Weights are copied unchanged (stored, not compressed: they do not
compress). Scores are recomputed from each experiment's own predictions.csv (mean line CER, %).

Resumable: adds the folders that are not in the zip yet, then stops before --max-seconds.
Usage: python3 build_progression_zip.py [--max-seconds 150]
"""
import argparse
import collections
import csv
import hashlib
import json
import time
import zipfile
from pathlib import Path

A = Path(__file__).resolve().parents[2]                      # Assignment/
ZIP = A / 'board2notes_reader_progression_20261118.zip'
R = 'results_{}/results/models/PP-OCRv5_server_rec{}'
MODELS = [  # folder name, source folder, (experiment, model name in predictions.csv), what changed
    ('00_downloaded_PP-OCRv5_server_rec', A / 'models/PP-OCRv5_server_rec_official_hf', ('versions_2x2', 'pretrained'),
     'PaddleOCR\'s reader as downloaded (Hugging Face PaddlePaddle/PP-OCRv5_server_rec @ b26c3587, Apache-2.0). No fine-tuning.'),
    ('01_finetuned_on_our_whiteboard_lines', A / R.format('server_c0_20261111', '_board2notes_C0'), ('server_c0_20261111', 'C0'),
     'Fine-tuned on our 1,113 whiteboard line crops with PaddleOCR\'s original setup (48x320 input, 40 CTC columns).'),
    ('02_plus_augmented_copies_NOT_KEPT', A / R.format('versions_2x2', '_board2notes_C1'), ('versions_2x2', 'C1'),
     '01 plus 3,156 pre-made augmented copies. No clear gain, so copies were dropped later.'),
    ('03_wide_input_still_40_columns_BROKEN', A / R.format('versions_2x2', '_board2notes_P40'), ('versions_2x2', 'P40'),
     'Wider input (64x1024) but still only 40 CTC columns: the reader breaks. Pooling-pair control for 04.'),
    ('04_width_pooling_one_column_per_8px', A / R.format('versions_2x2', '_board2notes_P128'), ('versions_2x2', 'P128'),
     'Same lines and input as 03; only the pooling differs: one CTC column per 8 px of width (128 at 1024 px).'),
    ('05_plus_long_lines', A / R.format('versions_2x2', '_board2notes_V3'), ('versions_2x2', 'V3'),
     '04 plus the long lines that need more than 40 columns (possible only after 04).'),
    ('06_baseline_for_training_strategy_tests', A / R.format('strategies_s0s3', '_board2notes_S0'), ('strategies_s0s3', 'S0'),
     'Clean baseline for cycles 5-6: our lines incl. long lines, no copies, minus 64 uncertain-writer crops. Run 1 of 2.'),
    ('07_plus_stagewise_learning_rate', A / R.format('strategies_s0s3', '_board2notes_S1'), ('strategies_s0s3', 'S1'),
     '06 plus stage-wise learning rates [0.1, 0.1, 0.25, 0.5, 1.0] (early stages learn 10x slower). Run 1 of 2. Kept.'),
    ('08_final_stagewise_lr_on_teammate_dataset', A / R.format('server_his_20261110', ''), ('server_his_20261110', 'S1his'),
     'Final model: 07\'s settings trained on the teammate\'s larger, leak-checked 80:10:10 dataset.'),
    ('extra_paper_notes_reader_IAM', A / R.format('versions_2x2', '_board2notes_V1b'), ('versions_2x2', 'V1b'),
     'Scalability: same training framework as the whiteboard reader below, trained on IAM paper handwriting only.'),
    ('extra_whiteboard_reader_same_settings_as_paper', A / R.format('versions_2x2', '_board2notes_V2'), ('versions_2x2', 'V2'),
     'Scalability control: our whiteboard lines incl. long lines, no copies; same settings as the paper reader.'),
]
SETS = [('whiteboard val', 'Whiteboard validation'), ('CNN test', 'CNN board'), ('long sentences (held out)', 'Long sentences'),
        ('IAM val', 'Paper handwriting (IAM)'), ('new writers', 'New writers'), ('his test', 'Teammate test set')]
FILES = ('inference.json', 'inference.pdiparams', 'inference.yml', 'verified.json', 'README.md', 'config.json')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def scores():
    acc = collections.defaultdict(list)
    for exp in sorted({m[2][0] for m in MODELS} | {'strategies_s0s3'}):
        for r in csv.DictReader(open(A / f'results_{exp}' / 'results' / 'predictions.csv', encoding='utf-8')):
            acc[(exp, r['model'], r['set'])].append(float(r['cer']))
    out = {}
    for name, _, (exp, model), _ in MODELS:
        row = {}
        for s, _ in SETS:
            v = acc.get((exp, model, s))
            if not v and model == 'pretrained':           # the downloaded reader was also scored in other runs
                v = acc.get(('strategies_s0s3', 'pretrained', s)) or acc.get(('server_his_20261110', 'pretrained', s))
            row[s] = round(100 * sum(v) / len(v), 1) if v else None
        out[name] = row
    return out


def readme(sc):
    head = '| folder | what changed | ' + ' | '.join(l for _, l in SETS) + ' |'
    L = ['# Board2Notes reader progression (18 Nov 2026)', '',
         'Every exported PP-OCRv5 server reader from our model development, named by what changed. '
         'Mean line CER (%), lower is better, case-insensitive, whitespace collapsed; recomputed from each run\'s '
         '`predictions.csv`. One training run per model (the slides\' two-run averages for 06/07 are in the deck). '
         '"–" = not scored on that set.', '',
         head, '|' + '---|' * (len(SETS) + 2)]
    for name, _, _, what in MODELS:
        L.append(f'| `{name}` | {what} | ' + ' | '.join('–' if sc[name][s] is None else f'{sc[name][s]:.1f}'
                                                      for s, _ in SETS) + ' |')
    L += ['', '## Notes', '',
          '- **Reading these models.** Each folder is a PaddleOCR inference export (`inference.json`, `.pdiparams`, '
          '`.yml`). 00-02 take 48 px tall input; 03-08 and the extras take 64 px tall input padded to at least 1024 px '
          '(as in their `inference.yml`). Our evaluation read 00 at 48 px padded to 1024 px; PaddleOCR\'s default '
          'resize (minimum width 320) reads long lines noticeably worse. The 9 Nov OCR package '
          '(`board2notes_ocr_package`) and `scalability_paper_notes_20261118/tools/run_paper_notes_eval.py` read '
          'them the verified way.',
          '- **Dictionary.** All folders keep PaddleOCR\'s 18,383-character dictionary (about 15,700 Chinese '
          'characters). Fine-tuning used English lines only.',
          '- **Teammate test set** for 00 and 08 is the teammate\'s held-out 169-line test split; the other sets are ours. '
          '08 is scored only there: the teammate\'s dataset contains most of our validation and CNN-board crops, so '
          '08\'s scores on our sets would not be held out.',
          '- 00 has no `verified.json`: it is PaddleOCR\'s own export. Reading it here reproduced our recorded '
          'predictions for it on 40 of 40 crops.',
          '- `verified.json` is each model\'s export check (exported model reads 30 fixed crops identically to its '
          'training checkpoint). `SOURCE.txt` gives the original folder and the weight file\'s SHA-256.',
          '- Not included: the gradient-routing runs (never exported), the CTC-everywhere runs, second runs of 06/07, '
          'training checkpoints and datasets. They stay in `Assignment/` at their original paths.']
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-seconds', type=float, default=150)
    a = ap.parse_args()
    t0 = time.time()
    for name, src, _, _ in MODELS:
        if not (src / 'inference.pdiparams').is_file():
            raise SystemExit(f'missing weights: {src}')
    have = set()
    if ZIP.is_file():
        with zipfile.ZipFile(ZIP) as z:
            have = {n.split('/')[1] for n in z.namelist() if n.count('/') >= 2 and n.endswith('SOURCE.txt')}
    for name, src, (exp, model), what in MODELS:
        if name in have:
            continue
        if time.time() - t0 > a.max_seconds:
            print('PARTIAL: run again to continue')
            return
        with zipfile.ZipFile(ZIP, 'a', compression=zipfile.ZIP_STORED) as z:
            for f in FILES:
                if (src / f).is_file():
                    z.write(src / f, f'board2notes_reader_progression/{name}/{f}')
            z.writestr(f'board2notes_reader_progression/{name}/SOURCE.txt',
                       f'{what}\nsource folder: Assignment/{src.relative_to(A)}\nexperiment: results_{exp}, model {model}\n'
                       f'inference.pdiparams sha256: {sha256(src / "inference.pdiparams")}\n')
        print('added', name, round(time.time() - t0, 1), 's')
    sc = scores()
    text = readme(sc)
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
    if 'board2notes_reader_progression/README.md' not in names:
        with zipfile.ZipFile(ZIP, 'a', compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('board2notes_reader_progression/README.md', text)
    (A / 'board2notes_reader_progression_20261118_README.md').write_text(text, encoding='utf-8')
    with zipfile.ZipFile(ZIP) as z:
        bad = z.testzip()
        n = len(z.namelist())
    print(f'DONE: {ZIP.name}, {n} files, {ZIP.stat().st_size / 1e6:.0f} MB, CRC check: {"OK" if bad is None else bad}')


if __name__ == '__main__':
    main()
