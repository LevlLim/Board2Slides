"""Builds notebooks/16_jh_full_photo_app_check.ipynb (reads saved outputs only)."""
import nbformat as nbf, sys
from pathlib import Path
A = Path(sys.argv[1]); out = A / 'whiteboard-ocr/notebooks/16_jh_full_photo_app_check.ipynb'
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 16 · JH full photos through the unchanged app pipeline

**Question.** Notebooks 14 and 15 found that S1 fails on JH's sentence crops mainly because the crops are upright cuts of sloped lines: once resized to 64 px high, the text covers too few of S1's 128 output positions. Does the **app** have this problem when it starts from the full photos?

**Pipeline (unchanged app code):**
1. EXIF load.
2. `PP-OCRv5_mobile_det` with the app's settings (thresh 0.20, box_thresh 0.40, unclip 1.2, 1600-px quick pass with small-text retry).
3. `group_into_lines`, then `spatial_order`.
4. `quad_crop`, which cuts along the rotated box.
5. The S1 reader (`RecognitionModel('S1')`).

This was run on the Mac in the app environment (`whiteboard-ocr/scripts/run_jh_full_photos_app.py`). It reuses notebook 15's runner code and stops if the app code has changed. No labels were used in the run.

**Photos.** `JH_datasetNov5Sentence/Photo1–7.jpg`, which hold the **59 JH sentence lines** of the new-writer set.

**Correspondence**, from images only and frozen in `claude_jh_fullphoto_20261109/PROTOCOL.json` before any app output was opened:
- Each dataset crop was located in its photo by SIFT + RANSAC. All 59 placed (160–460 inliers, scale 1.59, rotation 0°), which confirms the dataset crops are upright cuts.
- Each line is represented by its **core band**: the dense row band of the straightened crop, mapped back into the photo. The bands were checked visually, and no two of them overlap.
- **Link rule:** an app region and a line are linked if the region covers ≥ 20 % of the line's core, or ≥ 50 % of the region lies on it. Text is never used.
- Each line then falls into one of: **one-to-one**, **merge** (one region over several lines), **split** (one line over several regions), **missed**, or **extra region** (no dataset line; it may still be real writing that was never cropped).
- This is done for raw detector boxes and for grouped boxes separately, so grouping effects are not blamed on the detector.

**Kept separate.**
- *Recognition* is compared **only on one-to-one lines.* Each line is read by the same S1 object from four crops: the dataset original, tightened and straightened crops, and the app's own crop.
- *Missed, merged and split lines* are detection/grouping outcomes and are reported on their own.

Labels are used **only for scoring**, after correspondence. No training; app unchanged.
""")

code(r"""
from pathlib import Path
import json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image as IPImage
pd.set_option('display.max_colwidth', 70); pd.set_option('display.width', 220)
ASSIGNMENT_OVERRIDE = None
cands = [Path(ASSIGNMENT_OVERRIDE)] if ASSIGNMENT_OVERRIDE else []
for base in [Path.cwd(), *Path.cwd().parents]:
    cands += [base, base / 'Assignment']
A = next(p.resolve() for p in cands if (p / 'claude_jh_fullphoto_20261109/RESULTS.json').is_file())
O = A / 'claude_jh_fullphoto_20261109'; RUN = A / 'whiteboard-ocr/outputs/jh_full_photos_app'; V = O / 'visuals'
P = json.loads((O / 'PROTOCOL.json').read_text()); R = json.loads((O / 'RESULTS.json').read_text()); run = json.loads((RUN / 'run.json').read_text())
print('Protocol:', P['status'])
ck = run['cache_key']
print('Run:', run['created'], '| detector', run['detector'], ck['det_args'], '| reader', run['reader_identity']['model_id'],
      run['reader_identity']['params_sha256'][:12], '| paddle', ck['env']['paddlepaddle'], ck['env']['platform'])
print('Labels used in the run:', run['labels_used'], '| training:', not run['no_training'], '| app modified:', run['app_modified'])
print('Timings:', {k: round(v, 2) for k, v in run['timings'].items()}, '| detection per photo (median of 3):',
      [round(p['detect_seconds'], 2) for p in run['photos']])
""")

md(r"""
## 1 · Detection and grouping outcomes (no recognition involved)
""")

code(r"""
rows = []
for pid, v in sorted(R['per_photo'].items(), key=lambda x: int(x[0])):
    rows.append({'photo': int(pid), 'dataset lines': v['lines'], 'raw boxes': v['raw_boxes'], 'grouped regions': v['regions'],
                 'raw: one-to-one': v['raw'].get('one_to_one', 0), 'raw: split': v['raw'].get('split', 0), 'raw: merge': v['raw'].get('merge', 0),
                 'raw: missed': v['raw'].get('missed_line', 0), 'grouped: one-to-one': v['grouped'].get('one_to_one', 0),
                 'grouped: split': v['grouped'].get('split', 0), 'grouped: merge': v['grouped'].get('merge', 0),
                 'grouped: missed': v['grouped'].get('missed_line', 0), 'extra regions': v['grouped'].get('extra_region', 0), 'detector path': v['detect_path']})
DT = pd.DataFrame(rows).set_index('photo'); display(DT)
print('Totals - raw boxes:', R['totals']['raw'], '| after grouping:', R['totals']['grouped'])
print('Extra regions (no dataset line):', [(e['photo'], e['region'], e['text']) for e in R['extra']])
print('Non-one-to-one components after grouping:', R['other'] if R['other'] else 'none')
""")

md(r"""
**Overlays.** Left: raw detector boxes. Right: the grouped regions S1 reads. Each box carries its reading-order number.
- Green = one-to-one with a dataset line, blue = split, purple = merge, red = extra region.
- Thin grey = the dataset line core bands. A thick red band = a missed line.
""")

code(r"""
for f in R['visuals']['overlays']:
    display(IPImage(filename=str(V / f), width=1000))
""")

md(r"""
## 2 · Recognition on one-to-one lines: the same line, four crops, one reader

The four crops are the dataset **original**, **tightened** and **straightened** crops, and the **app** crop (`quad_crop` along the detector's rotated box). The app's own S1 object reads all four on the Mac.

Each line is scored against its dataset label (CER, lower-case, collapsed spaces). The intervals resample whole photos; with only 7 photos they are indicative.
""")

code(r"""
s = R['summary']
print(f"One-to-one lines: {s['n_one_to_one']} of {s['lines']}")
tab = pd.DataFrame([{'crop': k, 'mean line CER %': 100 * s[f'cer_{k}'], 'deletions': s[f'D_{k}'], 'lines with CER >= 15%': s[f'persist15_{k}'],
                     'median width at 64 px': s['median_width64'][k]} for k in ('original', 'tight', 'straightened', 'app')]).set_index('crop')
display(tab.round(2))
for k in ('original', 'tight', 'straightened'):
    x = s[f'app_minus_{k}']
    print(f"app - {k:12s}: {100*x['diff']:+.2f} points (photo bootstrap {100*x['ci_photo_bootstrap'][0]:+.1f} .. {100*x['ci_photo_bootstrap'][1]:+.1f}); app better on {x['app_better']}, worse on {x['app_worse']}")
print('Median |residual slope| of the writing inside the app crops:', s['app_residual_slope_abs_median'], 'deg')
""")

code(r"""
OO = pd.DataFrame([{'line': r['line'], 'photo': r['photo'], 'label': r['label'], 'app reading': r['app_text'], 'CER app': r['app']['cer'],
                    'CER original': r['original']['cer'], 'CER tight': r['tight']['cer'], 'CER straightened': r['straightened']['cer'],
                    'width64 app': r['app_width64'], 'width64 tight': r['tight']['width64'], 'width64 straightened': r['straightened']['width64'],
                    'app slope': r['app_residual_slope'], 'core covered': r['core_cover']} for r in R['one_to_one']])
display(Markdown('**Lines where the app reading is not perfect** (all one-to-one; character errors only):'))
display(OO[OO['CER app'] > 0].sort_values('CER app', ascending=False).round(3))
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
for k, c in (('tight', '#94a3b8'), ('straightened', '#16a34a'), ('app', '#2563eb')):
    ax[0].scatter(OO[f'width64 {k}'] if k != 'app' else OO['width64 app'], OO[f'CER {k}'] if k != 'app' else OO['CER app'], s=16, c=c, label=k)
ax[0].set_xlabel("width at 64 px (more = more of S1's 128 positions covered)"); ax[0].set_ylabel('line CER'); ax[0].legend(); ax[0].set_title('Wider input, fewer errors')
ax[1].bar(['original', 'tight', 'straightened', 'app'], [100 * s[f'cer_{k}'] for k in ('original', 'tight', 'straightened', 'app')], color=['#cbd5e1', '#94a3b8', '#16a34a', '#2563eb'])
ax[1].set_ylabel('mean line CER (%)'); ax[1].set_title('Same 59 lines, same S1 object')
plt.tight_layout(); plt.show()
""")

md(r"""
**Side by side.** Each row shows one line as four crops, in this order: original, tightened, straightened, app. Under each crop are its CER, missing characters (D) and width at 64 px, then S1's reading. Rows are ordered by the app's gain over the tightened crop, largest first.
""")

code(r"""
for f in R['visuals']['compare']:
    display(IPImage(filename=str(V / f), width=1100))
""")

md(r"""
## 3 · Exact reader inputs for every app region

For each photo, every region is shown in reading order:
- the crop as `quad_crop` cuts it (left);
- the exact S1 input (right): 64 px high, padded to 1024 px, with mid-grey padding;
- S1's reading and the dataset label.
""")

code(r"""
for f in R['visuals']['strips']:
    display(IPImage(filename=str(V / f), width=1000))
""")

md(r"""
## 4 · What this shows

**Detection and grouping.**
- After grouping, all 59 dataset lines are one-to-one: none are missed, merged or split.
- In the raw boxes, Photo 2 had one line split into two boxes, and `group_into_lines` joined it correctly.
- Two extra regions have no dataset line: a blank reading in Photo 2 and the Windows taskbar "Search" in Photo 5. That one is screen content, not board writing.

**Recognition.** The app's crops read at **0.9 % mean line CER**, against:
- 21.1 % for the original dataset crops;
- 16.4 % for the tightened crops;
- 5.4 % for the straightened crops.

The comparison is by the same S1 object on the same 59 lines, with these specifics:
- Missing characters: 5, against 232, 187 and 68.
- No app line reaches 15 % CER; the original crops had 29 such lines.
- The 13 imperfect app lines are one- to three-character substitutions such as u→v, a→o, U→V and l→T.

**Why.**
- `quad_crop` cuts along the detector's rotated, tight box, so lines come out level (median residual slope 0.4°).
- The crops are also short in height. At 64 px, the app crops are a median of **814 px wide**, against 386 px tightened and 500 px straightened, so the text covers far more of S1's 128 output positions.
- That is the same mechanism notebook 14 identified.
- Even the four lines that straightening could not fix (JH1.4, JH5.3, JH6.4, JH7.3, which have neighbouring-line ink in the dataset crop) read correctly or nearly so in the app.

**Implications.**
- The poor JH sentence scores in the new-writer benchmark are mostly an artifact of how the benchmark crops were cut. They do not describe what the app would do on these photos.
- Training conclusions drawn from that benchmark (for example S1 vs G0 on new writers) should be re-checked on corrected or app-generated crops.

**Caveats.**
- 7 photos from one writer and session.
- JH data has been used in development.
- Labels are unchanged and assumed correct.
- Extra regions are not counted as errors here. On the real page they would add text, which is a page-CER concern.
- Reading order was not evaluated (it is deferred).
""")

nb = nbf.v4.new_notebook(); nb.cells = C
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}
nbf.write(nb, out); print('wrote', out)
