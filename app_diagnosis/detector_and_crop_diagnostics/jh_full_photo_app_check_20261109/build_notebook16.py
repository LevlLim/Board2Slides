"""Verify saved JH full-photo results, write a report/gallery, and execute notebook 16.

No OCR inference or training. Run with Assignment/.venv-paddle/bin/python.
"""
from collections import defaultdict
from pathlib import Path
import base64
import csv
import hashlib
import html
import json
import random
import sys

import cv2
import numpy as np
import nbformat as nbf
from nbclient import NotebookClient
from rapidfuzz.distance import Levenshtein

HERE = Path(__file__).resolve().parent
A = HERE.parent
W = A / 'whiteboard-ocr'
RUN = W / 'outputs/jh_full_photos_app'
TARGET = W / 'notebooks/16_jh_full_photo_app_evaluation.ipynb'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
norm = lambda s: ' '.join(s.lower().split())
esc = lambda s: html.escape(str(s))


def verify():
    result = json.loads((HERE / 'RESULTS.json').read_text())
    protocol = json.loads((HERE / 'PROTOCOL.json').read_text())
    run = json.loads((RUN / 'run.json').read_text())
    place = json.loads((HERE / 'dataset_crop_placement.json').read_text())
    lines = json.loads((HERE / 'dataset_line_quads.json').read_text())
    readings = json.loads((RUN / 'dataset_crop_readings.json').read_text())
    assert len(place) == len(lines) == len(readings) == 59
    assert sha(HERE / 'dataset_crop_placement.json') == protocol['hashes']['placement']
    assert sha(HERE / 'dataset_line_quads.json') == protocol['hashes']['line_quads']
    assert run['reader_identity']['model_id'] == 'S1'
    assert run['reader_identity']['image_shape'] == [3, 64, 1024]
    assert run['labels_used'] is False and run['no_training'] and not run['app_modified']
    paths = {'runner_sha256': W/'scripts/run_jh_full_photos_app.py',
             'shared_runner_sha256': W/'scripts/run_detector_comparison.py',
             'backend_sha256': W/'ocr_prototype/ocr_backend.py',
             'text_lines_sha256': W/'preprocessing/text_lines.py'}
    assert all(sha(path) == run['cache_key'][k] for k,path in paths.items())
    for name,h in run['cache_key']['images'].items():
        assert sha(A/'JH_datasetNov5Sentence'/name) == h
    for p in place:
        assert sha(A/p['crop']) == p['crop_sha256']
    labels = {r['id']:r['label'] for r in csv.DictReader((A/'board2notes_newwriter_test_20261107/manifest.csv').open())}
    old = json.loads((A/'claude_margin_diagnostic_20261109/straighten/RESULTS_STRAIGHTEN.json').read_text())
    old = {r['name']:r for r in old['crops']}
    assert {r['line'] for r in result['one_to_one']} == {p['id'] for p in place}
    edits_checked = 0
    for r in result['one_to_one']:
        assert labels[r['line']] == r['label']
        for arm in ('app','original','tight','straightened'):
            text = r['app_text'] if arm == 'app' else r[arm]['text']
            n = len(norm(r['label']))
            e = Levenshtein.distance(norm(r['label']),norm(text))
            assert r[arm]['errors'] == e == sum(r[arm][k] for k in ('S','D','I'))
            assert r[arm]['chars'] == n and abs(r[arm]['cer']-e/n) < 1e-12
            edits_checked += 1
        assert readings[r['line']]['tight']['text'] == old[r['line']]['m']['S1']['T']['pred']
        assert readings[r['line']]['straightened']['text'] == old[r['line']]['m']['S1']['R']['pred']
    # Reproduce the paired photo-cluster confidence intervals from independently summed differences.
    for arm in ('original','tight','straightened'):
        parts = defaultdict(list)
        for r in result['one_to_one']:
            parts[r['photo']].append(r['app']['cer']-r[arm]['cer'])
        keys = sorted(parts); rng = random.Random(0); sims=[]
        for _ in range(4000):
            values=[v for k in (rng.choice(keys) for _ in keys) for v in parts[k]]
            sims.append(sum(values)/len(values))
        sims.sort()
        assert all(abs(x-y)<1e-12 for x,y in zip([sims[100],sims[3899]], result['summary']['app_minus_'+arm]['ci_photo_bootstrap']))
    # Confirm all saved reader-view PNGs correspond to the saved crop pixels and current app resize.
    sys.path.insert(0, str(W/'ocr_prototype'))
    from ocr_backend import resize_for_reader
    views = 0
    for pid in range(1,8):
        ph=json.loads((RUN/f'Photo_{pid}.json').read_text())
        assert ph['detect_deterministic']
        for r in ph['grouped']:
            image=cv2.imread(str(RUN/'crops'/r['files']['crop']))
            x=resize_for_reader(image,[3,64,1024])
            expected=np.clip((x.transpose(1,2,0)*.5+.5)*255,0,255).round().astype('uint8')
            actual=cv2.imread(str(RUN/'crops'/r['files']['reader']))
            assert np.array_equal(expected,actual)
            views += 1
    # Preserve the verified numerical record while adding only visual-context annotations.
    baseline = json.loads(json.dumps(result))
    for r in baseline['extra']:
        r.pop('visual_context', None)
    numerical_hash = hashlib.sha256(json.dumps(baseline, indent=1, ensure_ascii=False).encode()).hexdigest()
    previous_checks = HERE/'CHECKS_CODEX.json'
    if previous_checks.exists():
        previous = json.loads(previous_checks.read_text())
        assert numerical_hash == previous.get('recognition_results_sha256_without_visual_context', previous['result_sha256'])
    checks=dict(photos=7, matched_dataset_lines=59, independent_CER_checks=edits_checked,
        reader_view_pixels_verified=views, all_previous_tight_and_straightened_readings_identical=True,
        source_hashes_and_backend_hashes_match=True, paired_photo_intervals_reproduced=True,
        full_page_CER_not_evaluated=True, original_app_unchanged=True,
        placement_min_matches=min(p['matches'] for p in place),
        placement_min_inliers=min(p['inliers'] for p in place),
        placement_min_pixel_correlation=min(p['pixel_correlation'] for p in place),
        analysis_sha256=sha(HERE/'analyse_jh.py'), result_sha256=sha(HERE/'RESULTS.json'),
        recognition_results_sha256_without_visual_context=numerical_hash,
        visual_context_correction='Photo 5 is a projected Windows taskbar, not handwriting; raw OCR and scores preserved.')
    (HERE/'CHECKS_CODEX.json').write_text(json.dumps(checks,indent=2)+'\n')
    return result, checks


def data_url(path):
    mime='image/png' if path.suffix=='.png' else 'image/jpeg'
    return 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()


def gallery(result):
    body=['<!doctype html><meta charset="utf-8"><title>JH app crop review</title>',
          '<style>body{font:16px system-ui;max-width:1500px;margin:25px auto;padding:0 15px;background:#f5f7fa}article{padding:18px;background:white;border:1px solid #ccd3da;border-radius:10px;margin:20px 0}table{width:100%;table-layout:fixed;border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd;vertical-align:top;overflow-wrap:anywhere}img{max-width:100%;height:auto}small{color:#52616b}</style>',
          '<h1>Same JH line, four input crops, the same S1 reader</h1>',
          '<p>Saved outputs only. All 59 matched dataset lines are shown; no outcome-based filtering. These are development photos of one writer. Matching is geometric, with provisional assistant-reviewed bands. No full-page accuracy or reading-order claim.</p>']
    for r in sorted(result['one_to_one'],key=lambda r:(r['photo'],r['region'])):
        cols=[]
        for arm in ('original','tight','straightened','app'):
            path=A/r[arm]['file'] if arm!='app' else RUN/'crops'/r['app_files']['crop']
            text=r[arm]['text'] if arm!='app' else r['app_text']
            cols.append(f'<td><img src="{data_url(path)}" alt="{arm} crop"><p>{esc(text)}</p><small>CER {r[arm]["cer"]:.2%}; deletions {r[arm]["D"]}</small></td>')
        body += [f'<article id="line-{esc(r["line"])}"><h2>Photo {r["photo"]}, app region {r["region"]}</h2><small>{esc(r["line"])}</small><p>Label: <b>{esc(r["label"])}</b></p>',
            '<table><tr><th>Dataset original</th><th>Dataset tightened</th><th>Dataset straightened</th><th>App detector crop</th></tr><tr>'+''.join(cols)+'</tr></table>',
            '<details><summary>Exact app reader input, rendered from the normalised tensor (grey is padding)</summary>',
            f'<img src="{data_url(RUN/"crops"/r["app_files"]["reader"])}" alt="Actual S1 input"></details></article>']
    body += ['<h2>Regions with no matching dataset line</h2><p>Unmatched means outside this dataset-line inventory; it does not automatically mean a false detection.</p>']
    for r in result['extra']:
        body += [f'<article>Photo {r["photo"]}, region {r["region"]}; saved OCR reading: {esc(repr(r["text"]))}<p>{esc(r.get("visual_context", "Context not classified."))}</p><img src="{data_url(RUN/"crops"/r["files"]["crop"])}"></article>']
    (HERE/'review.html').write_text('\n'.join(body))


def report(result, checks):
    s=result['summary'];rows=[]
    for arm in ('original','tight','straightened','app'):
        rr=result['one_to_one'];corpus=sum(r[arm]['errors'] for r in rr)/sum(r[arm]['chars'] for r in rr)
        rows.append(f'| {arm} | {s["cer_"+arm]:.2%} | {corpus:.2%} | {s["D_"+arm]} | {s["persist15_"+arm]} |')
    text='''# JH full-photo app evaluation — completed 9 November 2026

The unchanged app pipeline already avoids most of the poor dataset-crop geometry on these seven JH photographs. S1 receives much wider, nearly horizontal writing when the mobile detector supplies the crop. No automatic tightening, additional deskew, retraining or application change was introduced.

## Matched-line recognition result

All 59 supplied dataset lines have a one-to-one geometric association after grouping, so no supplied line was excluded from this comparison. The matching uses frozen image-only bands and a heuristic overlap graph, not recognised text. This is not certified detector recall. Crop labels and full-photo annotation scope remain subject to the earlier review limitations.

All four inputs were read by the same S1 object in the same Mac process, using the app's resize and dictionary.

| Input | Mean line CER | Corpus CER | Deleted characters | Lines with CER >=15% |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

The app reads 46/59 matched lines perfectly under case-insensitive, collapsed-whitespace scoring. There are 20 character edits in 2,355 reference characters. These denominators exclude output with no matched dataset line; this is **not full-page CER** or “99% accurate on arbitrary whiteboards”.

## Detection and grouping

- Raw detector: 58 one-to-one associations, one split line, two unmatched regions.
- After the existing grouping code: 59 one-to-one associations and two unmatched regions; no merge/split/missed-line components under the frozen overlap rule.
- On Photo 2, grouping joins the detector's two pieces of “Sunlight warms the quiet forest” into one line. The result is read correctly. This shows useful existing grouping; no grouping code was changed.
- Photo 2's unmatched box covers the black board eraser and produces an empty reading. Photo 5's unmatched box covers the projected Windows taskbar (magnifier icon and “Search”), not handwriting. S1's saved reading remains “Q 5earch”. It is text-like projected interface content outside the handwritten-note target inventory, and would be unwanted output when the intended scope is handwritten notes. The matched-line score excludes both unmatched regions and is not a full-page score.

**Context correction, 9 November:** the earlier report misidentified the Photo 5 projection as faint handwriting. The wider image `q_search_check/q_search_context.jpg` shows the Windows logo, magnifier and Search box. This correction changes only the visual interpretation; it does not change source images, OCR predictions, matching or scores.

## What changed in the input

Median resized crop widths at height 64: dataset original 342 px; tightened 386 px; straightened 500 px; app crop 814 px. Median absolute residual slope of app crops is 0.4 degrees. Existing quad cropping rectifies the detector's tilted text boxes.

The full photos also contain a higher-resolution source: dataset crops map into them at approximately 1.59x scale and involve different image sampling/compression. Thus this comparison supports the **actual full-photo crop path**, but does not isolate geometry, horizontal scale, neighbouring ink, source resolution or compression as the sole cause.

The four JH crops still difficult after the manual straightening experiment (JH1.4, JH5.3, JH6.4, JH7.3) are all below 15% CER through the app. Inspect the gallery for exact readings and remaining errors; no labels were corrected.

## Uncertainty and provenance

Seven photographs from one already-inspected writer are development evidence. The intervals resample photos, holding this single S1 checkpoint fixed; they do not capture future-writer or training-seed uncertainty. There is no full-photo reading-order evaluation here, consistent with the user's decision to defer that work.

The earlier handoff's “at least 160 matches” claim is inaccurate. Saved placement records have a minimum of 110 matches, 97 RANSAC inliers and 0.8017 pixel correlation. These are sufficient to investigate, but the assistant-reviewed placements are not independent gold annotations. All 59 four-way image comparisons and Photo 2's split/grouping example were visually inspected for this analysis.

Codex completed Claude's existing analysis script with only portable path/font handling changes so it runs on the Mac. Matching thresholds, scoring and the saved detector/reader outputs were not changed. CHECKS_CODEX.json records the current analysis hash and independent checks. All 61 saved app reader-view images were reproduced exactly from saved crop pixels; all 236 line/arm CER records were recomputed. The seven-photo sources, code hashes and frozen placement hashes match the saved run identities. The repeated tight/straightened dataset readings match the earlier S1 readings.

## Decision

Retain the existing mobile-detector/S1 crop path for now. There is no demonstrated need to add another automatic margin or straightening stage on these photos. Do not replace historical benchmark scores: report the original-crop, transformed-crop and actual-app results with their distinct scopes.

The next model-side decision should use this corrected diagnosis, not the earlier claim that S1 cannot read JH handwriting. A practical next milestone is freeze this OCR configuration and check it on fresh photos/writers, while completing deployment. Reading-order changes remain deferred. This result alone does not settle faint short words, other writers, distant capture or complex layouts.

## View and reproduce

- `review.html`: all 59 four-way crop comparisons, full saved readings, exact app input previews and unmatched regions.
- `visuals/`: seven raw/grouped overlays, seven crop/input strips and five four-way contact sheets.
- `RESULTS.json`: original analysis output, per-line records and photo-cluster intervals.
- `CHECKS_CODEX.json`: independent scoring, identity and pixel checks.
- `../whiteboard-ocr/notebooks/16_jh_full_photo_app_evaluation.ipynb`: executed notebook, no model loading.

Run `Assignment/.venv-paddle/bin/python Assignment/claude_jh_fullphoto_20261109/analyse_jh.py` to regenerate the saved-output analysis. Run `build_notebook16.py` afterward to verify and rebuild the notebook. Neither starts inference or training.
'''
    (HERE/'REPORT.md').write_text(text)


def notebook(result):
    nb=nbf.v4.new_notebook();nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    nb.metadata['board2notes_generated_by']='build_notebook16.py'
    nb.cells=[
        nbf.v4.new_markdown_cell('''# 16 · Does the real app avoid the JH dataset-crop problem?

**Yes on these seven development photos:** the same S1 reads the app crops much better than the supplied upright dataset crops. This notebook displays saved outputs; it starts no training, inference or downloads.

The unchanged path is **PP-OCRv5_mobile_det → existing grouping → existing quad crop → S1**. Reading-order work remains deferred. Matching uses image-only geometry and provisional line bands. Recognition scores concern matched dataset lines, not complete pages or unseen future writers.'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json, html, re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, HTML, Markdown, Image as IPImage
roots=[q for p in [Path.cwd(),*Path.cwd().parents] for q in (p,p/'Assignment')]
A=next(p for p in roots if (p/'claude_jh_fullphoto_20261109/RESULTS.json').is_file())
D=A/'claude_jh_fullphoto_20261109'
RUN=A/'whiteboard-ocr/outputs/jh_full_photos_app'
R=json.loads((D/'RESULTS.json').read_text()); S=R['summary']
CHECKS=json.loads((D/'CHECKS_CODEX.json').read_text())
def table(headers,rows):
    e=lambda x:html.escape(str(x))
    display(HTML('<table><tr>'+''.join('<th style="padding:8px">'+e(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td style="padding:8px;border-bottom:1px solid #ddd">'+e(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'))
arms=['original','tight','straightened','app']
table(['Input','Matched lines','Mean line CER','Deletions','Lines at >=15% CER'],[[k,S['n_one_to_one'],f"{S['cer_'+k]:.2%}",S['D_'+k],S['persist15_'+k]] for k in arms])
print('Model: S1 seed 1. Same reader object, all four input types. Original app unchanged.')'''),
        nbf.v4.new_markdown_cell('''## 1. What this means

The recognizer did not suddenly learn new handwriting. It received better input from the existing detector/cropping path. The app cuts along tilted boxes, so the writing fills more of the resized image.

This comparison also changes source resolution and sampling: the full photos contain more pixels than the dataset crops. We cannot assign the entire benefit to one geometric mechanism. All 59 supplied lines matched, so none was silently dropped from this recognition table.'''),
        nbf.v4.new_code_cell('''fig,ax=plt.subplots(1,2,figsize=(12,3.7))
vals=[100*S['cer_'+k] for k in arms]
ax[0].bar(arms,vals,color=['#94a3b8','#71849c','#3f7d9c','#26786c'])
for i,y in enumerate(vals):ax[0].text(i,y+.35,f'{y:.2f}%',ha='center')
ax[0].set_ylabel('Mean line CER (%) — lower is better');ax[0].set_title('Same 59 lines, same frozen reader')
widths=[S['median_width64'][k] for k in arms]
ax[1].bar(arms,widths,color=['#94a3b8','#71849c','#3f7d9c','#26786c'])
for i,y in enumerate(widths):ax[1].text(i,y+10,f'{y:.0f}',ha='center')
ax[1].set_ylim(0,950);ax[1].set_ylabel('Median resized crop width at 64 px height');ax[1].set_title('The app supplies wider writing')
fig.tight_layout();plt.show()
table(['App minus dataset input','CER difference (pp)','Photo-cluster 95% interval (pp)','App better/worse'],[[k,f"{100*S['app_minus_'+k]['diff']:+.2f}",' to '.join(f'{100*v:+.2f}' for v in S['app_minus_'+k]['ci_photo_bootstrap']),f"{S['app_minus_'+k]['app_better']} / {S['app_minus_'+k]['app_worse']}"] for k in arms[:3]])'''),
        nbf.v4.new_markdown_cell('''## 2. Detection outcomes stay separate from recognition

The raw detector splits one supplied line into two pieces; existing grouping repairs that split. Two other regions have no match in the dataset-line inventory. These outcomes are based on a heuristic geometric graph, not standard detector recall. A match does not prove every character stroke is covered.

All photos and all matched lines remain visible below. Grey bands in overlays are reference geometry; green means one-to-one association; blue is a split candidate; red is unmatched.'''),
        nbf.v4.new_code_cell('''table(['Photo','Dataset lines','Raw boxes','App regions','Raw outcomes','Grouped outcomes'],[[pid,p['lines'],p['raw_boxes'],p['regions'],p['raw'],p['grouped']] for pid,p in R['per_photo'].items()])
for name in R['visuals']['overlays']:
    display(IPImage(filename=str(D/'visuals'/name),width=1100))'''),
        nbf.v4.new_markdown_cell('''## 3. Every app crop and its actual reader input

Left: crop pixels cut from the original photo. Right: saved rendering of the actual normalised S1 input, including grey padding. Wide inputs are fitted for display, so the displayed height may be smaller than 64 px. The stored PNG has the actual tensor dimensions. Captions show complete saved text where space permits; use the exact-line viewer below for untruncated readings.'''),
        nbf.v4.new_code_cell('''for name in R['visuals']['strips']:
    display(IPImage(filename=str(D/'visuals'/name),width=1200))'''),
        nbf.v4.new_markdown_cell('''## 4. Four versions of every matched line

The contact sheets show all 59 lines, sorted by the change from tightened input to app input. Their captions may truncate long strings. The following viewer shows the full text for any line, with the exact input image available under its expandable control.'''),
        nbf.v4.new_code_cell('''for name in R['visuals']['compare']:
    display(IPImage(filename=str(D/'visuals'/name),width=1300))'''),
        nbf.v4.new_code_cell('''gallery=(D/'review.html').read_text()
def show_line(line_id):
    found=re.search(r'<article id="line-'+re.escape(html.escape(line_id))+r'">.*?</article>',gallery,re.S)
    if found is None:raise KeyError(line_id)
    display(HTML('<style>table{width:100%;table-layout:fixed}td{vertical-align:top;overflow-wrap:anywhere;padding:8px}img{max-width:100%;height:auto}</style>'+found.group(0)))
LINE='jhs__train__JH6.4.jpg'  # Replace with any line ID from the table below.
show_line(LINE)
table(['Line ID','Label','App reading','App CER'],[[r['line'],r['label'],r['app_text'],f"{r['app']['cer']:.2%}"] for r in sorted(R['one_to_one'],key=lambda r:(r['photo'],r['region']))])'''),
        nbf.v4.new_markdown_cell('''## 5. Unmatched regions are not hidden

Photo 2 includes the black eraser: the detector proposes it, and S1 returns no text. Photo 5 includes the **projected Windows taskbar**, with a magnifier icon and “Search”, read as “Q 5earch”. It is not handwriting; the wider context image below establishes this. These regions are outside the 59-line handwritten-note inventory and are not included in matched-line CER. That is why 0.87% is **not a complete-page accuracy score**.

Correction dated 9 November: an earlier version incorrectly called the Photo 5 projection faint handwriting. Predictions and measured scores are unchanged.'''),
        nbf.v4.new_code_cell('''for r in R['extra']:
    print('Photo',r['photo'],'region',r['region'],'saved reading:',repr(r['text']))
    print('Visual context:',r.get('visual_context','Context not classified.'))
    display(IPImage(filename=str(RUN/'crops'/r['files']['crop']),width=650))
display(Markdown('**Photo 5 context: projected Windows taskbar**'))
display(IPImage(filename=str(D/'q_search_check/q_search_context.jpg'),width=900))
print('Matched recognition exact lines:',sum(r['app']['errors']==0 for r in R['one_to_one']),'/ 59')
print('Independent checks:',CHECKS)'''),
        nbf.v4.new_markdown_cell('''## 6. Decision and limits

**Keep the app unchanged for now.** This collection does not justify adding another automatic tightening or deskew stage. The existing detector and rotated crop already avoid most of the problematic upright-dataset geometry.

- These seven photos belong to one inspected development writer; no broad new-writer guarantee follows.
- The 59-line inventory is incomplete for all visible page content. Geometry is assistant-reviewed, not certified independent ground truth.
- The original dataset crops are lower-resolution samples than the full photos. Source detail, interpolation, crop extent and orientation differ together.
- Saved placement records have a minimum of **97 inliers**, not the handoff's claimed 160. The image matches and all four-way comparisons were inspected; do not present them as perfect ground truth.
- The model can accept widths above 1024; 1024 is a minimum here. Wider inputs may produce more than 128 output positions.
- Keep historical results and report the input version explicitly. No labels or previous scores were overwritten.
- No reading-order changes were made or evaluated. That work remains deferred as requested.

Next milestone: freeze the current OCR configuration and confirm on fresh writer/photo groups while progressing local/cloud deployment. Do not start another architecture experiment based on the old loose-crop scores alone.

Full report and self-contained crop gallery: `Assignment/claude_jh_fullphoto_20261109/REPORT.md` and `review.html`.''')
    ]
    if TARGET.exists():
        prior=nbf.read(TARGET,as_version=4)
        if prior.metadata.get('board2notes_generated_by')!='build_notebook16.py':
            raise RuntimeError('Notebook already exists and is not owned by this builder; preserve it.')
    nbf.write(nb,TARGET)  # Keep the usable notebook even if local-kernel permissions block execution.
    NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(TARGET.parent)}}).execute()
    nbf.validate(nb);nbf.write(nb,TARGET)
    print('Executed notebook:',TARGET)


if __name__=='__main__':
    result,checks=verify()
    gallery(result)
    report(result,checks)
    notebook(result)
