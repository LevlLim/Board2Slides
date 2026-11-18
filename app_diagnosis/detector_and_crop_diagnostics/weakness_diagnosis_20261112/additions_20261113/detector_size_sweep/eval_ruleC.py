"""Run the IMPLEMENTED large-writing rule (appfix copy of the app's text_lines.py) on every dev photo variant.
Non-triggered photos take the unchanged app path, so their boxes are reused from det.json['app'] after checking a few
for identity; triggered ones are detected again with the new code."""
import sys, os, json, time
sys.path.insert(0, '/home/claude/appfix/b2n_ocr'); os.environ['B2N_PACKAGE_ROOT'] = '/home/claude/fresh/pkg'
import numpy as np, cv2
import ocr_backend as ob
from preprocessing import text_lines as tl
sys.path.insert(0, '/home/claude/sweep'); import common
assert 'SIZE_RULE' in open(tl.__file__).read() and tl.SIZE_RULE == 'line_height', tl.__file__
refs = json.load(open('refs.json')); det = json.load(open('det.json'))
path_of = {r['pid']: r['photo'] for r in refs}
d = tl.Detector(ob.DETECTOR_NAME, det_args={'thresh': .20, 'box_thresh': .40, 'unclip_ratio': 1.2}, model_dir=ob.DETECTOR_DIR, runtime=ob.DETECTOR_RUNTIME)
out = json.load(open('det_ruleC.json')) if os.path.exists('det_ruleC.json') else {}
checked = 0
for vid, rec in det.items():
    if vid in out: continue
    pid, k = vid.rsplit('@', 1)
    img, _, _ = common.variant(common.load(path_of[pid]), int(k))
    t = time.time()
    first = d.boxes(img, limit=tl.FIRST_LIMIT, unclip=1.2)
    full_h = float(np.median([tl.line_height(p) for p in first])) if first else 0   # same estimate as the app code
    trig = full_h > tl.BIG_LINE_H
    if trig or checked < 3:
        b, info = d.detect(img, unclip=1.2); b, _ = tl.group_into_lines(b)
        boxes = [np.asarray(p).astype(float).tolist() for p in b]
        trig = info['path'].startswith('large writing')
        if not trig:
            same = boxes == rec['cfg']['app']; checked += 1
            assert same, f'{vid}: untriggered path differs from the old app boxes'
    else:
        boxes, info = rec['cfg']['app'], dict(path='unchanged app path (boxes reused)')
    out[vid] = dict(boxes=boxes, triggered=trig, line_px_full=round(full_h, 1), path=info['path'])
    json.dump(out, open('det_ruleC.json', 'w'))
    print(vid, 'TRIGGERED' if trig else '-', round(full_h), info['path'][:60], f'{time.time()-t:.0f}s', flush=True)
for vid, v in out.items(): det[vid]['cfg']['ruleC'] = v['boxes']
json.dump(det, open('det_with_ruleC.json', 'w'))
print('EVAL DETECT DONE', sum(v['triggered'] for v in out.values()), 'of', len(out), 'variants triggered', flush=True)
