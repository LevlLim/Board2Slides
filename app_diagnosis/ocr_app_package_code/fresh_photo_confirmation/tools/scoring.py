"""Scoring rules for the fresh-photo confirmation. They are copied from notebook 15 so the kit needs no other folder:
whiteboard-ocr/scripts/score_jy_annotated_photos.py (norm, overlap, evaluate_photo) and
evaluate_jy_full_pages.py (page_score).

    python tools/scoring.py --self-test     also checks against the originals when whiteboard-ocr/scripts is present
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

IOU_THRESHOLD = .25
INTERSECTION_MIN_AREA_THRESHOLD = .50

def _pure(a, b):
    """Plain Levenshtein distance (used when rapidfuzz is not installed; same result, slower)."""
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


try:
    from rapidfuzz.distance import Levenshtein as _L
    def levenshtein(a, b): return _L.distance(a, b)
except ImportError:
    levenshtein = _pure


def norm(s):
    return ' '.join(str(s).lower().split())


def page_score(reference, hypothesis):
    ref, hyp = norm(reference), norm(hypothesis)
    if not ref:
        raise ValueError('Reference must contain text')
    return levenshtein(ref, hyp), len(ref)


def overlap(a, b):
    import cv2, numpy as np
    a = cv2.convexHull(np.asarray(a, np.float32)); b = cv2.convexHull(np.asarray(b, np.float32))
    aa, ab = cv2.contourArea(a), cv2.contourArea(b)
    inter, _ = cv2.intersectConvexConvex(a, b)
    return dict(iou=float(inter / max(aa + ab - inter, 1e-9)), intersection_min_area=float(inter / max(min(aa, ab), 1e-9)))


def evaluate_photo(gt, pred):
    """gt: [{id, text, quad, proposed_reading_order}], pred: [{index, box, text}]. Bipartite overlap graph; text never used."""
    n, m = len(gt), len(pred)
    nb = {i: set() for i in range(n + m)}
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            ov = overlap(g['quad'], p['box'])
            if ov['iou'] >= IOU_THRESHOLD or ov['intersection_min_area'] >= INTERSECTION_MIN_AREA_THRESHOLD:
                nb[i].add(n + j); nb[n + j].add(i)
    seen, comps = set(), []
    for node in range(n + m):
        if node in seen: continue
        queue, found = [node], set()
        while queue:
            k = queue.pop()
            if k in found: continue
            found.add(k); queue.extend(nb[k] - found)
        seen |= found
        gi = sorted((k for k in found if k < n), key=lambda i: gt[i]['proposed_reading_order'])
        pi = sorted((k - n for k in found if k >= n), key=lambda i: pred[i]['index'])
        ref = norm(' '.join(gt[i]['text'] for i in gi)); hyp = norm(' '.join(pred[i]['text'] for i in pi))
        kind = ('unmatched_prediction' if not gi else 'missed_target' if not pi else 'one_to_one' if len(gi) == len(pi) == 1
                else 'merge_candidate' if len(pi) == 1 else 'split_candidate' if len(gi) == 1 else 'many_to_many_candidate')
        comps.append(dict(kind=kind, gt_ids=[gt[i]['id'] for i in gi], prediction_indices=[pred[i]['index'] for i in pi],
                          reference=ref, prediction=hyp, reference_characters=len(ref), errors=levenshtein(ref, hyp)))
    counts = {k: sum(c['kind'] == k for c in comps) for k in (
        'one_to_one', 'merge_candidate', 'split_candidate', 'many_to_many_candidate', 'missed_target', 'unmatched_prediction')}
    inc = [c for c in comps if c['gt_ids']]
    one = [c for c in comps if c['kind'] == 'one_to_one']
    summary = dict(annotated_targets=n, detected_regions=m, **counts,
                   missed_targets=sum(not nb[i] for i in range(n)), unmatched_predictions=sum(not nb[n + j] for j in range(m)),
                   annotated_target_errors=sum(c['errors'] for c in inc), annotated_target_characters=sum(c['reference_characters'] for c in inc),
                   matched_line_errors=sum(c['errors'] for c in one), matched_line_characters=sum(c['reference_characters'] for c in one))
    return summary, comps


def reordered_page(gt, comps, reference):
    rank = {l['id']: l['proposed_reading_order'] for l in gt}
    ordered = sorted(comps, key=lambda c: min((rank[i] for i in c['gt_ids']), default=10**6))
    return page_score(reference, ' '.join(c['prediction'] for c in ordered))[0]


def bootstrap(per, B=10000, seed=0, level=.95):
    """Percentile interval of pooled page CER, resampling photos."""
    rng = random.Random(seed); keys = list(per); vals = []
    for _ in range(B):
        s = [per[rng.choice(keys)] for _ in keys]
        vals.append(sum(p['errors'] for p in s) / sum(p['chars'] for p in s))
    vals.sort(); lo = vals[int((1 - level) / 2 * B)]; hi = vals[int((1 + level) / 2 * B) - 1]
    return lo, hi


def self_test():
    assert norm('  Hello\n  World ') == 'hello world'
    assert levenshtein('kitten', 'sitting') == 3 and page_score('Ab c', 'ab  d') == (1, 4)
    sq = lambda x, y, w=100, h=20: [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    gt = [dict(id='a', text='one', quad=sq(0, 0), proposed_reading_order=0), dict(id='b', text='two', quad=sq(0, 40), proposed_reading_order=1),
          dict(id='c', text='three', quad=sq(0, 80), proposed_reading_order=2), dict(id='d', text='four', quad=sq(300, 0), proposed_reading_order=3)]
    pred = [dict(index=1, box=sq(0, 0, h=60), text='one two'), dict(index=2, box=sq(0, 80), text='thre'), dict(index=3, box=sq(600, 600), text='x')]
    s, c = evaluate_photo(gt, pred)
    assert (s['merge_candidate'], s['one_to_one'], s['missed_target'], s['unmatched_prediction']) == (1, 1, 1, 1), s
    assert (s['matched_line_errors'], s['matched_line_characters']) == (1, 5)
    print('self-test: rules behave as specified')
    scripts = Path(__file__).resolve().parents[3] / 'whiteboard-ocr' / 'scripts'
    if scripts.is_dir():
        sys.path.insert(0, str(scripts))
        try:
            import score_jy_annotated_photos as o
            from evaluate_jy_full_pages import page_score as o_page
        except ImportError as exc:
            print(f'originals present but not importable here ({exc}); comparison skipped'); return
        assert (o.IOU_THRESHOLD, o.INTERSECTION_MIN_AREA_THRESHOLD) == (IOU_THRESHOLD, INTERSECTION_MIN_AREA_THRESHOLD)
        os_, oc = o.evaluate_photo(gt, pred)[:2]
        assert [(x['kind'], x['errors']) for x in oc] == [(x['kind'], x['errors']) for x in c]
        assert all(os_[k] == s[k] for k in os_ if k in s), (os_, s)
        assert o_page('Ab c', 'ab  d') == page_score('Ab c', 'ab  d')
        from rapidfuzz.distance import Levenshtein as RL
        rng = random.Random(1)
        for _ in range(300):
            a = ''.join(rng.choice('abc d') for _ in range(rng.randint(0, 30))); b = ''.join(rng.choice('abd e') for _ in range(rng.randint(0, 30)))
            assert levenshtein(a, b) == RL.distance(a, b) == _pure(a, b)
        print('self-test: identical to the notebook 15 originals in', scripts)
    else:
        print('notebook 15 originals not found next to the package; comparison skipped')


if __name__ == '__main__':
    if '--self-test' in sys.argv: self_test()
    else: print(__doc__)
