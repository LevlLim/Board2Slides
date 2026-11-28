"""Ground truth and the matching of detected lines to it (label = the human transcription, never the reader).

Ground truth (one ordered list of lines per photo, names as in board2notes_dataset_combined/images):
  * Airdrop photos (108): labels/board2notes_labels_v1_textonly.xlsx, sheet "Line Labels" (English lines only).
  * 17 Oct photos (41 curated): 17oct_rec_paddle/17oct_gt_lines.json (IMG_xxxx names -> oct17_xxx.png through
    labels/oct17_name_map.csv). Erased lines and non-text lines are left out.
  * The other 101 17 Oct photos have no labels: they are preprocessed, but give no training crops.

Matching (match_lines): which detected line shows which ground-truth line?
  The old crop set kept a crop only if the PRETRAINED reader already read it with CER <= 0.25. That keeps the
  lines the reader can read and throws away exactly the hard ones we want to train on (selection bias). Here the
  reader's text is used only to find the PAIRING; whether a pairing is trusted also uses evidence that does not
  depend on the reader being right:
    1. pairing: one-to-one assignment (Hungarian) minimising the line CER between reading and label;
       a pair needs CER < 1 (the reading shares something with the label).
    2. order support: the neighbouring detected line (previous or next in reading order) is a CONFIDENT pair
       (CER <= ACCEPT_CER) with the neighbouring label line, and this pair's CER is at most ORDER_MAX_CER.
       Handwritten notes are read in order, so a pairing that sits next to a sure one is right even if the
       reading is poor. (Only confident neighbours count: two poor readings next to each other prove nothing.)
    3. shape check: crop width / (height x label length) within a factor RATIO_TOL of the photo's median, so a
       box that shows half a line (or two lines) is not given the full label.
    4. ends check (v2): the reading must cover the label end to end. A crop that shows only some WORDS of its label
       line (the detector split the line, e.g. at an arrow) is labelled with those words if the reading matches
       them (CER <= ACCEPT_CER; label_span=True); otherwise, and for a crop with extra text around its label, it
       goes to review even when its CER is low.
    5. text rows (v3): a crop with two or more text rows (the detector merged lines) goes to review.
  status  see accept_status(): ends OK, one row, and (CER <= ACCEPT_CER, or order support with the shape OK)
          review   : paired, but neither strong reading nor order support, or shape off  -> contact sheet
  Thresholds are fixed here, before looking at the results.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ACCEPT_CER = 0.34     # pairing trusted on the reading alone (about 2 wrong characters in 6)
ORDER_MAX_CER = 0.6   # pairing trusted through order support only if the reading still shares 40 % (v2: was 0.75)
RATIO_TOL = 1.4       # shape check: allowed factor between a crop's width-per-character and the photo median (v2: 1.6)
SPAN_LABELS = False   # label a cut-off crop with the words it shows? Tried 23 Oct: 5 of 16 checked were wrong
                      # (the reader misses trailing words, two-line crops), so it is OFF: such crops go to review
ENDS_CHARS = 3        # ends check: at least this many label characters unexplained at the start/end of the reading
# v2 (23 Oct), after looking at the first build's accepted crops (label quality control, not a test result):
#   * "3. Viewpoint" was accepted with the label "3. Viewpoint Views" (CER 0.33): the detector had cut the line,
#     the crop shows only part of the label. Same for "controls motor speed" / "PWM - controls motor speed",
#     "Hubel and Wiesel" / "1959: Hubel and Wiesel", "by weights and" / "... biases". -> ends check.
#   * a printed sign "Notice to Candidates" got the label "Dropout, Batch Normalization" through order support at
#     CER 0.71. -> ORDER_MAX_CER 0.75 -> 0.6.
#   * the shape tolerance 1.6 let a crop with two thirds of its label's characters through. -> RATIO_TOL 1.4.
MIN_RATIO_LINES = 3   # photos with fewer confident pairs use the dataset median for the shape check


# ---------------------------------------------------------------- text normalisation and error rates (team rule)
def norm(s: str) -> str:
    """Team line-CER normalisation: lowercase, single spaces, trimmed."""
    return " ".join(str(s).lower().split())


def levenshtein(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def substring_distance(short, long):
    """Edit distance between `short` and the best-matching SUBSTRING of `long` (free gaps at both ends of long)."""
    prev = [0] * (len(long) + 1)
    for i, ca in enumerate(short, 1):
        cur = [i]
        for j, cb in enumerate(long, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return min(prev)


def ends_problem(reading: str, label: str) -> str:
    """'' if the reading covers the label end to end; otherwise what is wrong:
    'part of the label'  : the reading matches a piece of the label well, and the rest of the label (at least
                           ENDS_CHARS characters at its start or end) is not explained -> the crop is cut off;
    'extra text'         : the label matches a piece of the reading, and the reading has at least ENDS_CHARS more
                           characters around it -> the crop also shows a neighbouring word or line."""
    r, g = norm(reading), norm(label)
    if not r or not g:
        return ""
    full = levenshtein(r, g)
    in_label = substring_distance(r, g)        # reading inside the label
    if full - in_label >= ENDS_CHARS and in_label / max(len(r), 1) <= ACCEPT_CER:
        return "part of the label"
    in_reading = substring_distance(g, r)      # label inside the reading
    if full - in_reading >= ENDS_CHARS and in_reading / max(len(g), 1) <= ACCEPT_CER:
        return "extra text"
    return ""


def best_word_span(reading: str, label: str):
    """The run of whole label words that the reading matches best: (span text, its CER). Used when a crop shows
    only part of its label line (the detector split the line, e.g. at an arrow): the crop is then labelled with
    the words it shows, not the whole line."""
    words = label.split()
    best = (label, line_cer(reading, label))
    for a in range(len(words)):
        for b in range(a + 1, len(words) + 1):
            span = " ".join(words[a:b])
            c = line_cer(reading, span)
            if c < best[1] or (c == best[1] and len(span) > len(best[0])):
                best = (span, c)
    return best


def line_cer(pred: str, gt: str) -> float:
    g, p = norm(gt), norm(pred)
    return min(1.0, levenshtein(p, g) / max(len(g), 1))


# ---------------------------------------------------------------- ground truth
def load_ground_truth(dataset_dir, oct17_json=None):
    """{photo file name: [dict(line_id, text), ...] in label order}.
    dataset_dir: Assignment/board2notes_dataset_combined; oct17_json: Assignment/17oct_rec_paddle/17oct_gt_lines.json
    """
    import openpyxl
    dataset_dir = Path(dataset_dir)
    gt = defaultdict(list)
    wb = openpyxl.load_workbook(dataset_dir / "labels" / "board2notes_labels_v1_textonly.xlsx", read_only=True)
    rows = wb["Line Labels"].iter_rows(values_only=True)
    head = next(rows)
    col = {k: i for i, k in enumerate(head)}
    for r in rows:
        if r[col["record_type"]] != "GT line" or not r[col["ground_truth"]]:
            continue
        text = " ".join(str(r[col["ground_truth"]]).split())
        if text:
            gt[r[col["image_id"]]].append(dict(line_id=str(r[col["line_id"]]), text=text))
    if oct17_json and Path(oct17_json).exists():
        name = {}
        with open(dataset_dir / "labels" / "oct17_name_map.csv", newline="") as f:
            for r in csv.DictReader(f):
                name[Path(r["original_file"]).stem] = r["new_file"]
        for photo in json.loads(Path(oct17_json).read_text()):
            new = name.get(Path(photo["image_id"]).stem)
            if new is None:
                continue
            for ln in photo["lines"]:
                if ln.get("erased") or ln.get("line_type", "text") != "text":
                    continue
                text = " ".join(str(ln["label"]).split())
                if text and all(ord(c) < 128 for c in text):
                    gt[new].append(dict(line_id=ln["line_id"], text=text))
    return dict(gt)


def load_manifest(dataset_dir):
    with open(Path(dataset_dir) / "manifest.csv", newline="") as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def condition_of(row) -> str:
    """Photo condition for the by-condition table: the Airdrop label, or the curated role of a 17 Oct photo."""
    if row.get("condition"):
        return row["condition"]
    if row.get("curated_role"):
        return "17 Oct " + row["curated_role"].replace("_", " ")
    return "unlabelled"


# ---------------------------------------------------------------- pairing
def _assign(cost):
    """One-to-one assignment minimising cost (rows = detections, cols = labels)."""
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    except ImportError:                                   # greedy fallback
        pairs, used_r, used_c = [], set(), set()
        for flat in np.argsort(cost, axis=None):
            i, j = divmod(int(flat), cost.shape[1])
            if i not in used_r and j not in used_c:
                pairs.append((i, j)); used_r.add(i); used_c.add(j)
        return pairs


def pair_lines(readings, gt_texts):
    """readings: reader text per detected line (in reading order). Returns {det index: (gt index, cer)}."""
    if not readings or not gt_texts:
        return {}
    cost = np.array([[line_cer(p, g) for g in gt_texts] for p in readings])
    return {i: (j, float(cost[i, j])) for i, j in _assign(cost) if cost[i, j] < 1.0}


def accept_status(o):
    """v3 (23 Oct): a crop is accepted if it shows exactly its label (ends check: not cut off, no extra text, one
    text row) AND either the reading is close (CER <= ACCEPT_CER) or a confident neighbour supports it AND its shape
    fits. The shape check no longer blocks close readings: it sent correct single lines with an underline or an
    arrow ("CNN Architecture" at CER 0.0, "Pooling" with its arrow) to review, and the two failures it was meant
    to catch (cut-off and two-line crops) now have their own checks."""
    if o["ends"]:
        return "review"
    if o["cer"] <= ACCEPT_CER:
        return "accepted"
    return "accepted" if (o["order_support"] and o["shape_ok"]) else "review"


def match_lines(lines, gt_lines, dataset_ratio=None):
    """lines: [dict(reading=str, width=px, height=px), ...] in READING ORDER (one per detected line).
    gt_lines: [dict(line_id, text)] in label order.
    Returns one dict per detected line: gt_index, line_id, label, cer, order_support, ratio, shape_ok, status.
    dataset_ratio: fallback median width-per-character (in line heights) for photos with few confident pairs.
    """
    texts = [g["text"] for g in gt_lines]
    pairs = pair_lines([l["reading"] for l in lines], texts)
    out = []
    for i, l in enumerate(lines):
        if i not in pairs:
            out.append(dict(gt_index=None, line_id="", label="", cer=None, order_support=False, ratio=None,
                            shape_ok=None, status="unlabelled", ends="", label_span=False, rows=l.get("rows", 1)))
            continue
        j, c = pairs[i]
        # an ANCHOR neighbour: the previous / next detected line is a confident pair with the previous / next label
        support = c <= ORDER_MAX_CER and any(
            k in pairs and abs(pairs[k][0] - j) == 1 and pairs[k][1] <= ACCEPT_CER for k in (i - 1, i + 1))
        label, ends, span = texts[j], ends_problem(l["reading"], texts[j]), False
        if ends == "part of the label" and SPAN_LABELS:
            # the crop shows only some words of the line: label it with those words if the reading matches them
            sp, sc = best_word_span(l["reading"], texts[j])
            if sc <= ACCEPT_CER and len(norm(sp)) >= 2 * ENDS_CHARS and sp != texts[j]:
                label, c, ends, span = sp, sc, "", True
        ratio = (l["width"] / max(l["height"], 1)) / max(len(label), 1)
        rows = int(l.get("rows", 1))
        out.append(dict(gt_index=j, line_id=gt_lines[j]["line_id"], label=label, cer=round(c, 4),
                        order_support=bool(support), ratio=ratio, shape_ok=None, status="",
                        ends=ends or ("two or more lines" if rows >= 2 else ""), label_span=span, rows=rows))
    confident = [o["ratio"] for o in out if o["cer"] is not None and o["cer"] <= ACCEPT_CER and not o["ends"]]
    ref = float(np.median(confident)) if len(confident) >= MIN_RATIO_LINES else dataset_ratio
    for o in out:
        if o["gt_index"] is None:
            continue
        o["shape_ok"] = bool(ref is None or 1 / RATIO_TOL <= o["ratio"] / ref <= RATIO_TOL)
        o["status"] = accept_status(o)
        o["ratio"] = round(o["ratio"], 4)
    return out


# ---------------------------------------------------------------- hurdle metrics (slide 3 / labels Summary sheet)
def hurdle_counts(readings, gt_texts, only=None):
    """Counts for line coverage, exact-line accuracy, CER and WER of one photo, using the same pairing.
    Every ground-truth line counts; a line with no paired reading counts as fully wrong (reading "").
    Readings that pair with no label line (stray boxes) are not counted, as in the hurdle evaluation.
    only: score just these label lines (pairing still uses all of them)."""
    pairs = pair_lines(readings, gt_texts)
    pred_for = {j: readings[i] for i, (j, _) in pairs.items()}
    n = dict(lines=0, covered=0, exact=0, char_err=0, chars=0, word_err=0, words=0)
    for j, g in enumerate(gt_texts):
        if only is not None and j not in only:      # only = indices of the label lines to score (visible lines)
            continue
        n["lines"] += 1
        p = pred_for.get(j, "")
        n["covered"] += j in pred_for
        n["exact"] += norm(p) == norm(g)
        # each line's errors are capped at its length (team rule: line CER capped at 1), so one runaway
        # reading cannot outweigh the rest of the photo
        n["char_err"] += min(levenshtein(norm(p), norm(g)), len(norm(g))); n["chars"] += len(norm(g))
        n["word_err"] += min(levenshtein(norm(p).split(), norm(g).split()), len(norm(g).split()))
        n["words"] += len(norm(g).split())
    return n


def hurdle_rates(counts_list):
    t = defaultdict(int)
    for c in counts_list:
        for k, v in c.items():
            t[k] += v
    return dict(photos=len(counts_list), gt_lines=t["lines"],
                line_coverage=t["covered"] / max(t["lines"], 1), exact_line_accuracy=t["exact"] / max(t["lines"], 1),
                cer=t["char_err"] / max(t["chars"], 1), wer=t["word_err"] / max(t["words"], 1))
