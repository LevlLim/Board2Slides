"""From photos to a PaddleOCR text-recognition dataset (used by notebooks/07_preprocessing_pipeline.ipynb).

Per photo:  preprocess (pipeline.preprocess_photo) -> line crops, prepared exactly as the app feeds the reader
            -> the app's reader reads each crop (only to PAIR crops with labels, labels.match_lines)
            -> rows: one per crop (accepted / review / unlabelled)
Then:       split BY BOARD -> train-only augmentation -> label files in PaddleOCR SimpleDataSet format
            -> checks (leaks, CTC length, counts).
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import labels as L
from . import pipeline as P
from . import text_lines as tl

# ---------------------------------------------------------------- split (README rule: split by board)
BOARD_SPLIT = {"animal": "train", "quiz": "train", "lecture": "train",   # B1, B4, B5
               "television": "val",                                    # B2 (same as finetune_reader.py)
               "cnn": "test",                                          # B3 (the unseen board of every test so far)
               "oct17": "train"}
# finetune_reader.py splits the 17 Oct board BY PHOTO instead (these photos are its val / test). Kept here as an
# option (OCT17_BY_PHOTO=True) so results can be compared with the teammates' runs; the default follows the
# dataset README ("every photo of one board shows the same handwriting").
OCT17_VAL_PHOTOS = {"IMG_4736", "IMG_4705", "IMG_4673"}
OCT17_TEST_PHOTOS = {"IMG_4748", "IMG_4750", "IMG_4697"}

# Validation (decided 23 Oct): the television board alone gave 10 crops of 2 texts, too few to choose an epoch.
# So val = television + a fixed ~20 % of the LECTURE board's distinct texts: every crop of a held-out text goes to val,
# so val texts are never seen in training (same writer as train; the cnn test board stays fully unseen).
VAL_TEXT_BOARD = "lecture"
VAL_TEXT_FRACTION = 0.2

CROP_DIR = "crops/all"   # every crop in one folder (23 Oct: the team splits after merging all datasets)

CTC_SLOTS = 40       # PP-OCR rec heads see 40 time steps in training: a label needs len + repeated pairs <= 40
CTC_MARGIN = 2       # ... keep 2 spare (= the "limit 38" of option 1)


def split_of(row, oct17_by_photo=False):
    if row["board"] == "oct17" and oct17_by_photo:
        stem = Path(row["original_file"]).stem
        return "val" if stem in OCT17_VAL_PHOTOS else "test" if stem in OCT17_TEST_PHOTOS else "train"
    return BOARD_SPLIT[row["board"]]


def ctc_steps(label: str) -> int:
    # CTC needs a blank between two equal neighbouring characters
    return len(label) + sum(a == b for a, b in zip(label, label[1:]))


# ---------------------------------------------------------------- one photo
def process_photo(path, gt_lines, detector, reader, steps, out_dir, row_info, dataset_ratio=None):
    """Preprocess one photo, cut + prepare its line crops, pair them with labels, save them.
    Returns (rows, log). Crops go to out_dir/crops/all/<photo>_<nn>.png: one folder, no split (the split is made
    later, after merging with the other datasets; status and grouping columns are in the rows)."""
    path = Path(path)
    img = cv2.imread(str(path))
    res = P.preprocess_photo(img, detector, steps, f35=P.geo.focal35_from_exif(path), reader=reader)
    crops = P.line_crops(res, prepare=True)
    readings = reader.read([c["crop"] for c in crops]) if crops else []
    lines = [dict(reading=t, width=c["width"], height=c["height"], rows=tl.ink_rows(c["crop"]))
             for (t, _), c in zip(readings, crops)]
    matches = L.match_lines(lines, gt_lines, dataset_ratio) if gt_lines else \
        [dict(gt_index=None, line_id="", label="", cer=None, order_support=False, ratio=None, shape_ok=None,
              status="unlabelled", ends="", label_span=False) for _ in lines]
    rows = []
    for k, (c, (text, score), m) in enumerate(zip(crops, readings, matches)):
        status = m["status"]
        rel = Path(CROP_DIR) / f"{path.stem}_{k:03d}.png"
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / rel), c["crop"])
        rows.append(dict(crop=str(rel), photo=path.name, **row_info, line_no=k, status=status,
                         line_id=m["line_id"], label=m["label"], reading=text, reading_conf=round(score, 4),
                         cer=m["cer"], order_support=m["order_support"], shape_ok=m["shape_ok"], ratio=m["ratio"],
                         ends=m.get("ends", ""), label_span=m.get("label_span", False), text_rows=m.get("rows", 1),
                         box_h=round(c["height"], 1), box_w=round(c["width"], 1),
                         enlarged=bool(c["height"] < tl.REC_MIN_H), crop_h=c["crop"].shape[0], crop_w=c["crop"].shape[1],
                         ctc_steps=ctc_steps(m["label"]) if m["label"] else None,
                         pixel_md5=hashlib.md5(c["crop"].tobytes()).hexdigest()))
    log = dict(photo=path.name, **res["log"], gt_lines=len(gt_lines), crops=len(crops),
               accepted=sum(r["status"] == "accepted" for r in rows), review=sum(r["status"] == "review" for r in rows))
    return rows, log


def text_holdout(label, fraction=None):
    """Fixed, label-only choice of the held-out texts: the md5 of the normalised text, as a number in [0, 1).
    fraction None = VAL_TEXT_FRACTION as it is NOW (a default argument would freeze the value at definition)."""
    fraction = VAL_TEXT_FRACTION if fraction is None else fraction
    h = int(hashlib.md5(L.norm(label).encode()).hexdigest()[:8], 16) / 16 ** 8
    return h < fraction


def apply_val_texts(rows, board=None, fraction=None):
    """Move every labelled crop of `board` whose text is held out to the val split. Returns (#crops, #texts).
    None = VAL_TEXT_BOARD / VAL_TEXT_FRACTION as they are at call time."""
    board = VAL_TEXT_BOARD if board is None else board
    fraction = VAL_TEXT_FRACTION if fraction is None else fraction
    moved, texts = 0, set()
    for r in rows:
        if r["board"] == board and r["label"] and r["status"] in ("accepted", "review") and text_holdout(r["label"], fraction):
            r["split"] = "val"
            moved += 1
            texts.add(L.norm(r["label"]))
    return moved, len(texts)


def rematch(rows, gt, out_dir=None):
    """Pair the cached crops with their labels again with the CURRENT rules of labels.match_lines, without
    detecting or reading again (the readings and box sizes are in the rows). Use after a change to the pairing
    rules. Returns {(old status, new status): count}."""
    by_photo = defaultdict(list)
    for r in rows:
        if r["status"] != "augmented":
            by_photo[r["photo"]].append(r)
    changes = Counter()
    for photo, rs in by_photo.items():
        rs.sort(key=lambda r: int(r["line_no"]))
        g = gt.get(photo, [])
        lines = []
        for r in rs:
            if out_dir is not None and "text_rows" not in r and g:      # count text rows once, from the saved crop
                im = cv2.imread(str(Path(out_dir) / r["crop"]))
                r["text_rows"] = tl.ink_rows(im) if im is not None else 1
            lines.append(dict(reading=r["reading"] or "", width=float(r["box_w"]), height=float(r["box_h"]),
                              rows=int(r.get("text_rows", 1) or 1)))
        ms = L.match_lines(lines, g) if g else [None] * len(rs)
        for r, m in zip(rs, ms):
            old = r["status"]
            if m is None:
                m = dict(line_id="", label="", cer=None, order_support=False, ratio=None, shape_ok=None,
                         status="unlabelled", ends="", label_span=False)
            r.update(status=m["status"], line_id=m["line_id"], label=m["label"], cer=m["cer"],
                     order_support=m["order_support"], shape_ok=m["shape_ok"], ratio=m["ratio"], ends=m["ends"],
                     label_span=m["label_span"],
                     ctc_steps=ctc_steps(m["label"]) if m["label"] else None)
            changes[(old, r["status"])] += 1
    return changes


def recheck_shape(rows):
    """Second pass of the shape check for photos with fewer than labels.MIN_RATIO_LINES confident pairs: they
    were checked against nothing (shape_ok True); check them against the median of the whole dataset."""
    conf = [r["ratio"] for r in rows if r["cer"] is not None and r["cer"] <= L.ACCEPT_CER and r["ratio"]]
    if not conf:
        return 0
    ref = float(np.median(conf))
    by_photo = defaultdict(list)
    for r in rows:
        by_photo[r["photo"]].append(r)
    changed = 0
    for photo, rs in by_photo.items():
        if sum(r["cer"] is not None and r["cer"] <= L.ACCEPT_CER for r in rs) >= L.MIN_RATIO_LINES:
            continue
        for r in rs:
            if r["status"] not in ("accepted", "review"):
                continue
            ok = 1 / L.RATIO_TOL <= r["ratio"] / ref <= L.RATIO_TOL
            new = L.accept_status(dict(ends=r.get("ends", ""), cer=r["cer"], order_support=r["order_support"],
                                       shape_ok=ok))
            changed += new != r["status"]
            r["shape_ok"], r["status"] = ok, new
    return changed


# ---------------------------------------------------------------- train-only augmentation
def shrink_copy(crop, target_h, rng):
    """Small-text copy of a large crop: shrink it to target_h px (INTER_AREA = what a farther camera sees),
    slight blur half of the time, then the APP's enlargement back up (prepare_for_reader).
    So the reader learns from the exact kind of image the app gives it for small writing."""
    s = target_h / crop.shape[0]
    small = cv2.resize(crop, (max(4, int(round(crop.shape[1] * s))), target_h), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        small = cv2.GaussianBlur(small, (0, 0), rng.uniform(0.3, 0.7))
    return tl.prepare_for_reader(small, target_h)


def augment_train(rows, out_dir, copies=1, min_h=12, max_h=30, min_source_h=48, seed=20261022):
    """For every accepted TRAIN crop whose line is at least min_source_h px tall: `copies` shrunken copies with a
    random text height in [min_h, max_h] px. Returns the new rows (status 'augmented', same label)."""
    rng = random.Random(seed)
    new = []
    for r in rows:
        if r["split"] != "train" or r["status"] != "accepted" or r["box_h"] < min_source_h:
            continue
        crop = cv2.imread(str(out_dir / r["crop"]))
        for k in range(copies):
            th = rng.randint(min_h, max_h)
            img = shrink_copy(crop, th, rng)
            rel = Path("crops") / "train_augmented" / f"{Path(r['crop']).stem}_s{th}_{k}.png"
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / rel), img)
            new.append(dict(r, crop=str(rel), status="augmented", source_crop=r["crop"], aug_text_h=th,
                            crop_h=img.shape[0], crop_w=img.shape[1], enlarged=True,
                            pixel_md5=hashlib.md5(img.tobytes()).hexdigest()))
    return new


# ---------------------------------------------------------------- label files + checks
def write_label_files(rows, out_dir, include_review=False):
    """PaddleOCR SimpleDataSet files '<crop path relative to out_dir>\\t<label>':
        train.txt (accepted + augmented [+ review]), val.txt, test.txt (accepted [+ review]),
        and *_ctc38.txt: the same without labels longer than CTC_SLOTS - CTC_MARGIN steps.
    Returns {file name: number of lines}."""
    use = {"accepted", "augmented"} | ({"review"} if include_review else set())
    files = defaultdict(list)
    for r in rows:
        if r["status"] not in use or not r["label"]:
            continue
        if r["status"] == "augmented" and r["split"] != "train":
            continue
        line = f"{r['crop']}\t{r['label']}"
        files[f"{r['split']}.txt"].append(line)
        if ctc_steps(r["label"]) <= CTC_SLOTS - CTC_MARGIN:
            files[f"{r['split']}_ctc38.txt"].append(line)
    for name, lines in files.items():
        (out_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {k: len(v) for k, v in sorted(files.items())}


def migrate_crops(rows, out_dir):
    """Move crops saved by the first builds (crops/train/, crops/test/) into crops/all/ and update the rows.
    Returns the number of rows whose path changed."""
    out_dir, moved = Path(out_dir), 0
    for r in rows:
        if r["status"] == "augmented" or r["crop"].startswith(CROP_DIR + "/"):
            continue
        new = f"{CROP_DIR}/{Path(r['crop']).name}"
        src, dst = out_dir / r["crop"], out_dir / new
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists() and not dst.exists():
            src.rename(dst)
        if dst.exists():
            r["crop"] = new
            moved += 1
    return moved


def suggest_split(rows, manifest, oct17_by_photo=False, board=None, fraction=None):
    """Information only (no files): the split our own rules would give each crop -- by board, with the
    held-out lecture texts as validation. Written to the manifest as `suggested_split` for whoever splits the
    merged dataset."""
    for r in rows:
        r["split"] = split_of(manifest[r["photo"]], oct17_by_photo)
    apply_val_texts(rows, board, fraction)
    for r in rows:
        r["suggested_split"] = r.pop("split")


def write_label_file_all(rows, out_dir):
    """One folder, no split. PaddleOCR SimpleDataSet format '<crop path relative to out_dir>\t<label>':
        labels.txt        every accepted crop
        labels_ctc38.txt  the same without labels longer than CTC_SLOTS - CTC_MARGIN steps (for a training config
                          whose recognition head has CTC_SLOTS time steps)
        review.txt        crops paired with a label but not trusted: for a person to check; not for training as is
    Returns {file name: number of lines}."""
    files = {"labels.txt": [], "labels_ctc38.txt": [], "review.txt": []}   # all three always (re)written
    for r in rows:
        if not r["label"] or r["status"] not in ("accepted", "review"):
            continue
        line = f"{r['crop']}\t{r['label']}"
        if r["status"] == "accepted":
            files["labels.txt"].append(line)
            if ctc_steps(r["label"]) <= CTC_SLOTS - CTC_MARGIN:
                files["labels_ctc38.txt"].append(line)
        else:
            files["review.txt"].append(line)
    for name, lines in files.items():      # an empty list still overwrites a file left by an earlier run
        (Path(out_dir) / name).write_text("".join(l + "\n" for l in lines), encoding="utf-8")
    return {k: len(v) for k, v in sorted(files.items())}


def check_all(rows, out_dir):
    """Checks for the one-folder dataset. Returns [(check, ok, detail)]."""
    out = []
    missing = sum(not (Path(out_dir) / r["crop"]).exists() for r in rows)
    out.append(("every crop file exists", missing == 0, f"{missing} missing"))
    folders = {str(Path(r["crop"]).parent) for r in rows}
    out.append(("all crops in one folder", folders <= {CROP_DIR}, f"folders: {sorted(folders)}"))
    empty = sum(r["status"] in ("accepted", "review") and not r["label"] for r in rows)
    out.append(("every labelled crop has a label", empty == 0, f"{empty} empty labels"))
    md5 = defaultdict(set)
    for r in rows:
        if r["status"] == "accepted":
            md5[r["pixel_md5"]].add(r["label"])
    clash = sum(len(v) > 1 for v in md5.values())
    out.append(("no identical crop with two different labels", clash == 0, f"{clash} clashes"))
    keys = ("board", "photo", "repeat_shot_group", "label", "suggested_split")
    out.append(("grouping columns present for a later split", all(k in rows[0] for k in keys), ", ".join(keys)))
    return out


def write_manifest(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def check_dataset(rows):
    """Leak and sanity checks. Returns a list of (check, ok, detail); raises nothing, the notebook asserts."""
    out = []
    boards = defaultdict(set)
    for r in rows:
        boards[r["board"]].add(r["split"])
    multi = {b: s for b, s in boards.items() if len(s) > 1}
    allowed = {"oct17", VAL_TEXT_BOARD}          # oct17 only with OCT17_BY_PHOTO; lecture split by TEXT
    out.append(("every board in one split (except the text-held-out board)", set(multi) <= allowed,
                f"boards in more than one split: {multi}" if multi else "ok"))
    texts = defaultdict(set)
    for r in rows:
        if r["label"] and r["status"] in ("accepted", "review", "augmented"):
            texts[L.norm(r["label"])].add(r["split"])
    tv = sum({"train", "val"} <= s for s in texts.values())
    tt = sum({"train", "test"} <= s for s in texts.values())
    out.append(("no validation text in training", tv == 0, f"{tv} texts in both train and val"))
    out.append(("test texts in training (reported)", True, f"{tt} texts in both train and test"))
    md5 = defaultdict(set)
    for r in rows:
        md5[r["pixel_md5"]].add(r["split"])
    cross = sum(len(s) > 1 for s in md5.values())
    out.append(("no identical crop in two splits", cross == 0, f"{cross} identical crops across splits"))
    aug_out = sum(r["status"] == "augmented" and r["split"] != "train" for r in rows)
    out.append(("augmentation only in train", aug_out == 0, f"{aug_out} augmented crops outside train"))
    empty = sum(r["status"] in ("accepted", "review") and not r["label"] for r in rows)
    out.append(("every labelled crop has a label", empty == 0, f"{empty} empty labels"))
    return out


def summary_table(rows):
    c = Counter((r["split"], r["status"]) for r in rows)
    splits = sorted({s for s, _ in c})
    statuses = ["accepted", "review", "augmented", "unlabelled"]
    lines = ["split      " + "".join(f"{s:>12}" for s in statuses)]
    for sp in splits:
        lines.append(f"{sp:<11}" + "".join(f"{c[(sp, s)]:>12}" for s in statuses))
    return "\n".join(lines)
