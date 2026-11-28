#!/usr/bin/env python3
"""Board2Notes reader versions on RunPod: every run under the same budget, one comparison report.

  V1    historical baseline (Baseline_model, Oct 13): evaluated only, read the way PaddleOCR deploys it
  pre   PP-OCRv5_server_rec pretrained, no fine-tuning
  V1b   IAM handwriting lines only, our adapted recipe (64 px, width/8 positions)
  2x2 data experiment (adapted recipe, same budget, same validation and tests; every run, A and C included, also gets
  PaddleOCR's built-in random augmentation RecAug in memory, so this tests OUR pre-made augmented copies on top of it):
                              without our augmented copies   with our augmented copies
     short lines only (<=40)  A = V2short                    B = V3short
     short + long lines       C = V2                         D = V3
  C1    V3's data with PaddleOCR's ORIGINAL recipe (48x320 input, 40 positions): original vs adapted RECIPE.
        It changes height, width, pooling and which lines fit at once, so it is not a pooling test.
  P40   pooling-only pair: V3's data restricted to lines that fit 40 positions (identical list for both),
  P128  same 64x1024 input, starting weights, budget and validation. P40 trains with the original pooling
        (40 positions), P128 with ours (width/8). The ONLY difference is the training pooling.
        V3 cannot be one side: it trains on lines that need more than 40 positions, which P40 cannot use.
  (V4 = V3 reading photos after the preprocessing pipeline: scored later on the Mac with the exported V3.)

  Training-strategy batch (experiment strategies_s0s3, 7 Nov 2026): every run starts from the PRETRAINED weights.
  S0    clean baseline: V2's data WITHOUT the 64 teammates' crops (their writers are unknown and may be the
        new-writer test writers); our adapted recipe (64x1024, width/8 positions), no augmented copies
  S1    S0 + stage-wise learning rates in the backbone (lr_mult_list [0.1, 0.1, 0.25, 0.5, 1.0]; neck/heads 1.0)
  S2    S0 + focal CTC (each line's CTC loss L becomes L * (1 - exp(-L))^2; NRTR branch unchanged)
  S3    S0 + both.   Each with seed 1 and seed 2 (S0s2 ... S3s2): the same lines, so identical update counts.
        Seed 2 changes augmentation/dropout randomness (Global.seed) AND the batch order (its list is shuffled,
        because PaddleOCR's sampler orders batches by epoch number only).
  Guidance ablation (experiment guide_g0, 7 Nov 2026 evening):
  G0    S0 with use_guide OFF in the CTC branch's SVTR neck, so the CTC loss (the reader used at inference) also
        trains the backbone. Everything else is S0: same list, seed, budget, loss weights (1.0 / 1.0).
  G0s2  the same with seed 2 (S0s2's list).   Compared with the EXISTING S0/S0s2 (strategies_s0s3) on the Mac
        with merge_guide_report.py, after checking that scoring, data and environment are identical.
  Primary criterion, fixed BEFORE training: whiteboard-validation mean line CER, averaged over the two seeds.
  A setting "helps" only if that average is better with a 95% interval below 0 AND both seeds point the same way.
  IAM val is reported as the general-handwriting check; tests (CNN, held-out long, new writers) are for reporting.

Evaluation sets: whiteboard val (checkpoint choice only), CNN test board, held-out long sentences, IAM val.
The held-out long sentences are ~20% of the DISTINCT long sentences; every photo and augmented copy of them
is kept out of training. They test NEW SENTENCES IN THE SAME HANDWRITING ON THE SAME BOARD, not a new writer.

PaddleOCR changes (checked on the real backbone): 1 training pooling [1, 40] -> [1, width/2] (= width/8
positions), switchable back with B2N_POOL40=1; 2 reading pooling [3, 2] -> [feature height, 2] (identical at
48 px; at 64 px the original averaged only 3 of 4 feature rows); 3 export height taken from the training height.
The repeated-letter ("stutter") explanation for a 40-vs-width/8 mismatch is a HYPOTHESIS: P40 vs P128 tests it
with the repeated-letter metric in the report.

Usage (through run_runpod.sh):  python b2n_versions.py all --experiment strategies_s0s3 [--smoke-only] [--runs ...]
                                python b2n_versions.py check   (data checks only; no Paddle or GPU needed)
Testing only (CPU dry run on a small sample, never on RunPod): B2N_TEST=1 B2N_CPU=1 B2N_STEPS=.. B2N_BATCH=..
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, re, shutil, subprocess, sys, tempfile, time, zipfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
STORAGE = Path(os.environ.get("B2N_WORK", "/workspace/b2n"))
REPO = Path(os.environ.get("B2N_REPO", STORAGE / "PaddleOCR_versions"))   # separate from the old A/B checkout
PRETRAINED = Path(os.environ.get("B2N_PRETRAINED", REPO / "PP-OCRv5_server_rec_pretrained"))
BASE_CFG = REPO / "configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml"
DICT = REPO / "ppocr/utils/dict/ppocrv5_dict.txt"
DATA_ZIP = Path(os.environ.get("B2N_DATA_ZIP", HERE / "b2n_versions_data.zip"))
NW_ZIP = Path(os.environ.get("B2N_NEWWRITER_ZIP", HERE / "b2n_newwriter_data.zip"))   # new-writer test (7 Nov)
TEST_MODE = os.environ.get("B2N_TEST") == "1"
CPU = os.environ.get("B2N_CPU") == "1"
EXPERIMENT = WORK = DATA = RUNS = RESULTS = ROOT = None

# ---- settings shared by every run (changing any of them needs a NEW experiment name) ----
STEPS = int(os.environ.get("B2N_STEPS", 6000))        # training updates per run
BATCH = int(os.environ.get("B2N_BATCH", 16))
EVAL_STEP = int(os.environ.get("B2N_EVAL_STEP", 400)) # whiteboard validation every N updates; best kept
LR, WARMUP_STEPS, SEED = 1e-4, 300, 1
MAX_TEXT = 100                    # label limit for EVERY run, so validation never substitutes examples
NUM_WORKERS = int(os.environ.get("B2N_WORKERS", "2" if CPU else "8"))
HOLDOUT_FRACTION, HOLDOUT_SEED = 0.2, 20261105
EXPORT_SAME_MIN, EXPORT_CER_TOL = 0.95, 0.01   # exported model must read the fixture like its checkpoint
BOOTSTRAP = 2000
GEOMS = {"h64": dict(H=64, W=1024), "official": dict(H=48, W=320), "h48": dict(H=48, W=1024)}
TRAIN_FILES_V3 = ["wb_train", "long_train", "teammates", "aug_of_train", "aug_of_long_train"]
RUN_SPEC = {
    "V1b": dict(geom="h64", pool40=False, positions=128, files=["iam_train"], what="IAM handwriting lines only, adapted recipe"),
    "V2short": dict(geom="h64", pool40=False, positions=128, max_chars=40, files=["wb_train", "long_train", "teammates"],
                    what="A: our crops, short lines only (40 chars or fewer), no augmented copies"),
    "V3short": dict(geom="h64", pool40=False, positions=128, max_chars=40, files=TRAIN_FILES_V3,
                    what="B: our crops, short lines only + our augmented copies"),
    "V2": dict(geom="h64", pool40=False, positions=128, files=["wb_train", "long_train", "teammates"],
               what="C: our crops, short + long lines, no augmented copies (replaces IAM)"),
    "V3": dict(geom="h64", pool40=False, positions=128, files=TRAIN_FILES_V3, what="D: our crops, short + long lines + our augmented copies"),
    "C1": dict(geom="official", pool40=True, positions=40, files=TRAIN_FILES_V3,
               what="V3 data, PaddleOCR's original recipe (48x320, 40 positions)"),
    "P40": dict(geom="h64", pool40=True, positions=40, files=TRAIN_FILES_V3,
                what="pooling-only pair: original training pooling (40 positions), lines that fit 40"),
    "P128": dict(geom="h64", pool40=False, positions=40, files=TRAIN_FILES_V3,
                 what="pooling-only pair: our training pooling (width/8 positions), SAME lines as P40"),
}
LR_MULT = [0.1, 0.1, 0.25, 0.5, 1.0]          # PPHGNetV2_B4: stem, stage 1-4 (SVTR neck and heads stay at 1.0)
for _seed, _sfx in ((1, ""), (2, "s2")):
    for _n, _lr, _focal, _what in (("S0", False, False, "clean baseline: V2 data minus the 64 teammates' crops"),
                                   ("S1", True, False, "S0 + stage-wise backbone learning rates"),
                                   ("S2", False, True, "S0 + focal CTC loss"),
                                   ("S3", True, True, "S0 + stage-wise learning rates + focal CTC")):
        RUN_SPEC[_n + _sfx] = dict(geom="h64", pool40=False, positions=128, files=["wb_train", "long_train"],
                                   lr_mult=LR_MULT if _lr else None, focal=_focal, seed=_seed,
                                   what=f"{_what} (seed {_seed})")
for _seed, _sfx in ((1, ""), (2, "s2")):
    RUN_SPEC["G0" + _sfx] = dict(geom="h64", pool40=False, positions=128, files=["wb_train", "long_train"],
                                 guide=False, seed=_seed,
                                 what=f"S0 with guidance off: CTC loss also trains the backbone (seed {_seed})")
S_RUNS = ["S0", "S1", "S2", "S3", "S0s2", "S1s2", "S2s2", "S3s2"]
G_RUNS = ["G0", "G0s2"]
G_COMPARISONS = [("G0avg", "S0avg", "guidance off vs guided baseline (mean of 2 seeds)"),
                 ("G0", "S0", "guidance off, seed 1"), ("G0s2", "S0s2", "guidance off, seed 2"),
                 ("S0s2", "S0", "seed noise: guided recipe, different seed"),
                 ("G0s2", "G0", "seed noise: guidance-off recipe, different seed"),
                 ("G0avg", "pretrained", "guidance off vs no fine-tune")]
S_COMPARISONS = [("S1avg", "S0avg", "stage-wise LR vs baseline (mean of 2 seeds)"),
                 ("S2avg", "S0avg", "focal CTC vs baseline (mean of 2 seeds)"),
                 ("S3avg", "S0avg", "both vs baseline (mean of 2 seeds)"),
                 ("S3avg", "S1avg", "adding focal CTC to stage-wise LR (mean of 2 seeds)"),
                 ("S3avg", "S2avg", "adding stage-wise LR to focal CTC (mean of 2 seeds)"),
                 ("S1", "S0", "stage-wise LR, seed 1"), ("S1s2", "S0s2", "stage-wise LR, seed 2"),
                 ("S2", "S0", "focal CTC, seed 1"), ("S2s2", "S0s2", "focal CTC, seed 2"),
                 ("S3", "S0", "both, seed 1"), ("S3s2", "S0s2", "both, seed 2"),
                 ("S0s2", "S0", "seed noise: identical recipe, different seed"),
                 ("S0avg", "pretrained", "clean baseline vs no fine-tune")]
EVAL_SETS = {"whiteboard val": "val_wb", "CNN test": "test_cnn", "long sentences (held out)": "long_test", "IAM val": "val_iam",
             "new writers": "test_newwriter"}
COMPARISONS = [("V1b", "pretrained", "IAM-only adapted fine-tune vs no fine-tune"),
               ("V2", "V1b", "our whiteboard data vs IAM (replacement, not addition)"),
               ("V2", "V2short", "C vs A: adding long sentences, without our augmented copies"),
               ("V3", "V3short", "D vs B: adding long sentences, with our augmented copies"),
               ("V3short", "V2short", "B vs A: our augmented copies, short lines only"),
               ("V3", "V2", "D vs C: our augmented copies, short + long lines"),
               ("V3", "C1", "adapted recipe vs original recipe (several changes at once)"),
               ("P128", "P40", "training pooling ONLY (same lines, input, weights, budget)"),
               ("V3", "pretrained", "best adapted version vs no fine-tune"),
               ("V3", "V1", "best adapted version vs historical baseline")]
IDENTITY_SCHEMA = 6
STATIC_RUNTIME = "pir_no_default_passes_official_resize_v2"


# ------------------------------------------------------------------ helpers
def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(cmd, logfile=None, show=None, env=None):
    log("$", " ".join(str(c) for c in cmd)[:300])
    with open(logfile, "a") if logfile else open(os.devnull, "w") as lf:
        p = subprocess.Popen([str(c) for c in cmd], cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env={**os.environ, **(env or {})})
        tail = []
        for line in p.stdout:
            lf.write(line); lf.flush()
            tail = (tail + [line])[-60:]
            if show is None or show.search(line):
                print(line.rstrip()[:260], flush=True)
        p.wait()
    if p.returncode != 0:
        print("".join(tail), flush=True)
        raise SystemExit(f"FAILED (exit {p.returncode}): {' '.join(str(c) for c in cmd)[:200]}"
                         + (f"\nfull log: {logfile}" if logfile else ""))


def set_paths(experiment):
    global EXPERIMENT, WORK, DATA, RUNS, RESULTS, ROOT
    EXPERIMENT = experiment
    WORK = STORAGE / "experiments" / experiment
    DATA, RUNS, RESULTS = WORK / "data", WORK / "runs", WORK / "results"
    ROOT = DATA / "b2n_data"


def norm(s):
    return " ".join(str(s).lower().split())


def ctc_need(t):
    """CTC positions a label needs: one per character plus a blank between two equal neighbours ('ll')."""
    return len(t) + sum(a == b for a, b in zip(t, t[1:]))


def occupied(h, w, H, W):
    """Positions lying over the text after resizing to height H (diagnostic only: CTC may use the padding too)."""
    return min(W, math.ceil(H * w / h)) // 8


def repeats(s):
    s = norm(s)
    return sum(a == b and a != " " for a, b in zip(s, s[1:]))


def insertion_diagnostics(reference, prediction):
    """Deterministic edit alignment and a bounded repeated-substring diagnostic.

    Count inserted characters without cancelling them against deleted legitimate
    doubles. Flag insertions inside tandem repeats of 1-16 characters, so errors
    such as 'biaseses' are visible as well as 'boook'. Alignment can be ambiguous;
    these flags describe the output and do not establish the cause of an error.
    """
    a, b = norm(reference), norm(prediction)
    dp = [list(range(len(b) + 1))]
    for i, ca in enumerate(a, 1):
        row = [i]
        for j, cb in enumerate(b, 1):
            row.append(min(dp[i - 1][j] + 1, row[j - 1] + 1,
                           dp[i - 1][j - 1] + (ca != cb)))
        dp.append(row)
    i, j, inserted = len(a), len(b), set()
    while i or j:
        # Prefer a match/substitution on ties: avoid manufacturing insertions.
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            i -= 1; j -= 1
        elif j and dp[i][j] == dp[i][j - 1] + 1:
            inserted.add(j - 1); j -= 1
        else:
            i -= 1
    repeated_positions = set()
    if inserted:
        for width in range(1, min(16, len(b) // 2) + 1):
            for start in range(len(b) - 2 * width + 1):
                unit = b[start:start + width]
                if unit.strip() and unit == b[start + width:start + 2 * width]:
                    repeated_positions.update(range(start, start + 2 * width))
    return dict(edit_distance=dp[-1][-1], inserted_chars=len(inserted),
                repeated_inserted_chars=sum(k in repeated_positions and b[k] != " " for k in inserted))


def bucket(t):
    n = len(t)
    return "1-20 chars" if n <= 20 else ("21-40 chars" if n <= 40 else "over 40 chars")


# ------------------------------------------------------------------ 1. patches (separate checkout only)
def patch():
    edits = [("ppocr/modeling/backbones/rec_pphgnetv2.py", [
        ("import math\n", "import math\nimport os  # board2notes\n"),
        ("x = F.adaptive_avg_pool2d(x, [1, 40])",
         "x = F.adaptive_avg_pool2d(x, [1, 40] if os.environ.get('B2N_POOL40') == '1' else [1, x.shape[3] // 2])"
         "  # board2notes patch 1: width/8 training positions (B2N_POOL40=1 keeps the original 40)"),
        ("x = F.avg_pool2d(x, [3, 2])",
         "x = F.avg_pool2d(x, [x.shape[2], 2])  # board2notes patch 2: average the full feature height"
         " (identical to [3, 2] at 48 px)")]),
        ("ppocr/utils/export_model.py", [
        ('paddle.static.InputSpec(shape=[None, 3, 48, -1], dtype="float32")',
         'paddle.static.InputSpec(shape=[None, 3, int(__import__("os").environ.get("B2N_EXPORT_H", "48")), -1],'
         ' dtype="float32")  # board2notes patch 3: export at the training height')])]
    for rel, pairs in edits:
        f = REPO / rel
        s = f.read_text(encoding="utf-8")
        for old, new in pairs:
            if new in s:
                continue
            if s.count(old) != 1:
                raise SystemExit(f"patch: expected exactly one '{old[:50]}' in {rel}; repository version differs")
            s = s.replace(old, new)
        f.write_text(s, encoding="utf-8")
    log("PaddleOCR patches in place: training pooling, reading pooling, export height")


# ------------------------------------------------------------------ 2. data
def read_labels(path, prefix=""):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if "\t" in line:
            p, t = line.split("\t", 1)
            out.append((f"{prefix}{p}", t))
    return out


def write_set(name, lines):
    (DATA / f"{name}.txt").write_text("".join(f"{p}\t{t}\n" for p, t in lines), encoding="utf-8")


def load_set(name):
    return read_labels(DATA / f"{name}.txt")


def image_hw(path):
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise SystemExit(f"unreadable image {path}")
    return im.shape[0], im.shape[1]


def run_train_list(name):
    """The exact training lines of one run: only HARD limits (label length, positions the run has)."""
    spec = RUN_SPEC[name]
    info = read_json(DATA / "line_info.json")
    keep, why = [], Counter()
    for f in spec["files"]:
        for p, t in load_set(f):
            if spec.get("max_chars") and len(t) > spec["max_chars"]:   # short-only runs (copies share their original's text)
                why[f"longer than {spec['max_chars']} characters (short-only run)"] += 1
            elif len(t) > MAX_TEXT - 2:          # the NRTR encoder drops len >= MAX_TEXT - 1
                why["label longer than the limit"] += 1
            elif ctc_need(t) > spec["positions"]:
                why[f"needs more than {spec['positions']} positions"] += 1
            else:
                keep.append((p, t))
                G = GEOMS[spec["geom"]]
                if ctc_need(t) > occupied(info[p]["h"], info[p]["w"], G["H"], G["W"]):
                    why["kept, but text covers fewer positions than it needs (diagnostic)"] += 1
    return keep, why


def prepare(check_only=False):
    done = DATA / "prepared.json"
    if done.exists() and not check_only:
        saved = read_json(done)
        for name, digest in saved["label_sha256"].items():
            if sha(DATA / f"{name}.txt") != digest:
                raise SystemExit(f"prepared label file changed: {name}. Use a NEW experiment.")
        log("prepared data verified:", saved["counts"])
        return saved
    if not DATA_ZIP.is_file():
        raise SystemExit(f"missing {DATA_ZIP.name} next to this script")
    DATA.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "reader/manifest.csv").exists():
        log("extracting", DATA_ZIP.name, "...")
        with zipfile.ZipFile(DATA_ZIP) as z:
            z.extractall(DATA)
    rows = list(csv.DictReader(open(ROOT / "reader/manifest.csv", encoding="utf-8")))
    by_id = {r["id"]: r for r in rows}
    info, sets = {}, defaultdict(list)
    dimension_updates = []
    pick = {("whiteboard", "train"): "wb_train", ("teammates", "train"): "teammates",
            ("long_sentence", "long_unassigned"): "long_all", ("augmented", "aug_of_train"): "aug_of_train",
            ("augmented", "aug_of_long"): "aug_of_long_all", ("whiteboard", "val"): "val_wb",
            ("whiteboard", "test_cnn"): "test_cnn"}
    for r in rows:
        key = pick.get((r["source"], r["split"]))
        if key is None:
            raise SystemExit(f"unexpected source/split in manifest: {r['source']}/{r['split']}")
        p = "reader/" + r["path"]
        h, w = image_hw(ROOT / p)
        if (h, w) != (int(r["height"]), int(r["width"])):
            dimension_updates.append(dict(file=p, recorded=[int(r["height"]), int(r["width"])], actual=[h, w]))
        sets[key].append((p, r["label"]))
        info[p] = dict(h=h, w=w, id=r["id"], parent=r["parent"],
                       cluster=("sentence:" + norm(r["label"])) if r["source"] == "long_sentence" else (r["photo"] or r["id"]))
    # ---- new-writer test (teammates' handwriting, 2 writers, labels reviewed 7 Nov); never used for training ----
    if not NW_ZIP.is_file():
        raise SystemExit(f"missing {NW_ZIP.name} next to this script (the new-writer test set)")
    if not (ROOT / "newwriter/manifest.csv").exists():
        with zipfile.ZipFile(NW_ZIP) as z:
            z.extractall(DATA)
    for r in csv.DictReader(open(ROOT / "newwriter/manifest.csv", encoding="utf-8")):
        p = "newwriter/" + r["path"]
        h, w = image_hw(ROOT / p)
        sets["test_newwriter"].append((p, r["label"]))
        info[p] = dict(h=h, w=w, id=p, parent="", cluster=r["writer"] + ":" + r["photo"])
    for split, key in (("train", "iam_train"), ("val", "val_iam")):
        for p, t in read_labels(ROOT / f"iam/{split}.txt", "iam/"):
            sets[key].append((p, t))
            h, w = image_hw(ROOT / p)
            info[p] = dict(h=h, w=w, id=p, parent="", cluster=p)

    # ---- long-sentence holdout: ~20% of DISTINCT sentences; every photo and augmented copy stays out ----
    sentences = sorted({norm(t) for _, t in sets["long_all"]})
    k = max(1, round(HOLDOUT_FRACTION * len(sentences)))
    held = set(random.Random(HOLDOUT_SEED).sample(sentences, k))
    held_ids = {info[p]["id"] for p, t in sets["long_all"] if norm(t) in held}
    sets["long_test"] = [(p, t) for p, t in sets["long_all"] if norm(t) in held]
    sets["long_train"] = [(p, t) for p, t in sets["long_all"] if norm(t) not in held]
    aug_out = [(p, t) for p, t in sets["aug_of_long_all"] if info[p]["parent"] in held_ids or norm(t) in held]
    sets["aug_of_long_train"] = [x for x in sets["aug_of_long_all"] if x not in aug_out]
    for p, _ in sets["aug_of_long_train"]:
        parent = by_id.get(info[p]["parent"])
        assert parent is None or norm(parent["label"]) not in held, "an augmented copy of a held-out sentence is in training"

    missing = [p for s in sets.values() for p, _ in s if not (ROOT / p).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} images listed but missing, e.g. {missing[:3]}")
    n = {k_: len(v) for k_, v in sets.items()}
    if not TEST_MODE:
        expect = dict(wb_train=1100, val_wb=130, test_cnn=170, iam_train=6000, val_iam=900, long_train=180, long_test=30,
                      test_newwriter=150)
        for k_, lo in expect.items():
            assert n.get(k_, 0) >= lo, f"{k_} has {n.get(k_, 0)} lines (expected >= {lo}): wrong data zip?"

    unknown = Counter()
    if DICT.is_file():
        chars = set(DICT.read_text(encoding="utf-8").splitlines()) | {" "}
        unknown = Counter(c for s in sets.values() for _, t in s for c in t if c not in chars)
        if unknown:
            raise SystemExit(f"characters not in ppocrv5_dict: {dict(unknown)}")
    else:
        log("(dictionary check skipped: PaddleOCR not cloned yet)")

    # ---- leaks: no evaluation text or identical image anywhere in training ----
    train_keys = ["wb_train", "long_train", "teammates", "aug_of_train", "aug_of_long_train", "iam_train"]
    eval_keys = ["val_wb", "test_cnn", "long_test", "test_newwriter"]
    tt = {k_: {norm(t) for _, t in sets[k_]} for k_ in train_keys + eval_keys}
    train_text = set().union(*(tt[k_] for k_ in train_keys))
    for e in eval_keys:
        assert not (tt[e] & train_text), f"a {e} text also appears in training"
    assert not (tt["test_cnn"] & tt["val_wb"]) and not (tt["long_test"] & (tt["val_wb"] | tt["test_cnn"])), "eval sets overlap"
    md5 = lambda p: hashlib.md5((ROOT / p).read_bytes()).hexdigest()
    hh = {k_: {md5(p) for p, _ in sets[k_]} for k_ in train_keys + eval_keys}
    train_img = set().union(*(hh[k_] for k_ in train_keys))
    for e in eval_keys:
        assert not (hh[e] & train_img), f"an identical image is in training and {e}"

    # ---- validation identity: every validation label passes BOTH label encoders in every run ----
    too_long_val = [t for _, t in sets["val_wb"] if len(t) > MAX_TEXT - 2]
    assert not too_long_val, f"{len(too_long_val)} validation labels exceed the encoder limit: validation would substitute"

    for k_, v in sets.items():
        write_set(k_, v)
    write_json(DATA / "line_info.json", info)
    write_json(DATA / "dimension_updates.json", dimension_updates)
    rng = random.Random(0)
    fixture = sorted(rng.sample(sets["val_wb"], min(30, len(sets["val_wb"]))))
    write_set("export_fixture", fixture)                     # export equivalence: validation crops, never test

    runs = {}
    for name in RUN_SPEC:
        keep, why = run_train_list(name)
        if RUN_SPEC[name].get("seed", 1) != 1:
            # PaddleOCR's sampler orders batches by epoch number only (Global.seed changes augmentation and dropout,
            # not batch order), so a second seed also gets the SAME lines in a different file order.
            random.Random(RUN_SPEC[name]["seed"]).shuffle(keep)
        write_set(f"train_{name}", keep)
        runs[name] = dict(lines=len(keep), left_out={k_: v for k_, v in why.items() if not k_.startswith("kept")},
                          diagnostic={k_: v for k_, v in why.items() if k_.startswith("kept")})
    assert sha(DATA / "train_P40.txt") == sha(DATA / "train_P128.txt"), "P40 and P128 must train on identical lines"
    assert len({sha(DATA / f"train_{r}.txt") for r in S_RUNS if RUN_SPEC[r]["seed"] == 1}) == 1, "S0-S3 must train on identical lists"
    assert len({tuple(sorted(load_set(f"train_{r}"))) for r in S_RUNS}) == 1, "both seeds must train on the same lines"
    assert sha(DATA / "train_G0.txt") == sha(DATA / "train_S0.txt"), "G0 must train on S0's exact list"
    assert sha(DATA / "train_G0s2.txt") == sha(DATA / "train_S0s2.txt"), "G0s2 must train on S0s2's exact list (same order)"
    team = {p for p, _ in sets["teammates"]}
    team_ids = {info[p]["id"] for p in team}
    for p, _ in load_set("train_S0"):
        assert p not in team and info[p]["parent"] not in team_ids, f"a teammates' crop or copy is in S0: {p}"

    # smoke lists: a little of everything, every code path once
    write_set("smoke_train_h64", rng.sample(load_set("train_V3"), min(48, n["wb_train"])) + rng.sample(load_set("train_V1b"), min(16, n["iam_train"])))
    write_set("smoke_train_40", rng.sample(load_set("train_P40"), min(48, len(load_set("train_P40")))))
    write_set("smoke_val", rng.sample(sets["val_wb"], min(12, n["val_wb"])))
    for key in EVAL_SETS.values():
        write_set(f"smoke_{key}", rng.sample(sets[key], min(6, n[key])))

    readable = {g: {key: sum(ctc_need(t) <= occupied(info[p]["h"], info[p]["w"], G["H"], G["W"]) for p, t in sets[key])
                    for key in EVAL_SETS.values()} for g, G in GEOMS.items()}
    names = sorted(p.stem for p in DATA.glob("*.txt"))
    result = dict(counts=n, runs=runs, held_out_sentences=sorted(held), held_out_aug_copies_excluded=len(aug_out),
                  manifest_dimensions_updated=len(dimension_updates),
                  readable_in_text_area=readable, label_sha256={k_: sha(DATA / f"{k_}.txt") for k_ in names})
    if not check_only:
        write_json(done, result)
    log("data ready:", json.dumps(n))
    log(f"long-sentence holdout: {k} of {len(sentences)} distinct sentences -> {n['long_test']} crops held out, "
        f"{len(aug_out)} augmented copies of them excluded from training (new sentences, SAME handwriting and board)")
    log("checks passed: images readable, dictionary, no eval text/image in training, validation identical for every run")
    log(f"using decoded image dimensions; corrected {len(dimension_updates)} stale manifest size records "
        "in generated line_info.json (images and labels unchanged)")
    for name, r in runs.items():
        log(f"  {name}: {r['lines']} training lines; left out {r['left_out'] or 'none'}; {r['diagnostic'] or ''}")
    log("  P40 and P128 training lists are byte-identical; S0-S3 lists identical, no teammates' crop or copy; "
        "G0 = S0 and G0s2 = S0s2 lists byte-identical")
    return result


# ------------------------------------------------------------------ 3. configs and training
def experiment_payload(runs):
    from importlib.metadata import version, PackageNotFoundError
    try:
        paddle_v = version("paddlepaddle-gpu")
    except PackageNotFoundError:
        paddle_v = version("paddlepaddle")
    git = lambda *a: subprocess.check_output(["git", *a], cwd=REPO)
    return json.loads(json.dumps(dict(
        schema=IDENTITY_SCHEMA, runs=runs, test_mode=TEST_MODE, cpu=CPU, static_runtime=STATIC_RUNTIME,
        settings=dict(steps=STEPS, batch=BATCH, lr=LR, warmup=WARMUP_STEPS, eval_step=EVAL_STEP, seed=SEED,
                      max_text=MAX_TEXT, workers=NUM_WORKERS, geoms=GEOMS, run_spec=RUN_SPEC,
                      holdout=[HOLDOUT_FRACTION, HOLDOUT_SEED], export=[EXPORT_SAME_MIN, EXPORT_CER_TOL]),
        data_zip_sha256=sha(DATA_ZIP), newwriter_zip_sha256=sha(NW_ZIP), script_sha256=sha(Path(__file__)),
        repo_commit=git("rev-parse", "HEAD").decode().strip(),
        repo_diff_sha256=hashlib.sha256(git("diff", "--binary", "HEAD")).hexdigest(),
        pretrained_sha256=sha(str(PRETRAINED) + ".pdparams"), paddle=paddle_v, python=list(sys.version_info[:2])),
        default=str))


def initialize_experiment(payload):
    marker = WORK / "experiment.json"
    if marker.exists():
        prev = read_json(marker)
        if prev != payload:
            changed = sorted(k for k in set(prev) | set(payload) if prev.get(k) != payload.get(k))
            raise SystemExit(f"EXPERIMENT MISMATCH in {WORK}: {', '.join(changed)} changed. "
                             "Use a NEW --experiment name; nothing was deleted.")
        log("experiment identity verified:", WORK.name)
        return
    write_json(marker, payload)
    log("new experiment:", WORK)


def make_cfg(save_dir, geom, train_file, eval_file="val_wb", steps=STEPS, eval_step=EVAL_STEP, opts=None):
    import yaml
    opts = opts or {}
    G = GEOMS[geom]
    lines = sum(1 for _ in open(DATA / f"{train_file}.txt", encoding="utf-8"))
    per_epoch = max(1, math.ceil(lines / BATCH))
    epochs = max(1, round(steps / per_epoch))
    planned = epochs * per_epoch
    if planned < 2 * eval_step:                 # short runs (smoke): validate at least twice
        eval_step = max(1, planned // 3)
    cfg = yaml.safe_load(open(BASE_CFG))
    g = cfg["Global"]
    g.update(epoch_num=epochs, save_model_dir=str(save_dir), save_epoch_step=10 ** 6, eval_batch_step=[0, eval_step],
             print_batch_step=10 if planned < 500 else 50, pretrained_model=str(PRETRAINED), checkpoints=None,
             max_text_length=MAX_TEXT, distributed=False, use_gpu=not CPU, seed=opts.get("seed") or SEED,
             d2s_train_image_shape=[3, G["H"], G["W"]], character_dict_path=str(DICT), use_space_char=True,
             save_res_path=str(save_dir / "predicts.txt"))
    cfg["Optimizer"]["lr"].update(learning_rate=LR, warmup_epoch=round(min(WARMUP_STEPS, planned // 4) / per_epoch, 4))
    for h in cfg["Architecture"]["Head"]["head_list"]:
        if "NRTRHead" in h:
            h["NRTRHead"]["max_text_length"] = MAX_TEXT
    cfg["Metric"]["main_indicator"] = "norm_edit_dis"
    if opts.get("lr_mult"):          # stage-wise learning rates (Adam stays Adam; the schedule multiplies all of them)
        assert cfg["Architecture"]["Backbone"]["name"] == "PPHGNetV2_B4" and len(opts["lr_mult"]) == 5
        cfg["Architecture"]["Backbone"]["lr_mult_list"] = list(opts["lr_mult"])
    if opts.get("guide") is False:   # guidance off: the CTC branch's SVTR neck no longer detaches the backbone features
        neck = cfg["Architecture"]["Head"]["head_list"][0]["CTCHead"]["Neck"]
        assert neck["name"] == "svtr" and neck["use_guide"] is True, neck
        neck["use_guide"] = False
    if opts.get("focal"):            # focal CTC on the CTC branch only; MultiLoss passes the dict to CTCLoss
        losses = cfg["Loss"]["loss_config_list"]
        assert cfg["Loss"]["name"] == "MultiLoss" and list(losses[0]) == ["CTCLoss"], losses
        losses[0] = {"CTCLoss": {"use_focal_loss": True}}
    tr = cfg["Train"]
    tr["dataset"].update(data_dir=str(ROOT), label_file_list=[str(DATA / f"{train_file}.txt")])
    tr["sampler"].update(scales=[[G["W"], G["H"]]], first_bs=BATCH)
    tr["loader"].update(batch_size_per_card=BATCH, num_workers=NUM_WORKERS)
    ev = cfg["Eval"]
    ev["dataset"].update(data_dir=str(ROOT), label_file_list=[str(DATA / f"{eval_file}.txt")])
    for op in ev["dataset"]["transforms"]:
        if "RecResizeImg" in op:
            op["RecResizeImg"]["image_shape"] = [3, G["H"], G["W"]]
    ev["loader"].update(batch_size_per_card=min(16, BATCH), num_workers=min(4, NUM_WORKERS))
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "config.yml"
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False, allow_unicode=True)
    return path, dict(lines=lines, steps_per_epoch=per_epoch, epochs=epochs, planned_updates=planned, eval_step=eval_step)


SHOW = re.compile(r"epoch: \[\d+/\d+\], global_step: \d*0,|cur metric|best metric|Traceback|Error|out of memory", re.I)


def train(name, geom, pool40, train_file, eval_file="val_wb", steps=STEPS, eval_step=EVAL_STEP, what="", opts=None):
    d = RUNS / name
    cfg, plan = make_cfg(d, geom, train_file, eval_file, steps, eval_step, opts)
    ident = dict(config=sha(cfg), train_list=sha(DATA / f"{train_file}.txt"), pool40=pool40)
    if (d / "DONE").exists():
        done = read_json(d / "DONE")
        if done.get("identity") != ident or done.get("best_sha256") != sha(d / "best_accuracy.pdparams"):
            raise SystemExit(f"run {name}: finished earlier with a different config/data/weights; refusing to reuse")
        log(f"run {name}: already finished, skipped")
        return d
    cmd = [sys.executable, "-u", "tools/train.py", "-c", cfg]
    parts = [Path(str(d / "latest") + s) for s in (".pdparams", ".pdopt", ".states")]
    if any(p.exists() for p in parts):
        if not all(p.is_file() and p.stat().st_size > 0 for p in parts):
            raise SystemExit(f"run {name}: incomplete resume checkpoint ({[p.name for p in parts if not p.exists()]} "
                             "missing); refusing to restart silently without the optimizer state")
        if (d / "started.json").exists() and read_json(d / "started.json") != ident:
            raise SystemExit(f"run {name}: the saved checkpoint belongs to a different config/data")
        log(f"run {name}: resuming from its last completed epoch (weights + optimizer + state)")
        cmd += ["-o", f"Global.checkpoints={d / 'latest'}"]
    write_json(d / "started.json", ident)
    log(f"=== run {name}: {what} | {GEOMS[geom]['H']}x{GEOMS[geom]['W']} | pooling {'40 positions' if pool40 else 'width/8'} | "
        f"{plan['lines']} lines, {plan['epochs']} epochs x {plan['steps_per_epoch']} = {plan['planned_updates']} updates"
        f"{' | stage-wise LR' if (opts or {}).get('lr_mult') else ''}{' | focal CTC' if (opts or {}).get('focal') else ''}"
        f"{' | guidance OFF (CTC trains the backbone)' if (opts or {}).get('guide') is False else ''}"
        f" | seed {(opts or {}).get('seed') or SEED} ===")
    t0 = time.time()
    run(cmd, d / "train_log.txt", SHOW, env={"B2N_POOL40": "1" if pool40 else "0"})
    if not (d / "best_accuracy.pdparams").exists():
        raise SystemExit(f"run {name}: training ended without a best_accuracy checkpoint (no validation happened?)")
    best = re.findall(r"best metric, (.*)", (d / "train_log.txt").read_text(errors="replace"))
    write_json(d / "DONE", dict(minutes=round((time.time() - t0) / 60, 1), plan=plan, identity=ident,
                               best_metric_line=best[-1] if best else "", best_sha256=sha(d / "best_accuracy.pdparams")))
    prune(d)
    return d


def prune(d):
    """A finished run needs only best_accuracy.pdparams/.states (scoring and export read those). The resume files,
    the optimizer state and PaddleOCR's duplicate best_model/ are ~1.4 GB per run on a small network volume."""
    freed = 0
    for p in [d / "latest.pdparams", d / "latest.pdopt", d / "latest.states", d / "best_accuracy.pdopt", d / "best_model"]:
        if p.is_dir():
            freed += sum(f.stat().st_size for f in p.rglob("*") if f.is_file()); shutil.rmtree(p)
        elif p.is_file():
            freed += p.stat().st_size; p.unlink()
    log(f"{d.name}: removed resume/optimizer copies after DONE ({freed / 1e9:.1f} GB freed)")


# ------------------------------------------------------------------ 4. scoring
def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def flat_dir(key):
    d = DATA / "flat" / key
    d.mkdir(parents=True, exist_ok=True)
    names = {}
    for p, t in load_set(key):
        f = p.replace("/", "__")
        if not (d / f).exists():
            shutil.copy(ROOT / p, d / f)
        names[f] = (p, t)
    return d, names


def rows_from(model, preds, sets):
    info = read_json(DATA / "line_info.json")
    rows = []
    for set_name, key in sets.items():
        _, names = flat_dir(key)
        for f, (p, t) in names.items():
            pr = preds[key].get(f, "")
            diag = insertion_diagnostics(t, pr)
            e, n = diag["edit_distance"], max(len(norm(t)), 1)
            rows.append(dict(model=model, set=set_name, file=f, cluster=info[p]["cluster"], label=t, pred=pr,
                             err=e, chars=n, cer=e / n, exact_case=int(pr.strip() == t.strip()), length=bucket(t),
                             insertions=diag["inserted_chars"], extra_repeats=diag["repeated_inserted_chars"],
                             adjacent_double_excess=max(0, repeats(pr) - repeats(t))))
    return rows


def save_rows(out_dir, rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "predictions.csv"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    tmp.replace(out)
    return rows


def load_rows(out_dir):
    return list(csv.DictReader(open(out_dir / "predictions.csv", encoding="utf-8")))


def cached(out_dir, meta):
    return (out_dir / "predictions.csv").exists() and (out_dir / "scored.json").exists() and read_json(out_dir / "scored.json") == meta


def score_checkpoint(model, weights, cfg, out_dir, sets=EVAL_SETS):
    """Training-format weights read by tools/infer_rec.py: height H, width follows the crop, padded to >= W."""
    meta = dict(weights=sha(str(weights) + ".pdparams"), config=sha(cfg),
                sets={k: sha(DATA / f"{k}.txt") for k in sets.values()})
    if cached(out_dir, meta):
        log(f"{model}: cached predictions verified"); return load_rows(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = {}
    for key in sets.values():
        d, names = flat_dir(key)
        res = out_dir / f"infer_{key}.txt"
        res.unlink(missing_ok=True)
        run([sys.executable, "tools/infer_rec.py", "-c", cfg, "-o", f"Global.pretrained_model={weights}",
             f"Global.infer_img={d}", f"Global.save_res_path={res}", f"Global.use_gpu={not CPU}"],
            out_dir / "infer_log.txt", re.compile(r"Traceback|Error"))
        preds[key] = {}
        for line in open(res, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            preds[key][Path(parts[0]).name] = parts[1] if len(parts) > 1 else ""
        if set(preds[key]) != set(names):
            raise SystemExit(f"{model} {key}: predictions do not match the images")
    rows = save_rows(out_dir, rows_from(model, preds, sets))
    write_json(out_dir / "scored.json", meta)
    return rows


STATIC_READER = r'''
import json, math, sys
from pathlib import Path
import cv2, numpy as np, yaml
import paddle.inference as pi
from ppocr.data.imaug.rec_img_aug import resize_norm_img_chinese
model_dir, flat_root, keys, H, minW, dict_path, out, device = sys.argv[1:9]
H, minW = int(H), int(minW)
y = yaml.safe_load(open(Path(model_dir) / "inference.yml", encoding="utf-8"))
chars = (y.get("PostProcess") or {}).get("character_dict") or open(dict_path, encoding="utf-8").read().splitlines()
graph = Path(model_dir) / "inference.json"
if not graph.exists():
    graph = Path(model_dir) / "inference.pdmodel"
cfg = pi.Config(str(graph), str(Path(model_dir) / "inference.pdiparams"))
cfg.enable_use_gpu(500, 0) if device == "gpu" else cfg.disable_gpu()
# Keep checkpoint/export parity: legacy ir_optim=False alone does not
# suppress the default GPU optimisation list for Paddle 3.2 PIR JSON.
cfg.switch_ir_optim(False)
if not hasattr(cfg, "enable_custom_passes"):
    raise SystemExit("Paddle build lacks enable_custom_passes; refusing an unverified inference runtime")
cfg.enable_custom_passes([], True)
pred = pi.create_predictor(cfg)
inp = pred.get_input_handle(pred.get_input_names()[0])
table, res = None, {}
for key in keys.split(","):
    res[key] = {}
    for f in sorted(Path(flat_root, key).iterdir()):
        img = cv2.imread(str(f))
        if img is None:
            raise SystemExit(f"unreadable image: {f}")
        # Use the checkpoint reader's implementation, including its floor/ceil
        # rules for lines wider than minW. A one-pixel difference changes strokes.
        pad, _ = resize_norm_img_chinese(img, [3, H, minW])
        W = pad.shape[2]
        inp.reshape([1, 3, H, W]); inp.copy_from_cpu(pad[None]); pred.run()
        prob = pred.get_output_handle(pred.get_output_names()[0]).copy_to_cpu()[0]
        if table is None:
            table = [""] + list(chars) + ([" "] if prob.shape[-1] == len(chars) + 2 else [])
            if prob.shape[-1] != len(table):
                raise SystemExit(f"model has {prob.shape[-1]} classes but the dictionary gives {len(table)}")
        idx = prob.argmax(-1)
        res[key][f.name] = "".join(table[i] for k, i in enumerate(idx) if i != 0 and (k == 0 or i != idx[k - 1]))
json.dump(res, open(out, "w"), ensure_ascii=False)
print("STATIC READER DONE", {k: len(v) for k, v in res.items()})
'''


def score_static(model, model_dir, H, minW, out_dir, sets=EVAL_SETS):
    """An exported inference model read the way PaddleOCR deploys it (height H, width follows the crop, >= minW)."""
    model_dir = Path(model_dir)
    graph = model_dir / "inference.json"
    if not graph.exists():
        graph = model_dir / "inference.pdmodel"
    meta = dict(runtime=STATIC_RUNTIME, params=sha(model_dir / "inference.pdiparams"), graph=sha(graph),
                inference_config=sha(model_dir / "inference.yml"),
                H=H, minW=minW, dict=sha(DICT) if DICT.exists() else "",
                sets={k: sha(DATA / f"{k}.txt") for k in sets.values()})
    if cached(out_dir, meta):
        log(f"{model}: cached predictions verified"); return load_rows(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in sets.values():
        flat_dir(key)
    raw = out_dir / "static_preds.json"
    run([sys.executable, "-c", STATIC_READER, model_dir, DATA / "flat", ",".join(sets.values()), H, minW, DICT, raw,
         "cpu" if CPU else "gpu"], out_dir / "infer_log.txt", re.compile(r"Traceback|Error|DONE"))
    rows = save_rows(out_dir, rows_from(model, read_json(raw), sets))
    write_json(out_dir / "scored.json", meta)
    return rows


# ------------------------------------------------------------------ 5. export with a declared equivalence check
def export(name, geom, out):
    d = RUNS / name
    G = GEOMS[geom]
    ident = dict(runtime=STATIC_RUNTIME, checkpoint=sha(d / "best_accuracy.pdparams"), config=sha(d / "config.yml"), export_h=G["H"],
                 fixture=sha(DATA / "export_fixture.txt"))
    if (out / "verified.json").exists() and read_json(out / "verified.json").get("identity") == ident:
        log(f"{name}: verified export reused"); return out
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{name}-export-", dir=out.parent))
    run([sys.executable, "tools/export_model.py", "-c", d / "config.yml", "-o",
         f"Global.pretrained_model={d / 'best_accuracy'}", f"Global.save_inference_dir={stage}",
         "Global.model_name=PP-OCRv5_server_rec"], d / "export_log.txt", re.compile(r"Traceback|Error|saved", re.I),
        env={"B2N_EXPORT_H": str(G["H"])})
    fx = {"export fixture": "export_fixture"}
    a = {r["file"]: r for r in score_checkpoint(f"{name} checkpoint", d / "best_accuracy", d / "config.yml", d / "fixture_checkpoint", fx)}
    b = {r["file"]: r for r in score_static(f"{name} exported", stage, G["H"], G["W"], d / "fixture_exported", fx)}
    same = sum(a[f]["pred"] == b[f]["pred"] for f in a) / len(a)
    dcer = abs(sum(float(a[f]["cer"]) for f in a) - sum(float(b[f]["cer"]) for f in a)) / len(a)
    diffs = [(a[f]["label"], a[f]["pred"], b[f]["pred"]) for f in a if a[f]["pred"] != b[f]["pred"]]
    if same < EXPORT_SAME_MIN or dcer > EXPORT_CER_TOL:
        raise SystemExit(f"EXPORT CHECK FAILED for {name}: identical predictions {same:.0%} (need {EXPORT_SAME_MIN:.0%}), "
                         f"mean CER difference {dcer:.4f} (max {EXPORT_CER_TOL}). Examples (label | checkpoint | exported): "
                         f"{diffs[:5]}. The checkpoint is kept; the export was NOT published.")
    write_json(stage / "verified.json", dict(identity=ident, identical_predictions=same, mean_cer_difference=dcer,
                                            fixture_crops=len(a), differing=diffs))
    if out.exists():
        out.rename(out.with_name(out.name + f"_previous_{time.time_ns()}"))
    stage.rename(out)
    log(f"{name}: export verified on {len(a)} validation crops: {same:.0%} identical, mean CER difference {dcer:.4f}")
    return out


# ------------------------------------------------------------------ 6. report
def summarize(rows, by=("model", "set")):
    out = {}
    for r in rows:
        s = out.setdefault(tuple(r[k] for k in by), [0, 0, 0, 0.0, 0, 0])
        s[0] += 1; s[1] += float(r["err"]); s[2] += float(r["chars"]); s[3] += float(r["cer"])
        s[4] += float(r["exact_case"]); s[5] += float(r["extra_repeats"])
    return {k: dict(lines=n, mean_line_CER=c / n, corpus_CER=e / max(ch, 1), exact=x / n, extra_repeats=rp / n)
            for k, (n, e, ch, c, x, rp) in out.items()}


def bootstrap(a_rows, b_rows, field="cer"):
    """Mean difference A - B on the same crops; 95% interval resampling whole clusters (photos; for the held-out
    long sentences, whole sentences), because crops from one photo or one sentence are not independent."""
    A = {r["file"]: float(r[field]) for r in a_rows}; B = {r["file"]: float(r[field]) for r in b_rows}
    assert set(A) == set(B), "comparison on different crops"
    by = defaultdict(list)
    for r in a_rows:
        by[r["cluster"]].append(A[r["file"]] - B[r["file"]])
    groups = list(by.values())
    mean = lambda gs: sum(sum(g) for g in gs) / sum(len(g) for g in gs)
    rng = random.Random(0)
    sims = sorted(mean([groups[rng.randrange(len(groups))] for _ in groups]) for _ in range(BOOTSTRAP))
    better = sum(A[f] < B[f] - 1e-9 for f in A); worse = sum(A[f] > B[f] + 1e-9 for f in A)
    return mean(groups), sims[int(0.025 * BOOTSTRAP)], sims[int(0.975 * BOOTSTRAP) - 1], better, worse, len(groups)


LONG_VIEW = "held-out long lines only (over 40 chars)"


def seed_average(results):
    """Add 'S0avg' etc.: per crop, the mean of the seed-1 and seed-2 runs (only when both finished)."""
    for base in ("S0", "S1", "S2", "S3", "G0"):
        a, b = results.get(base), results.get(base + "s2")
        if not a or not b:
            continue
        bb = {(r["set"], r["file"]): r for r in b}
        assert set(bb) == {(r["set"], r["file"]) for r in a}, f"{base}: seeds scored on different crops"
        avg = []
        for r in a:
            o = bb[(r["set"], r["file"])]
            row = dict(r, model=base + "avg", pred=f"[seed 1] {r['pred']} | [seed 2] {o['pred']}")
            for k in ("err", "chars", "cer", "exact_case", "extra_repeats", "insertions", "adjacent_double_excess"):
                if k in r:
                    row[k] = (float(r[k]) + float(o[k])) / 2
            avg.append(row)
        results[base + "avg"] = avg


def report(results, prepared, out_dir, comparisons=COMPARISONS,
           test_sets=("CNN test", "long sentences (held out)", LONG_VIEW)):
    seed_average(results)
    if any(m in results for m in G_RUNS):
        comparisons = G_COMPARISONS
        test_sets = ("whiteboard val",) + tuple(test_sets) + ("new writers", "IAM val")
    elif any(m in results for m in S_RUNS):
        comparisons = S_COMPARISONS
        test_sets = ("whiteboard val",) + tuple(test_sets) + ("new writers", "IAM val")
    for m in list(results):   # a view of the held-out set with only lines over 40 characters (the long-line test)
        results[m] = [r for r in results[m] if r["set"] != LONG_VIEW] + [
            {**r, "set": LONG_VIEW} for r in results[m] if r["set"] == "long sentences (held out)" and r["length"] == "over 40 chars"]
    out_dir.mkdir(parents=True, exist_ok=True)
    allrows = [r for rows in results.values() for r in rows]
    summ = summarize(allrows)
    known = ["V1", "pretrained", "V1b", "V2short", "V3short", "V2", "V3", "C1", "P40", "P128"] + \
            [x for b in ("S0", "S1", "S2", "S3", "G0") for x in (b, b + "s2", b + "avg")]
    order = [m for m in known if m in results] + [m for m in results if m not in known]
    sets_present = list(dict.fromkeys(r["set"] for r in allrows))
    L = [f"# Board2Notes reader versions: results", "",
         f"Experiment `{EXPERIMENT}`; about {STEPS} updates x {BATCH} lines per training run (exact counts in "
         "'Training runs'; runs on the same training list have identical counts); "
         f"{'two seeds for S0-S3 (avg = per-crop mean of both)' if 'S0s2' in results else 'one seed'}. "
         "**Mean line CER**: per crop, case-insensitive, spaces normalised (lower is better). **Exact**: crop read "
         "perfectly, case-sensitive. **Repeated insertions**: aligned inserted characters inside repeated substrings "
         "of 1-16 characters, per crop. This is a diagnostic, not proof of a causal mechanism. "
         "Raw insertion counts and the older adjacent-double diagnostic are saved in predictions.csv.", "",
         ("Every training run, including those 'without our augmented copies', also gets PaddleOCR's built-in random "
          "augmentation (RecAug, applied in memory). The 2x2 therefore tests OUR augmented copies on top of it."
          if "V2short" in results else "Every training run gets PaddleOCR's built-in random augmentation (RecAug, in memory)."), "",
         "The held-out long sentences are new SENTENCES in the SAME handwriting on the SAME board; they do not test a "
         "new writer or a new board.", "", "| version | what |", "|---|---|",
         "| V1 | historical Baseline_model (Oct 13), read as deployed (48 px, width follows the crop) |",
         "| pretrained | PP-OCRv5_server_rec as downloaded, no fine-tuning (48 px) |"]
    for m in order:
        if m in RUN_SPEC:
            L.append(f"| {m} | {RUN_SPEC[m]['what']} |")
        elif m.endswith("avg") and m[:-3] in RUN_SPEC:
            L.append(f"| {m} | per-crop mean of {m[:-3]} and {m[:-3]}s2 |")
    L += ["", "## All sets", "", "| version | set | crops | mean line CER | corpus CER | exact | repeated insertions |",
          "|---|---|---|---|---|---|---|"]
    for s in sets_present:
        for m in order:
            x = summ.get((m, s))
            if x:
                L.append(f"| {m} | {s} | {x['lines']} | {x['mean_line_CER']:.4f} | {x['corpus_CER']:.4f} | {x['exact']:.3f} | {x['extra_repeats']:.3f} |")
    verdicts = {}
    primary_at = len(L)              # the primary-criterion table is inserted here, above the detailed comparisons
    for s in test_sets:
        L += ["", f"## Comparisons on {s} (rules fixed before training)", "",
              "**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos "
              "(or whole sentences for the held-out long sentences).", "",
              "| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |",
              "|---|---|---|---|---|---|---|"]
        for a, b, what in comparisons:
            ra = [r for r in results.get(a, []) if r["set"] == s]; rb = [r for r in results.get(b, []) if r["set"] == s]
            if not ra or not rb:
                continue
            d, lo, hi, bt, wr, ng = bootstrap(ra, rb)
            v = "better" if hi < 0 else ("worse" if lo > 0 else "no clear difference")
            verdicts[f"{s}: {a} vs {b}"] = dict(diff=d, lo=lo, hi=hi, verdict=v)
            L.append(f"| {a} vs {b} | {what} | {d:+.4f} | {lo:+.4f} .. {hi:+.4f} | {bt} / {wr} | {ng} | {v} |")
    pairs = [(t, c) for t, c in (("S1", "S0"), ("S2", "S0"), ("S3", "S0"), ("G0", "S0"))
             if f"{t}avg" in results and f"{c}avg" in results and any(x[0] == f"{t}avg" for x in comparisons)]
    if pairs:
        P = ["", "## Primary criterion (fixed before training): whiteboard validation, mean of 2 seeds", "",
              "A setting **helps** only if the seed-averaged 95% interval is below 0 AND both seeds improve on their "
              "own baseline. Whiteboard validation also chose each run's checkpoint, so its absolute values are slightly "
              "optimistic, equally for every run.", "",
              "If the rule is not met the conclusion is 'improvement not established', which is not proof of no effect; "
              "only a consistent deterioration (interval above 0 and both seeds worse) is reported as 'hurts'.", "",
              "| setting | seed 1 vs S0 | seed 2 vs S0s2 | mean of seeds | 95% interval | decision |", "|---|---|---|---|---|---|"]
        s = "whiteboard val"
        for t, c in pairs:
            v = verdicts.get(f"{s}: {t}avg vs {c}avg"); v1 = verdicts.get(f"{s}: {t} vs {c}"); v2 = verdicts.get(f"{s}: {t}s2 vs {c}s2")
            if v and v1 and v2:
                ok = v["hi"] < 0 and v1["diff"] < 0 and v2["diff"] < 0
                bad = v["lo"] > 0 and v1["diff"] > 0 and v2["diff"] > 0
                dec = "helps" if ok else ("hurts" if bad else "improvement not established")
                verdicts[f"primary {t}"] = dict(diff=v["diff"], lo=v["lo"], hi=v["hi"], verdict=dec)
                P.append(f"| {t} ({RUN_SPEC[t]['what'].rsplit(' (seed', 1)[0]}) | {v1['diff']:+.4f} | {v2['diff']:+.4f} | "
                         f"{v['diff']:+.4f} | {v['lo']:+.4f} .. {v['hi']:+.4f} | **{dec}** |")
        sn = verdicts.get("whiteboard val: S0s2 vs S0")
        if sn:
            P += ["", f"For scale: the same recipe with a different seed (S0s2 vs S0) differs by {sn['diff']:+.4f}. "
                  "Read the IAM and test tables below for what each setting does elsewhere; they do not change the decision."]
        L[primary_at:primary_at] = P
    if all(m in results for m in ("V2short", "V3short", "V2", "V3")):
        L += ["", "## 2x2: do our augmented copies help MORE when long sentences are included? (interaction)", "",
              "Per crop: (D - C) - (B - A) in CER, i.e. the gain from our augmented copies with long sentences minus "
              "the gain without them. Negative = the copies help more when long sentences are present.", "",
              "| set | interaction | 95% interval | verdict |", "|---|---|---|---|"]
        for s in test_sets:
            q = {m: {r["file"]: r for r in results[m] if r["set"] == s} for m in ("V2short", "V3short", "V2", "V3")}
            if not q["V2"]:
                continue
            diff = [dict(file=f, cluster=r["cluster"], cer=(float(q["V3"][f]["cer"]) - float(r["cer"]))
                         - (float(q["V3short"][f]["cer"]) - float(q["V2short"][f]["cer"]))) for f, r in q["V2"].items()]
            zero = [dict(d, cer=0.0) for d in diff]
            d, lo, hi, *_ = bootstrap(diff, zero)
            v = "copies help more with long data" if hi < 0 else ("copies help less with long data" if lo > 0 else "no clear interaction")
            verdicts[f"interaction {s}"] = dict(diff=d, lo=lo, hi=hi, verdict=v)
            L.append(f"| {s} | {d:+.4f} | {lo:+.4f} .. {hi:+.4f} | {v} |")
        L += ["", "Caution: short + long runs also have ~28% more original training lines, so 'adding long sentences' "
              "is partly 'more data'. Gains concentrated on the long-line test point to long sentences specifically."]
    pa, pb = ("P40", "P128") if "P40" in results else (None, None)
    if pa:
        L += ["", "## Repetition diagnostic (P40 vs P128 only)", "",
              "The hypothesis is that the training/inference position mismatch contributes to repetition. "
              "Count aligned insertions inside repeated substrings, including repeated syllables. "
              "Fewer flags are consistent with this hypothesis but do not prove it. Inspect the saved predictions "
              "and CER together; alignment and the substring rule are imperfect.", "",
              "| set | repeated insertions P40 - P128 | 95% interval | diagnostic |", "|---|---|---|---|"]
        for s in sets_present:
            ra = [r for r in results[pa] if r["set"] == s]; rb = [r for r in results[pb] if r["set"] == s]
            if ra and rb:
                d, lo, hi, *_ = bootstrap(ra, rb, "extra_repeats")
                v = "P40 has more repetition flags" if lo > 0 else (
                    "P128 has more repetition flags" if hi < 0 else "no clear difference in flags")
                verdicts[f"repeats {s}"] = dict(diff=d, lo=lo, hi=hi, verdict=v)
                L.append(f"| {s} | {d:+.3f} | {lo:+.3f} .. {hi:+.3f} | {v} |")
    nw = [dict(r, writer=r["cluster"].split(":")[0]) for r in allrows if r["set"] == "new writers"]
    if nw:
        bw = summarize(nw, ("model", "writer"))
        writers = sorted({r["writer"] for r in nw})
        L += ["", "## New writers, by writer (mean line CER)", "",
              "Teammates' handwriting that no S0-S3 run trained on (the V-runs of versions_2x2 saw 64 teammates' crops of "
              "unknown writer). Two writers only: treat as a first look at generalisation, not a population estimate.", "",
              "| model | " + " | ".join(f"{w} ({bw[(order[0] if (order[0], w) in bw else next(m for m in order if (m, w) in bw), w)]['lines']})" for w in writers) + " |",
              "|---|" + "---|" * len(writers)]
        for m in order:
            if any((m, w) in bw for w in writers):
                L.append(f"| {m} | " + " | ".join(f"{bw[(m, w)]['mean_line_CER']:.4f}" if (m, w) in bw else "" for w in writers) + " |")
    bl = summarize(allrows, ("model", "set", "length"))
    L += ["", "## By label length (does a version fail on long lines?)", "",
          "| set | length | " + " | ".join(order) + " |", "|---|---|" + "---|" * len(order)]
    for s in sets_present:
        for b in ("1-20 chars", "21-40 chars", "over 40 chars"):
            cells = [bl.get((m, s, b)) for m in order]
            if any(cells):
                n = next(c["lines"] for c in cells if c)
                L.append(f"| {s} | {b} ({n}) | " + " | ".join(f"{c['mean_line_CER']:.3f}" if c else "" for c in cells) + " |")
    L += ["", "## Training lines per run (only hard limits are applied)", "",
          "| run | lines | left out (reason) | kept although the text covers fewer positions than it needs |", "|---|---|---|---|"]
    for m, r in prepared["runs"].items():
        if m in results:
            L.append(f"| {m} | {r['lines']} | {r['left_out'] or 'none'} | {sum(r['diagnostic'].values())} |")
    L += ["", "## Training runs", "", "| run | lines | epochs | updates | minutes | best validation |", "|---|---|---|---|---|---|"]
    for m in order:
        if (RUNS / m / "DONE").exists():
            dn = read_json(RUNS / m / "DONE")
            L.append(f"| {m} | {dn['plan']['lines']} | {dn['plan']['epochs']} | {dn['plan']['planned_updates']} | {dn['minutes']} | {dn['best_metric_line'][:70]} |")
    L += ["", ("Limits: two seeds per setting (S0s2 vs S0 shows the size of seed-to-seed change); "
               if "S0s2" in results or "G0s2" in results else "Limits: one seed per run; ") +
          "the intervals reflect evaluation-cluster sampling, not training-seed "
          "variability. Validation shares source photos "
          "with training (different texts); the CNN test board has few distinct texts (see predictions_all.csv); "
          f"the held-out long sentences are {len(prepared['held_out_sentences'])} sentences from one board and writer; "
          "the new-writer set is 160 crops from 2 writers (intervals resample whole source photos)."]
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    write_json(out_dir / "summary.json", dict(summary={" | ".join(k): v for k, v in summ.items()}, comparisons=verdicts,
                                              held_out_sentences=prepared["held_out_sentences"]))
    save_rows(out_dir, allrows)
    print("\n".join(L), flush=True)


def pack():
    z = WORK / f"results_{EXPERIMENT}.zip"
    tmp = z.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in RESULTS.rglob("*"):
            if p.is_file() and "_previous_" not in str(p) and not any(x.startswith(".") for x in p.relative_to(RESULTS).parts):
                zf.write(p, Path("results") / p.relative_to(RESULTS))
        for p in RUNS.glob("*/*"):
            if p.is_file() and (p.suffix in (".txt", ".yml", ".csv", ".json") or p.name == "DONE"):
                zf.write(p, Path("runs") / p.relative_to(RUNS))
        zf.write(DATA / "prepared.json", "data/prepared.json"); zf.write(WORK / "experiment.json", "experiment.json")
        zf.write(Path(__file__), "source/b2n_versions.py")
    tmp.replace(z)
    shutil.copy(z, HERE / z.name)
    log(f"results: {HERE / z.name} ({z.stat().st_size / 1e6:.0f} MB)")


# ------------------------------------------------------------------ 7. steps
def score_baselines(sets, out_root):
    pre_cfg, _ = make_cfg(out_root / "pretrained", "h48", "train_V3")
    res = {"pretrained": score_checkpoint("pretrained", PRETRAINED, pre_cfg, out_root / "pretrained", sets)}
    res["V1"] = score_static("V1", ROOT / "baseline_v1", 48, 320, out_root / "V1", sets)
    return res


GRAD_CHECK = r"""
import json, sys, yaml, paddle
sys.path.insert(0, ".")
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
paddle.set_device(sys.argv[1])
out = {}
for tag, path in zip(sys.argv[2::2], sys.argv[3::2]):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    n = len(build_post_process(cfg["PostProcess"], cfg["Global"]).character)
    cfg["Architecture"]["Head"]["out_channels_list"] = {"CTCLabelDecode": n, "NRTRLabelDecode": n + 3}
    paddle.seed(0)
    m = build_model(cfg["Architecture"]); m.train()
    x = paddle.randn([2, 3, 64, 256])
    y = m.head.ctc_head(m.head.ctc_encoder(m.backbone(x)))      # the CTC branch only
    y = y[-1] if isinstance(y, (list, tuple)) else y
    (y.astype("float32") ** 2).mean().backward()
    g = lambda mod: sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)
    out[tag] = dict(use_guide=cfg["Architecture"]["Head"]["head_list"][0]["CTCHead"]["Neck"]["use_guide"],
                    backbone=g(m.backbone), ctc_neck=g(m.head.ctc_encoder))
print("GRADCHECK " + json.dumps(out))
"""


def check_guide_switch(d):
    """(1) G0's config differs from S0's ONLY in use_guide (and its own paths/list name); same for G0s2 vs S0s2.
    (2) With guidance on, a CTC-only loss sends zero gradient to the backbone; with it off, a nonzero one."""
    import yaml
    def flat(x, k=""):
        if isinstance(x, dict):
            return {kk: vv for a, b in x.items() for kk, vv in flat(b, f"{k}.{a}" if k else str(a)).items()}
        if isinstance(x, list):
            return {kk: vv for i, b in enumerate(x) for kk, vv in flat(b, f"{k}.{i}").items()}
        return {k: x}
    cfgs = {}
    for g, s in (("G0", "S0"), ("G0s2", "S0s2")):
        fs = {}
        for r in (g, s):
            spec = RUN_SPEC[r]
            path, _ = make_cfg(d / f"cfgcheck_{r}", spec["geom"], f"train_{r}",
                               opts=dict(guide=spec.get("guide"), seed=spec.get("seed")))
            fs[r] = flat(yaml.safe_load(open(path))); cfgs[r] = path
        diff = {k for k in set(fs[g]) | set(fs[s]) if fs[g].get(k) != fs[s].get(k)}
        allowed = {"Architecture.Head.head_list.0.CTCHead.Neck.use_guide", "Global.save_model_dir", "Global.save_res_path",
                   "Train.dataset.label_file_list.0"}
        assert diff <= allowed and "Architecture.Head.head_list.0.CTCHead.Neck.use_guide" in diff, f"{g} vs {s}: {sorted(diff)}"
        assert fs[g]["Global.seed"] == fs[s]["Global.seed"] and fs[g]["Global.pretrained_model"] == fs[s]["Global.pretrained_model"]
        losses = [k for k in fs[g] if k.startswith("Loss.")]
        assert not any("weight" in k for k in losses), losses          # both loss weights stay at the default 1.0
    log("  config check: G0 vs S0 and G0s2 vs S0s2 differ only in use_guide (plus their own output paths and list "
        "names; list contents identical, same seed, same pretrained weights, loss weights 1.0)")
    lf = d / "guide_check.log"
    lf.unlink(missing_ok=True)
    run([sys.executable, "-c", GRAD_CHECK, "cpu" if CPU else "gpu", "guided", str(cfgs["S0"]), "unguided", str(cfgs["G0"])],
        logfile=lf, show=re.compile("GRADCHECK"))
    out = lf.read_text(errors="replace")
    line = [l for l in out.splitlines() if l.startswith("GRADCHECK ")]
    assert line, out[-2000:]
    gc = json.loads(line[-1][len("GRADCHECK "):])
    assert gc["guided"]["use_guide"] is True and gc["unguided"]["use_guide"] is False, gc
    assert gc["guided"]["backbone"] == 0.0 and gc["guided"]["ctc_neck"] > 0, gc
    assert gc["unguided"]["backbone"] > 0 and gc["unguided"]["ctc_neck"] > 0, gc
    write_json(d / "guide_check.json", gc)
    log(f"  gradient check (CTC branch only): guided backbone grad = 0, unguided backbone grad = {gc['unguided']['backbone']:.3g} > 0")


def smoke(prepared, runs=None):
    """Every code path the full run uses, on a few lines: all three training setups, both scorers, export with the
    equivalence check at 64 px and 48 px, the report. Writes OK only if everything passed."""
    d = RUNS / "smoke"
    if (d / "OK").exists():
        log("SMOKE OK: passed earlier for these exact experiment settings; skipped"); return
    setups = {}
    for r in (runs or ["P128", "P40", "C1"]):
        s = RUN_SPEC[r]
        key = (s["geom"], s["pool40"], tuple(s.get("lr_mult") or ()), bool(s.get("focal")), s.get("guide", True))
        setups.setdefault(key, (f"smoke_{r.removesuffix('s2')}", "smoke_train_40" if s["positions"] == 40 else "smoke_train_h64"))
    log(f"=== SMOKE TEST: {len(setups)} training setups ({', '.join(n for n, _ in setups.values())}); scoring; export; report ===")
    if any(r in G_RUNS for r in (runs or [])):
        check_guide_switch(d)
    sets = {s: f"smoke_{k}" for s, k in EVAL_SETS.items()}
    results = score_baselines(sets, d)
    for (geom, pool40, lr_mult, focal, guide), (name, lst) in setups.items():
        rd = train(name, geom, pool40, lst, eval_file="smoke_val", steps=24 if not TEST_MODE else STEPS,
                   eval_step=8, what="smoke", opts=dict(lr_mult=list(lr_mult) or None, focal=focal, guide=guide))
        results[name] = score_checkpoint(name, rd / "best_accuracy", rd / "config.yml", rd / "scores", sets)
        export(name, geom, rd / "inference")
    names = [n for n, _ in setups.values()]
    report(results, prepared, d / "report", comparisons=[(n, "pretrained", "smoke") for n in names] + [(names[0], "V1", "smoke")])
    write_json(d / "OK", dict(at=time.strftime("%Y-%m-%d %H:%M:%S"), models=sorted(results)))
    log(f"SMOKE OK: training ({len(setups)} setups), scoring (checkpoint + exported), export checks, report all work. "
        "Smoke CER values mean nothing.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["all", "check"])
    ap.add_argument("--experiment", default="guide_g0")
    ap.add_argument("--runs", default=",".join(G_RUNS), help="comma list from " + ",".join(RUN_SPEC))
    ap.add_argument("--smoke-only", action="store_true")
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", a.experiment):
        ap.error("--experiment: letters, digits, _ or - only")
    runs = [r for r in a.runs.split(",") if r]
    if [r for r in runs if r not in RUN_SPEC]:
        ap.error(f"unknown runs in {runs}")
    if ("P40" in runs) != ("P128" in runs):
        ap.error("P40 and P128 are a matched pair: run both or neither")
    set_paths(a.experiment)
    if a.step == "check":
        prepare(check_only=True); return
    patch()
    initialize_experiment(experiment_payload(runs))
    RUNS.mkdir(parents=True, exist_ok=True)
    prepared = prepare()
    smoke(prepared, runs)
    if a.smoke_only:
        log("smoke test only: stopping here"); return
    results = score_baselines(EVAL_SETS, RUNS)
    for name in runs:
        s = RUN_SPEC[name]
        d = train(name, s["geom"], s["pool40"], f"train_{name}", what=s["what"],
                  opts=dict(lr_mult=s.get("lr_mult"), focal=s.get("focal"), seed=s.get("seed"), guide=s.get("guide")))
        results[name] = score_checkpoint(name, d / "best_accuracy", d / "config.yml", d / "scores")
        cer = summarize(results[name])[(name, "whiteboard val")]["mean_line_CER"]
        log(f"{name}: whiteboard val mean line CER {cer:.4f}")
    for name in runs:
        export(name, RUN_SPEC[name]["geom"], RESULTS / "models" / f"PP-OCRv5_server_rec_board2notes_{name}")
    report(results, prepared, RESULTS)
    pack()
    log("DONE")


if __name__ == "__main__":
    main()
