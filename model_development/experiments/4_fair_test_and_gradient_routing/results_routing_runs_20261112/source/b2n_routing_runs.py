#!/usr/bin/env python3
"""Board2Notes server reader: the fair guidance test and the gradient-routing runs (12 Nov 2026).

Why: G0 (guidance off) beat S0 on unfamiliar handwriting, but G0's chosen checkpoints are from epochs 15/19 and S0's
from 38/71, so "guidance off" and "trained less" were mixed up. Here every run saves a checkpoint every 5 epochs and
every checkpoint is scored, so recipes are compared AT THE SAME EPOCH as well as at their validation-chosen best.

Runs (all: S0's data = whiteboard train + long train, 64x1024, width/8 positions, same budget, LR and loss weights 1/1):
  S0, S0s2   guidance on: only the NRTR loss trains the backbone                         (seed 1, 2)
  G0, G0s2   guidance off: the CTC loss also trains the whole backbone                    (seed 1, 2)
  RE, REs2   routing "early": CTC trains the stem + stages 1-2 (and its neck/head); stages 3-4 get NRTR only
  RL, RLs2   routing "late":  CTC trains stages 3-4 (and its neck/head); stem + stages 1-2 get NRTR only
Order: S0, G0, S0s2, G0s2 (the fair test), then the layer diagnosis at epochs 15 and 40, then RE, RL, REs2, RLs2.
Partial results are zipped after every run, so a stopped pod still leaves usable results.

Scored at every 5th epoch (5..30, then 40, 50, 60, 70): whiteboard validation (familiar), new writers, fresh boards
(152 crops packaged here, test only). The validation-chosen best checkpoint is also scored on every usual set.
Layer diagnosis (b2n_routing_diag.py worker, no training): at epochs 15 and 40 of S0/S0s2/G0/G0s2, per part of the
network the CTC vs NRTR gradient, the weight change from pretrained and the stage-feature similarity; at epoch 40
also the CTC head vs NRTR head readings. Because both recipes are compared at the same epoch, the training-length
confound of the 12 Nov local analysis does not apply.

Usage (through run_routing_runs.sh):  python b2n_routing_runs.py all
Data/plan check only (seconds):       python b2n_routing_runs.py check
"""
from __future__ import annotations

import argparse, csv, json, os, random, re, shutil, subprocess, sys, time, urllib.request, zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2n_versions as bv  # noqa: E402
import b2n_routing_patch as rp  # noqa: E402

SERVER_SCRIPT_SHA256 = "a4f4ff156075772a99b4c9a6eaabeead3401442f4ad04f5a167cc9e26eaa19a8"
PRETRAINED_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams"
FAKE = os.environ.get("B2N_RR_FAKE_DATA") == "1"          # CPU code test only: reuse prepared lists, never on RunPod
SAVE_EVERY = int(os.environ.get("B2N_RR_SAVE_EVERY", 5))
SCORE_EPOCHS = [int(x) for x in os.environ.get("B2N_RR_SCORE_EPOCHS", "5,10,15,20,25,30,40,50,60,70").split(",")]
DIAG_EPOCHS = [int(x) for x in os.environ.get("B2N_RR_DIAG_EPOCHS", "15,40").split(",")]
FAIR = ["S0", "G0", "S0s2", "G0s2"]
ROUTED = ["RE", "RL", "REs2", "RLs2"]
ORDER = FAIR + ROUTED
RECIPES = {"S0": ["S0", "S0s2"], "G0": ["G0", "G0s2"], "RE": ["RE", "REs2"], "RL": ["RL", "RLs2"]}
WHAT = {"S0": "guidance on (NRTR trains the backbone)", "G0": "guidance off (CTC + NRTR train the whole backbone)",
        "RE": "routing early: CTC trains stem + stages 1-2", "RL": "routing late: CTC trains stages 3-4"}
CTC_TRAINS = {"RE": "stem,1,2", "RL": "3,4"}
for _seed, _sfx in ((1, ""), (2, "s2")):
    for _n, _parts in CTC_TRAINS.items():
        bv.RUN_SPEC[_n + _sfx] = dict(geom="h64", pool40=False, positions=128, files=["wb_train", "long_train"],
                                      guide=False, seed=_seed, ctc_trains=_parts,
                                      what=f"{WHAT[_n]}; NRTR trains everything (seed {_seed})")
EP_SETS = {"whiteboard val": "val_wb", "new writers": "test_newwriter"}
FRESH = HERE / "fresh_boards"
COMPARE = [("G0", "S0", "guidance off vs on (THE fair test)"), ("RE", "G0", "routing early vs full guidance off"),
           ("RL", "G0", "routing late vs full guidance off"), ("RE", "S0", "routing early vs guidance on"),
           ("RL", "S0", "routing late vs guidance on"), ("RE", "RL", "early vs late")]
log = bv.log


# ------------------------------------------------------------------ setup
def link_data():
    for other in ("guide_g0", "strategies_s0s3", "server_c0_20261111", "mobile_official_20261110"):
        src = bv.STORAGE / "experiments" / other / "data" / "b2n_data"
        if (src / "reader/manifest.csv").exists():
            bv.ROOT.mkdir(parents=True, exist_ok=True)
            for c in src.iterdir():
                if c.name != "his" and not (bv.ROOT / c.name).exists():
                    (bv.ROOT / c.name).symlink_to(c.resolve(), target_is_directory=c.is_dir())
            log(f"data folders linked from {other}"); return other
    log("no extracted data found on the volume: the data zips will be extracted (~0.8 GB)")


def ensure_pretrained():
    f = Path(str(bv.PRETRAINED) + ".pdparams")
    if not f.is_file() or f.stat().st_size == 0:
        log("downloading", PRETRAINED_URL)
        urllib.request.urlretrieve(PRETRAINED_URL, str(f) + ".part")
        os.replace(str(f) + ".part", f)
    log(f"server pretrained weights: {f} ({bv.sha(f)[:12]})")


def check_lists():
    for r in ROUTED:
        twin = "G0" + ("s2" if r.endswith("s2") else "")
        assert bv.sha(bv.DATA / f"train_{r}.txt") == bv.sha(bv.DATA / f"train_{twin}.txt"), f"{r} must train on {twin}'s list"
    out = dict(lines=len(bv.load_set("train_S0")))
    old = bv.STORAGE / "experiments" / "guide_g0" / "data" / "train_G0.txt"
    if old.is_file():
        out["same_list_as_guide_g0"] = bv.sha(old) == bv.sha(bv.DATA / "train_G0.txt")
        log(f"training list identical to the 7 Nov G0 run: {out['same_list_as_guide_g0']}")
    log(f"lists: S0 = G0 = RE = RL (seed 1) and the same for seed 2; {out['lines']} training lines")
    return out


def fresh_items():
    if not (FRESH / "labels.tsv").is_file():
        return []
    out = []
    for p, t in bv.read_labels(FRESH / "labels.tsv")[:int(os.environ.get("B2N_RR_FRESH_LIMIT", 10 ** 6))]:
        board, _photo, line = Path(p).stem.split("_")           # A_5095_1 -> writing A1 (one board, one line)
        out.append((Path(p).name, FRESH / p, t, board + line))
    return out


# ------------------------------------------------------------------ training with saved epochs
_orig_make_cfg = bv.make_cfg


def make_cfg(save_dir, geom, train_file, eval_file="val_wb", steps=bv.STEPS, eval_step=bv.EVAL_STEP, opts=None):
    import yaml
    path, plan = _orig_make_cfg(save_dir, geom, train_file, eval_file, steps, eval_step, opts)
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    cfg["Global"]["save_epoch_step"] = SAVE_EVERY
    if (opts or {}).get("ctc_trains"):
        assert (opts or {}).get("guide") is False, "routing needs guidance off"
        cfg["Global"]["b2n_ctc_trains"] = opts["ctc_trains"]
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False, allow_unicode=True)
    return path, plan


bv.make_cfg = make_cfg


def train(name):
    spec = bv.RUN_SPEC[name]
    opts = {k: spec[k] for k in ("seed", "guide", "lr_mult", "focal", "ctc_trains") if k in spec}
    d = bv.train(name, spec["geom"], spec["pool40"], f"train_{name}", what=spec["what"], opts=opts)
    clear_resume_setting(d)
    return d


def clear_resume_setting(d):
    """PaddleOCR saves the merged config into the run folder. After a resumed run it contains
    Global.checkpoints=<run>/latest, and every later reading with that config would try to load the deleted resume
    file instead of the epoch checkpoint. Reset it (found by the stop-and-restart code check on 12 Nov)."""
    import yaml
    path = d / "config.yml"
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    if cfg["Global"].get("checkpoints"):
        cfg["Global"]["checkpoints"] = None
        yaml.safe_dump(cfg, open(path, "w"), sort_keys=False, allow_unicode=True)
        log(f"{d.name}: cleared the resume setting from config.yml before scoring")


# ------------------------------------------------------------------ scoring (one reading pass per checkpoint)
def combo_dir():
    d = bv.DATA / "flat" / "epoch_combo"
    d.mkdir(parents=True, exist_ok=True)
    info = bv.read_json(bv.DATA / "line_info.json")
    items = {}
    for set_name, key in EP_SETS.items():
        for p, t in bv.load_set(key):
            f = p.replace("/", "__")
            if not (d / f).exists():
                shutil.copy(bv.ROOT / p, d / f)
            items[f] = (set_name, t, info[p]["cluster"] if p in info else "unknown")
    for f, src, t, writing in fresh_items():
        g = "fresh__" + f
        if not (d / g).exists():
            shutil.copy(src, d / g)
        items[g] = ("fresh boards", t, f"fresh:{writing}")
    return d, items


def score_all(model, weights, cfg, out_dir):
    meta = dict(weights=bv.sha(str(weights) + ".pdparams"), config=bv.sha(cfg))
    if (out_dir / "rows.json").exists() and (out_dir / "meta.json").exists() and bv.read_json(out_dir / "meta.json") == meta:
        return bv.read_json(out_dir / "rows.json")
    d, items = combo_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    res = out_dir / "infer.txt"; res.unlink(missing_ok=True)
    bv.run([sys.executable, "tools/infer_rec.py", "-c", cfg, "-o", f"Global.pretrained_model={weights}",
            f"Global.infer_img={d}", f"Global.save_res_path={res}", f"Global.use_gpu={not bv.CPU}"],
           out_dir / "infer_log.txt", re.compile(r"Traceback|Error"))
    pred = {}
    for line in open(res, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        pred[Path(parts[0]).name] = parts[1] if len(parts) > 1 else ""
    if set(pred) != set(items):
        raise SystemExit(f"{model}: predictions do not match the images ({len(pred)} vs {len(items)})")
    rows = []
    for f, (s, t, cl) in items.items():
        a, b = bv.norm(t), bv.norm(pred[f])
        sub, dele, ins = ops(a, b)
        rows.append(dict(model=model, set=s, file=f, cluster=cl, label=t, pred=pred[f], err=sub + dele + ins,
                         chars=max(len(a), 1), cer=(sub + dele + ins) / max(len(a), 1), sub=sub, dele=dele, ins=ins))
    bv.write_json(out_dir / "rows.json", rows); bv.write_json(out_dir / "meta.json", meta)
    return rows


def ops(a, b):
    """Substitutions, deletions, insertions of reading b against label a (one optimal alignment)."""
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): D[i][0] = i
    for j in range(m + 1): D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    i, j, s, d, ins = n, m, 0, 0, 0
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            s += a[i - 1] != b[j - 1]; i -= 1; j -= 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            d += 1; i -= 1
        else:
            ins += 1; j -= 1
    return s, d, ins


def score_run(name, d):
    cfg = d / "config.yml"
    out = {}
    for e in SCORE_EPOCHS:
        w, od = d / f"iter_epoch_{e}", d / "epochs" / f"e{e:02d}"
        if Path(str(w) + ".pdparams").is_file():
            out[e] = score_all(f"{name}@{e}", w, cfg, od)
        elif (od / "rows.json").is_file():                       # scored earlier; weights already tidied away
            out[e] = bv.read_json(od / "rows.json")
    out["best"] = score_all(f"{name}@best", d / "best_accuracy", cfg, d / "epochs" / "best")
    full = bv.score_checkpoint(name, d / "best_accuracy", cfg, d / "scores")      # the usual sets, the usual scorer
    a = {(r["set"], r["file"]): r["pred"] for r in out["best"] if r["set"] in EP_SETS}
    b = {(r["set"], r["file"]): r["pred"] for r in full if r["set"] in EP_SETS}
    same = sum(a[k] == b.get(k) for k in a) / max(len(a), 1)
    if same < 0.999:
        raise SystemExit(f"{name}: one-pass scoring differs from the usual scorer ({same:.1%} identical)")
    log(f"{name}: scored epochs {[e for e in out if e != 'best']} and best; one-pass reading = usual scorer ({same:.0%})")
    return out, full


def tidy(name, d):
    """Optimizer copies of the epoch checkpoints are never needed; epoch weights are kept only for the diagnosis."""
    freed = 0
    for p in d.glob("iter_epoch_*"):
        e = int(re.search(r"iter_epoch_(\d+)", p.name).group(1))
        keep = p.suffix == ".pdparams" and e in DIAG_EPOCHS
        if not keep and p.is_file():
            freed += p.stat().st_size; p.unlink()
    log(f"{name}: removed epoch optimizer copies and unneeded epoch weights ({freed / 1e9:.1f} GB)")


# ------------------------------------------------------------------ layer diagnosis
def diagnose(epoch):
    import b2n_routing_diag as rd
    out = bv.RESULTS / f"diag_e{epoch:02d}"
    if (out / "REPORT.md").is_file():
        return out
    found = {}
    for r in FAIR:
        w = bv.RUNS / r / f"iter_epoch_{epoch}"
        if not Path(str(w) + ".pdparams").is_file():
            log(f"diagnosis at epoch {epoch}: {r} has no checkpoint, skipped"); return None
        stub = out / "readers" / r
        stub.mkdir(parents=True, exist_ok=True)
        (stub / "best_accuracy.pdparams").unlink(missing_ok=True)
        (stub / "best_accuracy.pdparams").symlink_to(Path(str(w) + ".pdparams"))
        shutil.copy(bv.RUNS / r / "config.yml", stub / "config.yml")
        scores = stub / "scores"; scores.mkdir(exist_ok=True)
        rows = bv.read_json(bv.RUNS / r / "epochs" / f"e{epoch:02d}" / "rows.json")
        with open(scores / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=["set", "file", "pred"]); wr.writeheader()
            for x in rows:
                wr.writerow(dict(set=x["set"], file=x["file"], pred=x["pred"]))
        found[r] = stub
    info = rd.check_runs(found)
    sets = {s: [[p.replace("/", "__"), str(bv.ROOT / p), t] for p, t in bv.load_set(k)] for s, k in EP_SETS.items()}
    sets["fresh boards"] = [["fresh__" + f, str(src), t] for f, src, t, _ in fresh_items()]
    parts = "heads,weights,features,gradients" if epoch == max(DIAG_EPOCHS) else "weights,features,gradients"
    job = dict(out=str(out), device="cpu" if bv.CPU else "gpu", parts=parts.split(","), sets=sets,
               weights={"pretrained": str(bv.PRETRAINED), **{r: str(found[r] / "best_accuracy") for r in FAIR}},
               configs={r: str(found[r] / "config.yml") for r in FAIR})
    bv.write_json(out / "job.json", job)
    bv.run([sys.executable, "-u", str(HERE / "b2n_routing_diag.py"), "worker", str(out / "job.json")],
           logfile=out / "worker_log.txt", show=re.compile(r"^\[\d\d:\d\d:\d\d\]|Traceback|Error|WORKER"),
           env={"B2N_POOL40": "0", "PYTHONPATH": str(bv.REPO)})
    rd.summarise(out, sets, found, info)
    md = (out / "REPORT.md").read_text(encoding="utf-8").replace(
        "# Where do the CTC and NRTR losses act? (server readers, no training)",
        f"# Where do the CTC and NRTR losses act? Checkpoints at epoch {epoch} of every run (same training length)")
    (out / "REPORT.md").write_text(md, encoding="utf-8")
    return out


# ------------------------------------------------------------------ report
def seed_mean(rows_list):
    by = [{(r["set"], r["file"]): r for r in rows} for rows in rows_list]
    keys = set(by[0])
    assert all(set(b) == keys for b in by), "seeds scored on different crops"
    out = []
    for k, r in by[0].items():
        row = dict(r)
        for f in ("err", "chars", "cer", "sub", "dele", "ins"):
            row[f] = sum(float(b[k][f]) for b in by) / len(by)
        out.append(row)
    return out


def recipe_rows(scored, recipe, when):
    rr = [scored[m][when] for m in RECIPES[recipe] if m in scored and when in scored[m]]
    return seed_mean(rr) if len(rr) == len(RECIPES[recipe]) else None


def per100(rows, field):
    return 100 * sum(float(r[field]) for r in rows) / sum(float(r["chars"]) for r in rows)


def report(scored, lists):
    sets = ["whiteboard val", "new writers", "fresh boards"]
    L = ["# Fair guidance test and gradient routing (server reader)", "",
         "Every run saved a checkpoint every 5 epochs and each one was scored on whiteboard validation (familiar), "
         "new writers and fresh boards. Recipe numbers are the mean of 2 seeds. Mean line CER (%), lower is better.", "",
         "| Recipe | What |", "|---|---|"] + [f"| {k} | {v} |" for k, v in WHAT.items()] + [""]
    epochs = sorted({e for m in scored.values() for e in m if e != "best"})
    for s in sets:
        L += [f"## {s}: CER by epoch", "", "| Recipe | " + " | ".join(f"e{e}" for e in epochs) + " | best (val-chosen) |",
              "|---|" + "---|" * (len(epochs) + 1)]
        for rc in RECIPES:
            cells = []
            for e in epochs + ["best"]:
                rows = recipe_rows(scored, rc, e)
                x = [r for r in rows if r["set"] == s] if rows else []
                cells.append(f"{100 * sum(float(r['cer']) for r in x) / len(x):.1f}" if x else "-")
            L.append(f"| {rc} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Same-epoch differences, A - B in points [95% interval resampling photos / writers / writings]", ""]
    comp = {}
    for a, b, what in COMPARE:
        L += [f"**{a} - {b}**: {what}", "", "| When | " + " | ".join(sets) + " |", "|---|" + "---|" * len(sets)]
        for e in epochs + ["best"]:
            ra, rb = recipe_rows(scored, a, e), recipe_rows(scored, b, e)
            if not ra or not rb:
                continue
            cells = []
            for s in sets:
                xa = [r for r in ra if r["set"] == s]; xb = [r for r in rb if r["set"] == s]
                if not xa:
                    cells.append("-"); continue
                dlt, lo, hi, *_ = bv.bootstrap(xa, xb)
                comp[f"{a}-{b}|{e}|{s}"] = dict(diff=dlt, lo=lo, hi=hi)
                mark = " better" if hi < 0 else (" worse" if lo > 0 else "")
                cells.append(f"{100 * dlt:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]{mark}")
            L.append(f"| {'e' + str(e) if e != 'best' else 'best'} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Error types (per 100 reference characters): substitutions / deletions / insertions", "",
          "| Recipe | When | " + " | ".join(sets) + " |", "|---|---|" + "---|" * len(sets)]
    for rc in RECIPES:
        for e in [x for x in (15, 40, "best") if x in epochs or x == "best"]:
            rows = recipe_rows(scored, rc, e)
            if not rows:
                continue
            cells = []
            for s in sets:
                x = [r for r in rows if r["set"] == s]
                cells.append(f"{per100(x, 'sub'):.1f} / {per100(x, 'dele'):.1f} / {per100(x, 'ins'):.1f}" if x else "-")
            L.append(f"| {rc} | {'e' + str(e) if e != 'best' else 'best'} | " + " | ".join(cells) + " |")
    L += ["", "## Training", "", "| Run | Best epoch (validation) | Minutes |", "|---|---|---|"]
    for r in ORDER:
        dn = bv.RUNS / r / "DONE"
        if dn.exists():
            j = bv.read_json(dn)
            be = re.findall(r"best_epoch: (\d+)", j.get("best_metric_line", ""))
            L.append(f"| {r} | {be[-1] if be else '?'} | {j['minutes']} |")
    L += ["", f"Lists: {lists}.", "",
          "Reading the results: the fair test is G0 - S0 at the SAME epoch. Guidance off only 'really helps' unfamiliar "
          "handwriting if new writers / fresh boards are better at matched epochs, in both seeds' average, with an "
          "interval below 0. The layer diagnosis (diag_e15/, diag_e40/) says which parts the CTC loss pushes. RE and RL "
          "test whether sending the CTC loss only to early or only to late stages keeps (or beats) G0's result."]
    bv.RESULTS.mkdir(parents=True, exist_ok=True)
    (bv.RESULTS / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    bv.write_json(bv.RESULTS / "summary.json", dict(comparisons=comp, lists=lists))
    print("\n".join(L), flush=True)


def pack():
    z = bv.WORK / f"results_{bv.EXPERIMENT}.zip"
    tmp = z.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in bv.RESULTS.rglob("*"):
            if p.is_file() and not p.name.endswith(".pdparams"):
                zf.write(p, Path("results") / p.relative_to(bv.RESULTS))
        for p in bv.RUNS.rglob("*"):
            if p.is_file() and (p.suffix in (".txt", ".yml", ".csv", ".json") or p.name == "DONE") and "flat" not in p.parts:
                zf.write(p, Path("runs") / p.relative_to(bv.RUNS))
        if (bv.WORK / "experiment.json").exists():
            zf.write(bv.WORK / "experiment.json", "experiment.json")
        for f in (Path(__file__), Path(bv.__file__), HERE / "b2n_routing_patch.py", HERE / "b2n_routing_diag.py"):
            zf.write(f, f"source/{f.name}")
    tmp.replace(z)
    shutil.copy(z, HERE / z.name)
    log(f"results so far: {HERE / z.name} ({z.stat().st_size / 1e6:.0f} MB)")


def routing_test():
    """Before any routed run: one real training batch, routed gradient == CTC (allowed parts) + NRTR, exactly."""
    d = bv.WORK / "routing_test"
    if (d / "PASSED").exists():
        return
    cfg, _ = make_cfg(d, "h64", "train_G0", opts=dict(guide=False, seed=1, ctc_trains="stem,1,2"))
    bv.run([sys.executable, "-u", str(HERE / "test_routing_grad.py"), str(bv.REPO), str(cfg), str(bv.PRETRAINED)],
           d / "test_log.txt", re.compile(r"ROUTING|max_rel_error|Traceback|Error"), env={"B2N_POOL40": "0"})
    (d / "PASSED").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("routing patch test passed on a real batch (routed gradient = CTC on the chosen parts + NRTR, exactly)")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["all", "check"])
    ap.add_argument("--experiment", default="routing_runs_20261112")
    ap.add_argument("--runs", default=",".join(ORDER))
    a = ap.parse_args()
    if bv.sha(bv.__file__) != SERVER_SCRIPT_SHA256:
        raise SystemExit("b2n_versions.py differs from the pinned server script")
    runs = [r for r in a.runs.split(",") if r]
    assert all(r in ORDER for r in runs), runs
    bv.set_paths(a.experiment)
    if not FAKE:
        bv.DATA.mkdir(parents=True, exist_ok=True)
        link_data()
        bv.prepare()   # the one-time scan of every image; saved, so the full run after `check` reuses it
    lists = check_lists()
    assert len(fresh_items()) == 152 or FAKE, f"fresh boards: {len(fresh_items())} crops, expected 152"
    free = shutil.disk_usage(bv.STORAGE).free / 1e9
    log(f"plan: {', '.join(runs)} | checkpoint every {SAVE_EVERY} epochs | scored at {SCORE_EPOCHS} + best | "
        f"diagnosis at epochs {DIAG_EPOCHS} | {free:.0f} GB free on the volume")
    if free < 12 and not FAKE:
        raise SystemExit(f"only {free:.0f} GB free; this run needs about 12 GB at its peak")
    if a.step == "check":
        log("CHECK OK: lists, fresh crops, disk space"); return
    changed = rp.apply(bv.REPO)
    log(f"routing patch {'applied to' if changed else 'already in'} {bv.REPO / 'tools/program.py'}")
    bv.patch()
    if not FAKE:
        ensure_pretrained()
        payload = bv.experiment_payload(runs)
        payload["routing"] = dict(script_sha256=bv.sha(Path(__file__)), patch_sha256=bv.sha(HERE / "b2n_routing_patch.py"),
                                  save_every=SAVE_EVERY, score_epochs=SCORE_EPOCHS, diag_epochs=DIAG_EPOCHS)
        bv.initialize_experiment(payload)
    bv.RUNS.mkdir(parents=True, exist_ok=True)
    if any(r in ROUTED for r in runs):
        routing_test()
    scored = {}
    for r in runs:
        d = train(r)
        scored[r], _ = score_run(r, d)
        tidy(r, d)
        report(scored, lists); pack()
        if r in FAIR and all((bv.RUNS / x / "DONE").exists() for x in FAIR):
            for e in DIAG_EPOCHS:
                if all(Path(str(bv.RUNS / x / f"iter_epoch_{e}") + ".pdparams").is_file() for x in FAIR):
                    diagnose(e)
            pack()
    report(scored, lists); pack()
    log("DONE")


if __name__ == "__main__":
    main()
