#!/usr/bin/env python3
"""C0: the missing first rung of the server ladder, trained with the unchanged server package b2n_versions.py.

  C0  V2's data (our crops: whiteboard train + long train + teammates, NO augmented copies) with PaddleOCR's
      ORIGINAL recipe (48x320 input, 40 positions). Lines that need more than 40 positions are left out, exactly as
      for C1. Same starting weights, budget (~6,000 updates), learning rate, validation and scoring as every run.

Ladder (old runs from experiment versions_2x2; each step changes ONE thing, checked by list hashes):
  pretrained -> C0   fine-tuning on our crops (original recipe)
  C0 -> C1           + our augmented copies (C1 = C0's crops + their copies; same recipe)
  C1 -> P40          input 48x320 -> 64x1024, still 40 positions (C1, P40 and P128 train on one identical list)
  P40 -> P128        training pooling 40 positions -> width/8
  P128 -> V3         + the lines that need more than 40 positions (long sentences)
Side comparison: V2 vs C0 (adapted vs original recipe without copies; several changes at once).

Checks before training: the data lists of this experiment must be byte-identical to versions_2x2's (train_C1,
train_P40, train_P128, train_V2, train_V3, val_wb, test_cnn, long_test, val_iam, export_fixture); C0's lines must be
exactly C1's non-copy lines; and the pretrained reader must give the SAME predictions on whiteboard validation as
in versions_2x2 (same reading code). Old rows come from c0_reference_predictions.csv.gz (versions_2x2 predictions).

Usage (through run_server_c0.sh):  python b2n_server_c0.py all --experiment server_c0_20261111
Data check only, no Paddle/GPU:     B2N_WORK=<scratch> python b2n_server_c0.py check-data
"""
from __future__ import annotations

import argparse, csv, gzip, io, json, os, re, sys, time, urllib.request, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2n_versions as bv  # noqa: E402

SERVER_SCRIPT_SHA256 = "a4f4ff156075772a99b4c9a6eaabeead3401442f4ad04f5a167cc9e26eaa19a8"
PRETRAINED_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams"
REF = json.loads((HERE / "c0_reference.json").read_text(encoding="utf-8"))
REF_ROWS = HERE / "c0_reference_predictions.csv.gz"
NAME = "C0"
bv.RUN_SPEC[NAME] = dict(geom="official", pool40=True, positions=40, files=["wb_train", "long_train", "teammates"],
                         what="V2 data (our crops, no augmented copies), PaddleOCR's original recipe (48x320, 40 positions)")
OLD_SETS = ["whiteboard val", "CNN test", "long sentences (held out)", "IAM val"]
LADDER = [("C0", "pretrained", "fine-tuning on our crops (original recipe)"),
          ("C1", "C0", "+ our augmented copies (original recipe)"),
          ("P40", "C1", "input 48x320 -> 64x1024, still 40 positions (same list)"),
          ("P128", "P40", "training pooling 40 -> width/8 (same list)"),
          ("V3", "P128", "+ lines needing more than 40 positions"),
          ("V2", "C0", "side: adapted vs original recipe, no copies (several changes)"),
          ("V3", "C1", "side: adapted vs original recipe, with copies (several changes)")]
log = bv.log


def link_data_from(other):
    src = bv.STORAGE / "experiments" / other / "data" / "b2n_data"
    if not (src / "reader/manifest.csv").exists():
        log(f"no extracted data at {src}: the data zips will be extracted instead (~0.8 GB)"); return
    bv.ROOT.mkdir(parents=True, exist_ok=True)
    for c in src.iterdir():
        if c.name != "his" and not (bv.ROOT / c.name).exists():
            (bv.ROOT / c.name).symlink_to(c.resolve(), target_is_directory=c.is_dir())
    log(f"data folders linked from {other}")


def ensure_pretrained():
    f = Path(str(bv.PRETRAINED) + ".pdparams")
    if not f.is_file() or f.stat().st_size == 0:
        log("downloading", PRETRAINED_URL)
        urllib.request.urlretrieve(PRETRAINED_URL, str(f) + ".part")
        os.replace(str(f) + ".part", f)
    log(f"server pretrained weights: {f} ({bv.sha(f)[:12]})")


def check_lists():
    """Same data as versions_2x2, and C0 = exactly C1's crops without their augmented copies."""
    bad = [k for k, v in REF["lists"].items() if bv.sha(bv.DATA / f"{k}.txt") != v]
    if bad:
        raise SystemExit(f"data lists differ from versions_2x2: {bad}. Not comparable; nothing trained.")
    c0, c1 = bv.load_set(f"train_{NAME}"), bv.load_set("train_C1")
    copies = {p for f in ("aug_of_train", "aug_of_long_train") for p, _ in bv.load_set(f)}
    assert len(set(c0)) == len(c0) and set(c0) <= set(c1), "C0 has a line that C1 does not have"
    extra = set(c1) - set(c0)
    assert extra <= {(p, t) for p, t in c1 if p in copies}, "C1 minus C0 is not only augmented copies"
    assert not ({p for p, _ in c0} & copies), "an augmented copy is in C0"
    info = bv.read_json(bv.DATA / "line_info.json")
    ids = {info[p]["id"] for p, _ in c0}
    orphans = sum(info[p]["parent"] not in ids for p, _ in extra)
    s = dict(C0_lines=len(c0), C1_lines=len(c1), copies_added_in_C1=len(extra),
             copies_whose_crop_is_not_in_C0=orphans)
    log(f"lists identical to versions_2x2; C0 = C1 without its augmented copies: {s}")
    return s


def old_rows():
    with gzip.open(REF_ROWS, "rt", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r)
    return by


def check_pretrained(old):
    """Same reading code as versions_2x2: the pretrained reader must read whiteboard validation identically."""
    cfg, _ = bv.make_cfg(bv.RUNS / "pretrained", "h48", "train_V3")
    new = bv.score_checkpoint("pretrained", bv.PRETRAINED, cfg, bv.RUNS / "pretrained", {"whiteboard val": "val_wb"})
    a = {r["file"]: r["pred"] for r in new}
    b = {r["file"]: r["pred"] for r in old["pretrained"] if r["set"] == "whiteboard val"}
    assert set(a) == set(b), "pretrained check: different validation crops"
    same = sum(a[f] == b[f] for f in a) / len(a)
    log(f"pretrained reader on whiteboard validation: {same:.1%} identical to versions_2x2")
    if same < 1.0:
        raise SystemExit("reading differs from versions_2x2 (pretrained predictions changed): the ladder would mix "
                         "two scorers. Nothing trained.")
    return same


def report(c0_rows, old, lists, same, out_dir):
    res = {m: [r for r in rows if r["set"] in OLD_SETS] for m, rows in old.items()}
    res[NAME] = c0_rows
    order = ["pretrained", "C0", "C1", "P40", "P128", "V3", "V2"]
    rows = [r for m in order for r in res[m] if r["set"] in OLD_SETS]
    summ = bv.summarize(rows)
    L = ["# Server ladder: pretrained -> C0 -> C1 -> P40 -> P128 -> V3", "",
         f"Experiment `{bv.EXPERIMENT}` trains only C0; the other rows are the versions_2x2 predictions "
         f"(identical data lists checked; pretrained reads whiteboard validation {same:.0%} identically, so the same "
         "scorer). One seed per model; whiteboard validation picked each checkpoint and is the primary measure.", "",
         "| model | what | " + " | ".join(OLD_SETS) + " |", "|---|---|" + "---|" * len(OLD_SETS)]
    what = {"pretrained": "PP-OCRv5_server_rec as downloaded", **{m: bv.RUN_SPEC[m]["what"] for m in order[1:]}}
    for m in order:
        L.append(f"| {m} | {what[m]} | " + " | ".join(f"{summ[(m, s)]['mean_line_CER'] * 100:.2f}" for s in OLD_SETS) + " |")
    L += ["", "Mean line CER (%). Each step, A - B in points with a 95% interval (whole photos / sentences resampled):", "",
          "| step | what changes | " + " | ".join(OLD_SETS) + " |", "|---|---|" + "---|" * len(OLD_SETS)]
    comp = {}
    for a, b, w in LADDER:
        cells = []
        for s in OLD_SETS:
            d, lo, hi, *_ = bv.bootstrap([r for r in res[a] if r["set"] == s], [r for r in res[b] if r["set"] == s])
            comp[f"{s}: {a} vs {b}"] = dict(diff=d, lo=lo, hi=hi)
            mark = " (better)" if hi < 0 else (" (worse)" if lo > 0 else "")
            cells.append(f"{d * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]{mark}")
        L.append(f"| {a} vs {b} | {w} | " + " | ".join(cells) + " |")
    nw = [r for r in c0_rows if r["set"] == "new writers"]
    if nw:
        L += ["", f"C0 on the new-writer test: {bv.summarize(nw)[(NAME, 'new writers')]['mean_line_CER'] * 100:.2f}% "
              "(the old runs were not scored on it on the GPU, so no ladder for this set)."]
    if (bv.RUNS / NAME / "DONE").exists():
        dn = bv.read_json(bv.RUNS / NAME / "DONE")
        L += ["", f"C0 training: {dn['plan']['lines']} lines, {dn['plan']['epochs']} epochs, "
              f"{dn['plan']['planned_updates']} updates, {dn['minutes']} min; best: {dn['best_metric_line'][:80]}"]
    L += ["", f"Lists: {lists}.",
          "Limits: one seed each (seed-to-seed noise on whiteboard validation was about 1 point); C0 was trained "
          "today, the others on 6 Nov with the same training code and data (repo changes since then touch only the "
          "mobile backbone)."]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    bv.write_json(out_dir / "summary.json", dict(summary={" | ".join(k): v for k, v in summ.items()}, comparisons=comp,
                                                 lists=lists, pretrained_identical=same))
    bv.save_rows(out_dir, c0_rows)
    print("\n".join(L), flush=True)


def pack():
    z = bv.WORK / f"results_{bv.EXPERIMENT}.zip"
    tmp = z.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in bv.RESULTS.rglob("*"):
            if p.is_file() and "_previous_" not in str(p):
                zf.write(p, Path("results") / p.relative_to(bv.RESULTS))
        for p in bv.RUNS.glob("*/*"):
            if p.is_file() and (p.suffix in (".txt", ".yml", ".csv", ".json") or p.name == "DONE"):
                zf.write(p, Path("runs") / p.relative_to(bv.RUNS))
        zf.write(bv.WORK / "experiment.json", "experiment.json")
        for f in (Path(__file__), Path(bv.__file__), HERE / "c0_reference.json"):
            zf.write(f, f"source/{f.name}")
    tmp.replace(z)
    import shutil
    shutil.copy(z, HERE / z.name)
    log(f"results: {HERE / z.name} ({z.stat().st_size / 1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["all", "check-data"])
    ap.add_argument("--experiment", default="server_c0_20261111")
    ap.add_argument("--link-data-from", default="mobile_official_20261110")
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", a.experiment):
        ap.error("--experiment: letters, digits, _ or - only")
    if bv.sha(bv.__file__) != SERVER_SCRIPT_SHA256:
        raise SystemExit("b2n_versions.py differs from the pinned server script")
    bv.set_paths(a.experiment)
    if a.step == "check-data":
        bv.DATA.mkdir(parents=True, exist_ok=True)
        link_data_from(a.link_data_from)
        bv.prepare()
        lists = check_lists()
        old = old_rows()
        fake = [dict(r, model=NAME) for r in old["C1"]]          # wiring test of the report only
        report(fake, old, lists, 1.0, bv.WORK / "check_report")
        log("CHECK-DATA OK (the report above uses C1's rows as a stand-in for C0)")
        return
    bv.patch()
    ensure_pretrained()
    payload = bv.experiment_payload([NAME])
    payload["c0"] = dict(script_sha256=bv.sha(Path(__file__)), reference_sha256=bv.sha(HERE / "c0_reference.json"),
                         reference_rows_sha256=bv.sha(REF_ROWS), linked_data_from=a.link_data_from)
    bv.initialize_experiment(payload)
    bv.RUNS.mkdir(parents=True, exist_ok=True)
    link_data_from(a.link_data_from)
    bv.prepare()
    lists = check_lists()
    old = old_rows()
    same = check_pretrained(old)
    spec = bv.RUN_SPEC[NAME]
    d = bv.train(NAME, spec["geom"], spec["pool40"], f"train_{NAME}", what=spec["what"])
    rows = bv.score_checkpoint(NAME, d / "best_accuracy", d / "config.yml", d / "scores")
    bv.export(NAME, spec["geom"], bv.RESULTS / "models" / f"PP-OCRv5_server_rec_board2notes_{NAME}")
    report(rows, old, lists, same, bv.RESULTS)
    pack()
    log("DONE")


if __name__ == "__main__":
    main()
