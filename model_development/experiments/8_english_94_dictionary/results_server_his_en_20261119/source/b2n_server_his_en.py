#!/usr/bin/env python3
"""Final-model recipe with a 94-character English dictionary (19 Nov 2026): S1his_en.

Same as S1his (the final model, 10 Nov) in every setting except the dictionary: en_dict.txt (digits, letters, ASCII
symbols) plus space, instead of PaddleOCR's 18,383-character ppocrv5_dict. Consequences, by construction:
  - the CTC head's last layer and the NRTR head's character layers change size, so PaddleOCR cannot load those
    pretrained weights and they start from scratch (every other layer starts from the same pretrained weights);
  - teammate lines with characters outside the 94 are left out of training (~45 files: the 'approx', 'theta',
    'eta' symbols) and of his test (lines with 'x'-times and arrow symbols), so every model is scored on the same crops.
The shared server data checks (bv.prepare) still run with the full dictionary, as in every earlier run.
For a fair comparison the 10 Nov final model (S1his, full dictionary) and the downloaded reader are scored on exactly
the same his-test and his-val crops in this experiment.

Original description of the S1his recipe:
Server reader (PP-OCRv5_server_rec) with the S1 recipe, trained on the teammate's leak-fixed 80:10:10 split.

Built on the unchanged server package b2n_versions.py (pinned SHA-256), so training, scoring and the export check are
the same code as every server run. Only the data source differs:
  train       his_data lists (75619db2, stratified 80:10:10): S1his = all train files incl. his augmented copies;
              S1his_orig (optional) = originals only
  validation  his validation (179 crops): picks the best checkpoint, as val_wb does for our runs
  test        his test (169 crops): scored once, after training
Recipe = server S1: 64x1024, width/8 training pooling, full-height reading pooling, stage-wise backbone LR
[0.1, 0.1, 0.25, 0.5, 1.0], guidance on, CTC/NRTR 1.0/1.0, Adam 1e-4, 300 warm-up steps, cosine, batch 16,
~6,000 updates, official PP-OCRv5_server_rec_pretrained start.
His pool contains our whiteboard-validation, CNN and new-writer crops, so this model is NOT scored on our sets.

Usage (through run_server_his.sh): python b2n_server_his.py all --experiment server_his_20261110 \
                                       --link-data-from mobile_official_20261110 [--runs S1his,S1his_orig]
"""
from __future__ import annotations

import argparse, json, os, random, re, sys, time, urllib.request, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2n_versions as bv  # noqa: E402

FULL_DICT = bv.DICT                              # ppocrv5_dict.txt (18,383 characters)
EN_DICT = HERE / "en_dict.txt"                     # 94 characters (+ space via use_space_char)
EN_DICT_SHA256 = "e7f8d6c05fe813714f43a975e8464b70733c1d775531512a0538639e98b1bfc8"

SERVER_SCRIPT_SHA256 = "a4f4ff156075772a99b4c9a6eaabeead3401442f4ad04f5a167cc9e26eaa19a8"
PRETRAINED_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams"
HIS_DIR = Path(os.environ.get("B2N_HIS_DIR", "/workspace/b2n_hisdata_final"))
HIS_REVISION = "75619db23eb950d86d18e87e557954ca9a48cf6f"
HIS_LISTS = {"his_train_full": "abd894a606cc9fa24c6097ff72abe883cdb62318dd84268a7212e38c0507e13c",
             "his_train_orig": "73cd8bd2c66fd5b7222f77bd336abd2d96b28ef49eeaa7b04e85c31bcfc2389d",
             "his_val_orig": "33dd43100ae348c85d31bab23beea675c4fa19c318fa1bcb60bb72a2afbc62c1",
             "his_test_orig": "c7b761c87de49347582d88794dd095a1bd0e0b3cef84363c1c9f4213ef2798cd"}
HIS_SETS = {"his val": "his_val", "his test": "his_test"}
GEOM, POSITIONS = "h64", 128
SPEC = {"S1his_en": dict(his_list="his_train_full",
                         what="final-model recipe (server S1 on the teammate's train incl. his copies), 94-character English dictionary")}
OLD_FINAL = ("server_his_20261110", "S1his")       # the 10 Nov final model, full dictionary, scored here for comparison
OPTS = dict(lr_mult=bv.LR_MULT, seed=1)          # exactly S1's training options (guidance on, no focal loss)
log = bv.log


def norm_key(p):
    """crop name of a teammate file (augmented variants -> their crop), as in his_data/prep_his_data.py"""
    b = re.sub(r"\.[A-Za-z]+$", "", p.split("/")[-1]); m = re.match(r"^(lighting|original|geometry)_", b)
    if m: b = b[m.end():]
    b = re.sub(r"_[0-9a-f]{8}$", "", b)
    return re.sub(r"_(dim|bright|blur|rotated\d*|blemish|zoomout|zoomin|scale|perspective\d*|glare|cropped|clean\d*)$", "", b)


def link_data_from(other):
    """Reuse another experiment's extracted copy of the SAME data zips (no second ~0.8 GB extraction)."""
    src = bv.STORAGE / "experiments" / other / "data" / "b2n_data"
    if not (src / "reader/manifest.csv").exists():
        raise SystemExit(f"--link-data-from {other}: no extracted data at {src}")
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


def prepare_his(runs):
    lists = HIS_DIR / "lists"
    for name, digest in HIS_LISTS.items():
        f = lists / f"{name}.txt"
        if not f.is_file() or bv.sha(f) != digest:
            raise SystemExit(f"{f}: missing or not the checked split (run his_data/prep_his_data.py; expected {digest[:12]})")
    link = bv.ROOT / "his"
    if not link.exists():
        link.symlink_to((HIS_DIR / "raw").resolve(), target_is_directory=True)
    chars = set(bv.DICT.read_text(encoding="utf-8").splitlines()) | {" "}
    out, why = {}, {}
    todo = [(f"train_{r}", SPEC[r]["his_list"]) for r in runs] + [("his_val", "his_val_orig"), ("his_test", "his_test_orig")]
    for key, name in todo:
        keep, w = [], {}
        for p, t in bv.read_labels(lists / f"{name}.txt", "his/"):
            reason = ("label longer than the limit" if len(t) > bv.MAX_TEXT - 2 else
                      f"needs more than {POSITIONS} positions" if bv.ctc_need(t) > POSITIONS else
                      "character not in the dictionary" if any(c not in chars for c in t) else None)
            if reason:
                w[reason] = w.get(reason, 0) + 1
            else:
                keep.append((p, t))
        bv.write_set(key, keep); out[key] = len(keep); why[key] = w
    for k in ("his_val", "his_test"):
        assert not ({norm_key(p) for p, _ in bv.load_set(k)} & {norm_key(p) for r in runs for p, _ in bv.load_set(f"train_{r}")}), \
            f"{k}: a crop is also in a training list"
    assert not ({norm_key(p) for p, _ in bv.load_set("his_val")} & {norm_key(p) for p, _ in bv.load_set("his_test")})
    rng = random.Random(0)
    bv.write_set("smoke_train_his", rng.sample(bv.load_set(f"train_{runs[0]}"), 48))
    bv.write_set("smoke_his_val", rng.sample(bv.load_set("his_val"), 12))
    bv.write_set("smoke_his_test", rng.sample(bv.load_set("his_test"), 6))
    info = bv.read_json(bv.DATA / "line_info.json")
    for k in ("his_val", "his_test", "smoke_his_val", "smoke_his_test"):
        for p, t in bv.load_set(k):
            if p not in info:
                h, w = bv.image_hw(bv.ROOT / p)
                key = norm_key(p); m = re.match(r"^((lecture|oct17|cnn|quiz|television|animal)_\d+)_\d+$", key)
                info[p] = dict(h=h, w=w, id=p, parent="", cluster="his:" + (m.group(1) if m else key))
    bv.write_json(bv.DATA / "line_info.json", info)
    log(f"teammate data ({HIS_REVISION[:8]}, leak-fixed 80:10:10): {out}; left out by the shared label limits: {why}")
    return dict(lines=out, left_out=why)


def smoke(runs):
    d = bv.RUNS / "smoke"
    if (d / "OK").exists():
        log("SMOKE OK: passed earlier for these exact settings; skipped"); return
    d.mkdir(parents=True, exist_ok=True)
    rd = bv.train("smoke_S1his", GEOM, False, "smoke_train_his", eval_file="smoke_his_val", steps=24, eval_step=8,
                  what="smoke", opts=OPTS)
    import yaml
    c = yaml.safe_load(open(rd / "config.yml", encoding="utf-8"))
    bb = c["Architecture"]["Backbone"]; neck = c["Architecture"]["Head"]["head_list"][0]["CTCHead"]["Neck"]
    assert bb["name"] == "PPHGNetV2_B4" and bb["lr_mult_list"] == list(bv.LR_MULT) and neck["use_guide"] is True, (bb, neck)
    assert c["Global"]["d2s_train_image_shape"] == [3, 64, 1024] and c["Eval"]["dataset"]["label_file_list"][0].endswith("smoke_his_val.txt")
    sets = {"his val": "smoke_his_val", "his test": "smoke_his_test"}
    bv.score_checkpoint("smoke_S1his", rd / "best_accuracy", rd / "config.yml", rd / "scores", sets)
    bv.export("smoke_S1his", GEOM, rd / "inference")
    bv.write_json(d / "OK", dict(at=time.strftime("%Y-%m-%d %H:%M:%S")))
    log("SMOKE OK: server S1 recipe on the teammate's data trains, validates on his validation, scores and exports")


def report(results, his, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    allrows = [r for rows in results.values() for r in rows]
    summ = bv.summarize(allrows); bl = bv.summarize(allrows, ("model", "set", "length"))
    order = [m for m in ("pretrained", "S1his_fulldict", "S1his_en") if m in results]
    L = ["# Final-model recipe with a 94-character English dictionary (S1his_en) vs the 10 Nov final model (S1his_fulldict)", "",
         "Same crops for every model: his-test and his-val lines whose labels use only the 94 characters (+ space). "
         "S1his_fulldict = the 10 Nov final model (full 18,383-character dictionary), unchanged, read here.", "",
         f"Experiment `{bv.EXPERIMENT}`. Data: TYW02/whiteboard-notes-v1 @ {HIS_REVISION[:8]}, split by "
         "his_data/prep_his_data.py (groups of the same writing never split; stratified by source and line length). "
         "Best checkpoint chosen on HIS validation; HIS test scored once. Not comparable with our-data tables "
         "(different training and evaluation data); his pool contains our evaluation crops, so no our-set scores.", "",
         "| run | what |", "|---|---|", "| pretrained | PP-OCRv5_server_rec as downloaded |"]
    L += [f"| {m} | {SPEC[m]['what']} |" for m in order if m in SPEC]
    if "S1his_fulldict" in results:
        L.append("| S1his_fulldict | the 10 Nov final model (S1his), full dictionary, read on the same crops |")
    L += ["", "| model | set | crops | mean line CER | corpus CER | exact |", "|---|---|---|---|---|---|"]
    for s in HIS_SETS:
        for m in order:
            x = summ.get((m, s))
            if x:
                L.append(f"| {m} | {s} | {x['lines']} | {x['mean_line_CER']:.4f} | {x['corpus_CER']:.4f} | {x['exact']:.3f} |")
    L += ["", "By label length (mean line CER):", "", "| set | length | " + " | ".join(order) + " |", "|---|---|" + "---|" * len(order)]
    for s in HIS_SETS:
        for b in ("1-20 chars", "21-40 chars", "over 40 chars"):
            cells = [bl.get((m, s, b)) for m in order]
            if any(cells):
                n = next(c["lines"] for c in cells if c)
                L.append(f"| {s} | {b} ({n}) | " + " | ".join(f"{c['mean_line_CER']:.3f}" if c else "" for c in cells) + " |")
    verdicts = {}
    comps = [("S1his_en", "pretrained", "fine-tuned (94 characters) vs no fine-tune")]
    if "S1his_fulldict" in results:
        comps.append(("S1his_en", "S1his_fulldict", "94-character vs full dictionary, same recipe, same crops"))
    L += ["", "| comparison | set | CER difference | 95% interval | verdict |", "|---|---|---|---|---|"]
    for a, b, what in comps:
        for s in HIS_SETS:
            ra = [r for r in results.get(a, []) if r["set"] == s]; rb = [r for r in results.get(b, []) if r["set"] == s]
            if ra and rb:
                d, lo, hi, *_ = bv.bootstrap(ra, rb)
                v = "better" if hi < 0 else ("worse" if lo > 0 else "no clear difference")
                verdicts[f"{s}: {a} vs {b}"] = dict(diff=d, lo=lo, hi=hi, verdict=v)
                L.append(f"| {a} vs {b} ({what}) | {s} | {d:+.4f} | {lo:+.4f} .. {hi:+.4f} | {v} |")
    L += ["", "## Training runs", "", "| run | lines | epochs | updates | minutes | best validation |", "|---|---|---|---|---|---|"]
    for m in order:
        if (bv.RUNS / m / "DONE").exists():
            dn = bv.read_json(bv.RUNS / m / "DONE")
            L.append(f"| {m} | {dn['plan']['lines']} | {dn['plan']['epochs']} | {dn['plan']['planned_updates']} | {dn['minutes']} | {dn['best_metric_line'][:70]} |")
    L += ["", f"Lines left out by the shared label limits (98 characters, {POSITIONS} positions): {his['left_out']}.",
          "Limits: one seed each; the two dictionaries differ in the heads' starting weights as well as in size (see source docstring)."]
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    bv.write_json(out_dir / "summary.json", dict(summary={" | ".join(k): v for k, v in summ.items()}, comparisons=verdicts, his=his))
    bv.save_rows(out_dir, allrows)
    print("\n".join(L), flush=True)


def pack():
    z = bv.WORK / f"results_{bv.EXPERIMENT}.zip"
    tmp = z.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in bv.RESULTS.rglob("*"):
            if p.is_file() and "_previous_" not in str(p) and not any(x.startswith(".") for x in p.relative_to(bv.RESULTS).parts):
                zf.write(p, Path("results") / p.relative_to(bv.RESULTS))
        for p in bv.RUNS.glob("*/*"):
            if p.is_file() and (p.suffix in (".txt", ".yml", ".csv", ".json") or p.name == "DONE"):
                zf.write(p, Path("runs") / p.relative_to(bv.RUNS))
        zf.write(bv.WORK / "experiment.json", "experiment.json")
        zf.write(Path(__file__), "source/b2n_server_his_en.py"); zf.write(Path(bv.__file__), "source/b2n_versions.py"); zf.write(EN_DICT, "source/en_dict.txt")
    tmp.replace(z)
    import shutil
    shutil.copy(z, HERE / z.name)
    log(f"results: {HERE / z.name} ({z.stat().st_size / 1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["all"])
    ap.add_argument("--experiment", default="server_his_en_20261119")
    ap.add_argument("--link-data-from", default="mobile_official_20261110")
    ap.add_argument("--runs", default="S1his_en", help="comma list from " + ",".join(SPEC))
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", a.experiment):
        ap.error("--experiment: letters, digits, _ or - only")
    runs = [r for r in a.runs.split(",") if r]
    if not runs or [r for r in runs if r not in SPEC]:
        ap.error(f"runs must come from {list(SPEC)}")
    if bv.sha(bv.__file__) != SERVER_SCRIPT_SHA256:
        raise SystemExit("b2n_versions.py differs from the pinned server script")
    if bv.sha(EN_DICT) != EN_DICT_SHA256 or len(EN_DICT.read_text(encoding="utf-8").splitlines()) != 94:
        raise SystemExit("en_dict.txt is not the checked 94-character dictionary")
    bv.set_paths(a.experiment)
    bv.patch()
    ensure_pretrained()
    payload = bv.experiment_payload(runs)
    payload["his"] = dict(revision=HIS_REVISION, lists=HIS_LISTS, spec=SPEC, opts=OPTS, geom=GEOM,
                          script_sha256=bv.sha(Path(__file__)), linked_data_from=a.link_data_from)
    bv.initialize_experiment(payload)
    bv.RUNS.mkdir(parents=True, exist_ok=True)
    link_data_from(a.link_data_from)
    bv.DICT = FULL_DICT
    bv.prepare()                                  # the server data checks and lists (unchanged, full dictionary)
    bv.DICT = EN_DICT                             # from here on: the 94-character dictionary
    his = prepare_his(runs)                       # keeps only teammate lines the 94 characters can write
    smoke(runs)
    bv.DICT = FULL_DICT                           # the downloaded reader is read with its own dictionary
    cfg, _ = bv.make_cfg(bv.RUNS / "pretrained_his", GEOM, f"train_{runs[0]}", "his_val")
    results = {"pretrained": bv.score_checkpoint("pretrained", bv.PRETRAINED, cfg, bv.RUNS / "pretrained_his", HIS_SETS)}
    old = bv.STORAGE / "experiments" / OLD_FINAL[0] / "runs" / OLD_FINAL[1]
    if Path(str(old / "best_accuracy") + ".pdparams").is_file() and (old / "config.yml").is_file():
        results["S1his_fulldict"] = bv.score_checkpoint("S1his_fulldict", old / "best_accuracy", old / "config.yml",
                                                        bv.RUNS / "S1his_fulldict_scores", HIS_SETS)
        log(f"10 Nov final model read on the same crops: his test mean line CER "
            f"{bv.summarize(results['S1his_fulldict'])[('S1his_fulldict', 'his test')]['mean_line_CER']:.4f}")
    else:
        log(f"10 Nov final model not found at {old}: compare with its reported 7.04% by hand (different crops)")
    bv.DICT = EN_DICT
    for name in runs:
        d = bv.train(name, GEOM, False, f"train_{name}", eval_file="his_val", what=SPEC[name]["what"], opts=OPTS)
        results[name] = bv.score_checkpoint(name, d / "best_accuracy", d / "config.yml", d / "scores", HIS_SETS)
        log(f"{name}: his val mean line CER {bv.summarize(results[name])[(name, 'his val')]['mean_line_CER']:.4f}")
    for name in runs:
        bv.export(name, GEOM, bv.RESULTS / "models" / f"PP-OCRv5_server_rec_board2notes_{name}")
    report(results, his, bv.RESULTS)
    pack()
    log("DONE")


if __name__ == "__main__":
    main()
