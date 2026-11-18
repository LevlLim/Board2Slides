"""Board2Notes: where in the server backbone do the CTC and NRTR losses act?  (12 Nov 2026, NO training)

Answers, for the already-trained server readers, the questions behind "route a loss to certain layers":
  1. weights   How far each part of the network (stem, stages 1-4, CTC neck, CTC head, NRTR head) moved from the
               pretrained weights, for S0 / S0s2 (guidance on) and G0 / G0s2 (guidance off), and whether the
               guidance-on vs guidance-off difference in each part is larger than the seed-to-seed difference.
  2. features  Linear CKA (0 = unrelated, 1 = same up to rotation/scale) between the stage outputs of two readers on
               the same images: in which stage do guidance-off features differ from guidance-on features, compared
               with two seeds of the same recipe?  Familiar (whiteboard val) and unfamiliar (new writers + fresh
               boards) images separately.
  3. gradients On the same fixed training batches, the CTC-loss gradient and the NRTR-loss gradient in each part:
               size, ratio and direction (cosine).  The CTC branch is measured with guidance off for every reader
               (the weights are identical; guidance only cuts the gradient), so it shows what CTC WOULD push.
               Sanity check first: with guidance on, the CTC gradient in the backbone must be exactly 0.
  4. heads     Read the same crops with the CTC head (used in the app) and with the NRTR head (the attention
               decoder, training only) of each reader.  Check first: our CTC readings must match the readings the
               original scoring saved.

Nothing is trained, exported or changed; S0/S0s2/G0/G0s2 checkpoints and data are only read.
Uses the pinned b2n_versions.py unchanged (paths, PaddleOCR patches, scorer).

  bash run_routing_diag.sh            (on the pod; results in results_routing_diag_20261112.zip)
"""
import argparse, csv, json, math, os, re, shutil, sys, time, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2n_versions as bv  # noqa: E402

EXPERIMENT = "routing_diag_20261112"
READERS = ["S0", "S0s2", "G0", "G0s2"]
RUN_HOME = {"S0": "strategies_s0s3", "S0s2": "strategies_s0s3", "G0": "guide_g0", "G0s2": "guide_g0"}
DATA_EXPERIMENT = "strategies_s0s3"          # S0 and G0 used the same lists (wb_train + long_train)
SETS = {"whiteboard val": "val_wb", "new writers": "test_newwriter"}
FRESH = HERE / "fresh_boards"                 # 152 fresh-board crops, test only (never trained on)
GRAD_BATCHES = int(os.environ.get("B2N_DIAG_BATCHES", 24))
CKA_POSITIONS = int(os.environ.get("B2N_DIAG_POSITIONS", 6000))
CKA_VAL_IMAGES = int(os.environ.get("B2N_DIAG_VAL_IMAGES", 300))
LIMIT = int(os.environ.get("B2N_DIAG_LIMIT", 0))          # test only: read at most this many crops per set
GROUPS = ["stem", "stage 1", "stage 2", "stage 3", "stage 4", "CTC neck (SVTR)", "CTC head", "NRTR head"]


def group(name):
    m = re.match(r"backbone\.stages\.(\d)\.", name)
    if m:
        return f"stage {int(m.group(1)) + 1}"
    if name.startswith("backbone.stem."):
        return "stem"
    if name.startswith("head.ctc_encoder.") or name.startswith("head.encoder_reshape."):
        return "CTC neck (SVTR)"
    if name.startswith("head.ctc_head."):
        return "CTC head"
    if name.startswith("head.gtc_head.") or name.startswith("head.before_gtc."):
        return "NRTR head"
    return None                              # unused backbone.last_conv / fc of the classification variant


def is_stat(name):
    return name.endswith("._mean") or name.endswith("._variance")


# ================================================================== worker (runs inside the PaddleOCR checkout)
def worker(job_path):
    import numpy as np, paddle, yaml
    sys.path.insert(0, os.getcwd())
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process
    from ppocr.losses import build_loss
    from ppocr.data import build_dataloader, create_operators, transform
    from ppocr.utils.logging import get_logger
    job = json.loads(Path(job_path).read_text())
    out = Path(job["out"]); out.mkdir(parents=True, exist_ok=True)
    paddle.set_device(job["device"])
    logger = get_logger()
    say = lambda *a: print(time.strftime("[%H:%M:%S]"), *a, flush=True)

    def load_cfg(path, guide=None):
        cfg = yaml.safe_load(open(path, encoding="utf-8"))
        if guide is not None:
            neck = cfg["Architecture"]["Head"]["head_list"][0]["CTCHead"]["Neck"]
            assert neck["name"] == "svtr", neck
            neck["use_guide"] = guide
        post = build_post_process(cfg["PostProcess"], cfg["Global"])
        n = len(post.character)
        cfg["Architecture"]["Head"]["out_channels_list"] = {"CTCLabelDecode": n, "SARLabelDecode": n + 2,
                                                             "NRTRLabelDecode": n + 3}
        return cfg, post

    def build(cfg, weights, strict=True):
        paddle.seed(0)
        m = build_model(cfg["Architecture"])
        sd = paddle.load(weights + ".pdparams")
        ms = m.state_dict()
        missing = [k for k in ms if k not in sd and group(k) is not None]
        bad = [k for k in ms if k in sd and list(ms[k].shape) != list(sd[k].shape)]
        if missing or bad:
            msg = f"{weights}: {len(missing)} missing, {len(bad)} wrong-shape tensors, e.g. {(missing + bad)[:3]}"
            if strict:
                raise SystemExit(msg)
            say("  NOTE (pretrained, as in training's own loading): " + msg)
        m.set_state_dict({k: v for k, v in sd.items() if k in ms and k not in bad})
        return m

    weights, cfgs = job["weights"], job["configs"]
    base_cfg = cfgs["S0"]
    norm = lambda s: " ".join(str(s).lower().split())

    def lev(a, b):
        p = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            c = [i]
            for j, cb in enumerate(b, 1):
                c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (ca != cb)))
            p = c
        return p[-1]

    # ---- 1. weight change (state dicts only)
    if "weights" in job["parts"]:
        say("part 1: weight change per part")
        sds = {k: paddle.load(v + ".pdparams") for k, v in weights.items()}
        keys = [k for k in sds["pretrained"] if group(k) and not is_stat(k)
                and all(k in s and list(s[k].shape) == list(sds["pretrained"][k].shape) for s in sds.values())]
        acc = {}
        for k in keys:
            g = group(k)
            w0 = sds["pretrained"][k].astype("float64")
            a = acc.setdefault(g, {"params": 0, "pre_sq": 0.0})
            a["params"] += int(np.prod(w0.shape)); a["pre_sq"] += float((w0 ** 2).sum())
            for r in READERS:
                a[f"d_{r}"] = a.get(f"d_{r}", 0.0) + float(((sds[r][k].astype("float64") - w0) ** 2).sum())
            for x, y in (("S0", "G0"), ("S0s2", "G0s2"), ("S0", "S0s2"), ("G0", "G0s2"), ("S0", "G0s2"), ("S0s2", "G0")):
                a[f"{x}-{y}"] = a.get(f"{x}-{y}", 0.0) + float(((sds[x][k].astype("float64") - sds[y][k].astype("float64")) ** 2).sum())
            e1 = sds["G0"][k].astype("float64") - sds["S0"][k].astype("float64")       # guidance effect, seed 1
            e2 = sds["G0s2"][k].astype("float64") - sds["S0s2"][k].astype("float64")   # guidance effect, seed 2
            a["e12"] = a.get("e12", 0.0) + float((e1 * e2).sum())
            a["e11"] = a.get("e11", 0.0) + float((e1 * e1).sum()); a["e22"] = a.get("e22", 0.0) + float((e2 * e2).sum())
        rows = []
        for g in GROUPS:
            a = acc[g]; base = math.sqrt(a["pre_sq"])
            row = dict(part=g, params=a["params"], pretrained_norm=base)
            for r in READERS:
                row[f"moved_{r}"] = math.sqrt(a[f"d_{r}"]) / base
            for p in ("S0-G0", "S0s2-G0s2", "S0-S0s2", "G0-G0s2", "S0-G0s2", "S0s2-G0"):
                row[f"dist_{p}"] = math.sqrt(a[p]) / base
            same_seed = (row["dist_S0-G0"] + row["dist_S0s2-G0s2"]) / 2
            seeds = (row["dist_S0-S0s2"] + row["dist_G0-G0s2"]) / 2
            row["guidance_over_seed"] = same_seed / seeds if seeds else float("nan")
            row["guidance_direction_cos"] = a["e12"] / math.sqrt(a["e11"] * a["e22"]) if a["e11"] and a["e22"] else float("nan")
            rows.append(row)
        json.dump(rows, open(out / "weights.json", "w"), indent=1)
        del sds

    # ---- shared: crops for reading / features
    ops_cache = {}

    def infer_ops(cfg_path):
        if cfg_path in ops_cache:
            return ops_cache[cfg_path]
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        tr = []
        for op in cfg["Eval"]["dataset"]["transforms"]:
            name = list(op)[0]
            if "Label" in name:
                continue
            if name == "RecResizeImg":
                op[name]["infer_mode"] = True
            if name == "KeepKeys":
                op[name]["keep_keys"] = ["image"]
            tr.append(op)
        g = dict(cfg["Global"]); g["infer_mode"] = True
        ops_cache[cfg_path] = create_operators(tr, g)
        return ops_cache[cfg_path]

    def tensor(path, ops):
        with open(path, "rb") as f:
            b = transform({"image": f.read()}, ops)
        return paddle.to_tensor(np.expand_dims(b[0], 0))

    sets = job["sets"]       # {set: [[file_key, abs_path, label], ...]}

    # ---- 4. heads (first, because it also validates the reading path)
    if "heads" in job["parts"]:
        say("part 4: CTC head vs NRTR head readings")
        rows = []
        for r in READERS:
            cfg, post = load_cfg(cfgs[r])
            nrtr_post = build_post_process({"name": "NRTRLabelDecode"}, cfg["Global"])
            m = build(cfg, weights[r]); m.eval()
            ops = infer_ops(cfgs[r])
            for s, items in sets.items():
                for key, path, label in items:
                    with paddle.no_grad():
                        x = tensor(path, ops)
                        f = m.backbone(x)
                        ctc = post(m.head(f))[0][0]
                        nr = nrtr_post(m.head.gtc_head(m.head.before_gtc(f)))[0][0]
                    a = norm(label)
                    rows.append(dict(model=r, set=s, file=key, label=label, ctc=ctc, nrtr=nr,
                                     ctc_err=lev(a, norm(ctc)), nrtr_err=lev(a, norm(nr)), chars=max(len(a), 1)))
            say(f"  {r}: read {sum(len(v) for v in sets.values())} crops with both heads")
            del m
        with open(out / "heads.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---- 2. features (linear CKA per stage)
    if "features" in job["parts"]:
        say("part 2: stage features (linear CKA)")
        rng = np.random.default_rng(0)
        groups_img = {"familiar": [p for s in ("whiteboard val",) for _, p, _ in sets.get(s, [])][:CKA_VAL_IMAGES],
                      "unfamiliar": [p for s in ("new writers", "fresh boards") for _, p, _ in sets.get(s, [])]}
        feats, idx = {}, {}
        for r in ["pretrained"] + READERS:
            cfg, _ = load_cfg(cfgs.get(r, base_cfg))
            m = build(cfg, weights[r], strict=r != "pretrained"); m.eval()
            ops = infer_ops(cfgs.get(r, base_cfg))
            for gname, paths in groups_img.items():
                per_stage = [[] for _ in range(5)]
                for p in paths:
                    with paddle.no_grad():
                        x = m.backbone.stem(tensor(p, ops))
                        outs = [x]
                        for st in m.backbone.stages:
                            x = st(x); outs.append(x)
                    for i, o in enumerate(outs):
                        per_stage[i].append(o.mean(axis=2)[0].transpose([1, 0]).numpy())   # positions x channels
                for i in range(5):
                    a = np.concatenate(per_stage[i], 0)
                    k = (gname, i)
                    if k not in idx:
                        idx[k] = np.sort(rng.choice(len(a), min(CKA_POSITIONS, len(a)), replace=False))
                    feats[(r, gname, i)] = a[idx[k]].astype(np.float64)
            say(f"  {r}: features collected"); del m

        def cka(x, y):
            x = x - x.mean(0); y = y - y.mean(0)
            xy = np.linalg.norm(x.T @ y) ** 2
            return float(xy / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))
        rows = []
        pairs = [("S0", "G0"), ("S0s2", "G0s2"), ("S0", "S0s2"), ("G0", "G0s2"), ("pretrained", "S0"), ("pretrained", "G0")]
        for gname in groups_img:
            for i, sname in enumerate(["stem", "stage 1", "stage 2", "stage 3", "stage 4"]):
                row = dict(images=gname, stage=sname, positions=int(feats[("S0", gname, i)].shape[0]),
                           channels=int(feats[("S0", gname, i)].shape[1]))
                for x, y in pairs:
                    row[f"{x}~{y}"] = cka(feats[(x, gname, i)], feats[(y, gname, i)])
                sh = feats[("G0", gname, i)][np.random.default_rng(1).permutation(len(feats[("G0", gname, i)]))]
                row["control_S0~shuffled_G0"] = cka(feats[("S0", gname, i)], sh)   # positions mismatched: must be low
                rows.append(row)
        json.dump(rows, open(out / "features.json", "w"), indent=1)
        del feats

    # ---- 3. gradients
    if "gradients" in job["parts"]:
        say("part 3: CTC vs NRTR gradients on fixed training batches")
        cfg, _ = load_cfg(base_cfg, guide=False)
        tcfg = json.loads(json.dumps(cfg))
        tcfg["Train"]["loader"].update(num_workers=0)
        loader = build_dataloader(tcfg, "Train", paddle.get_device(), logger, seed=0)
        batches = []
        for b in loader:
            batches.append([t.cpu() if hasattr(t, "cpu") else t for t in b])
            if len(batches) >= GRAD_BATCHES:
                break
        say(f"  cached {len(batches)} training batches of {batches[0][0].shape}")
        loss_fn = build_loss(cfg["Loss"])
        dev = lambda b: [paddle.to_tensor(t.numpy()) if hasattr(t, "numpy") else t for t in b]

        def grads(m):
            g = {}
            for n, p in m.named_parameters():
                k = group(n)
                if k and p.grad is not None:
                    g.setdefault(k, []).append((n, p.grad.astype("float64").flatten()))
            return g

        def clear(m):
            for p in m.parameters():
                p.clear_gradient(False)

        # sanity: guidance on -> CTC gradient in the backbone exactly 0
        gcfg, _ = load_cfg(base_cfg, guide=True)
        m = build(gcfg, weights["S0"]); m.train(); clear(m)
        b = dev(batches[0]); paddle.seed(1000)
        L = loss_fn(m(b[0], data=b[1:]), b)
        L["CTCLoss"].backward()
        bb = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters()
                 if n.startswith("backbone.") and p.grad is not None)
        neck = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters()
                   if n.startswith("head.ctc_encoder.") and p.grad is not None)
        assert bb == 0.0 and neck > 0, (bb, neck)
        sanity = dict(guided_ctc_backbone_grad=bb, guided_ctc_neck_grad=neck)
        del m
        rows = []
        for r in ["pretrained"] + READERS:
            m = build(cfg, weights[r], strict=r != "pretrained"); m.train()
            acc = {g: dict(ctc_sq=[], nrtr_sq=[], cos=[]) for g in GROUPS}
            sums = {}
            losses = []
            for bi, b in enumerate(batches):
                b = dev(b); clear(m); paddle.seed(1000 + bi)
                L = loss_fn(m(b[0], data=b[1:]), b)
                losses.append((float(L["CTCLoss"]), float(L["NRTRLoss"])))
                L["CTCLoss"].backward(retain_graph=True)
                gc = grads(m); clear(m)
                L["NRTRLoss"].backward()
                gn = grads(m); clear(m)
                for g in GROUPS:
                    c = paddle.concat([t for _, t in gc.get(g, [])]) if gc.get(g) else None
                    n_ = paddle.concat([t for _, t in gn.get(g, [])]) if gn.get(g) else None
                    cs = float((c * c).sum()) if c is not None else 0.0
                    ns = float((n_ * n_).sum()) if n_ is not None else 0.0
                    acc[g]["ctc_sq"].append(cs); acc[g]["nrtr_sq"].append(ns)
                    if c is not None and n_ is not None and cs > 0 and ns > 0:
                        acc[g]["cos"].append(float((c * n_).sum()) / math.sqrt(cs * ns))
                    for tag, v in (("ctc", c), ("nrtr", n_)):
                        if v is not None:
                            sums[(g, tag)] = v if (g, tag) not in sums else sums[(g, tag)] + v
            for g in GROUPS:
                a = acc[g]
                ctc = float(np.mean(np.sqrt(a["ctc_sq"]))); nr = float(np.mean(np.sqrt(a["nrtr_sq"])))
                sc, sn = sums.get((g, "ctc")), sums.get((g, "nrtr"))
                rows.append(dict(model=r, part=g, ctc_grad=ctc, nrtr_grad=nr,
                                 ctc_share=ctc / (ctc + nr) if ctc + nr else float("nan"),
                                 cos_per_batch=float(np.mean(a["cos"])) if a["cos"] else float("nan"),
                                 cos_summed=(float((sc * sn).sum()) / math.sqrt(float((sc * sc).sum()) * float((sn * sn).sum())))
                                 if sc is not None and sn is not None else float("nan"),
                                 ctc_consistency=(math.sqrt(float((sc * sc).sum())) / (len(batches) * ctc)) if sc is not None and ctc else float("nan"),
                                 nrtr_consistency=(math.sqrt(float((sn * sn).sum())) / (len(batches) * nr)) if sn is not None and nr else float("nan"),
                                 ctc_loss=float(np.mean([l[0] for l in losses])), nrtr_loss=float(np.mean([l[1] for l in losses]))))
            say(f"  {r}: gradients measured on {len(batches)} batches"); del m
        json.dump(dict(sanity=sanity, rows=rows, batches=len(batches), batch_shape=list(batches[0][0].shape)),
                  open(out / "gradients.json", "w"), indent=1)
    say("WORKER DONE")


# ================================================================== controller
def find_runs():
    found, problems = {}, []
    for r in READERS:
        d = Path(os.environ.get(f"B2N_RUN_{r}", bv.STORAGE / "experiments" / RUN_HOME[r] / "runs" / r))
        if not (d / "best_accuracy.pdparams").is_file() or not (d / "config.yml").is_file():
            problems.append(f"{r}: no best_accuracy.pdparams + config.yml in {d}")
        found[r] = d
    pre = Path(str(bv.PRETRAINED) + ".pdparams")
    if not pre.is_file():
        problems.append(f"pretrained weights missing: {pre}")
    if problems:
        raise SystemExit("Cannot run, these files are not on the volume:\n  " + "\n  ".join(problems))
    return found


def check_runs(found):
    """The four readers must be the runs we scored: same train list (S0 = G0 data), use_guide as expected."""
    import yaml
    info = {}
    for r, d in found.items():
        cfg = yaml.safe_load(open(d / "config.yml", encoding="utf-8"))
        guide = cfg["Architecture"]["Head"]["head_list"][0]["CTCHead"]["Neck"]["use_guide"]
        lst = cfg["Train"]["dataset"]["label_file_list"][0]
        info[r] = dict(dir=str(d), use_guide=guide, train_list=lst, train_list_sha=bv.sha(lst),
                       checkpoint_sha=bv.sha(d / "best_accuracy.pdparams"), seed=cfg["Global"]["seed"])
    assert info["S0"]["use_guide"] is True and info["S0s2"]["use_guide"] is True, info
    assert info["G0"]["use_guide"] is False and info["G0s2"]["use_guide"] is False, info
    # seed 2 uses the same lines in a different order, so compare S0 with G0 and S0s2 with G0s2
    assert info["S0"]["train_list_sha"] == info["G0"]["train_list_sha"], "S0 and G0 train lists differ"
    assert info["S0s2"]["train_list_sha"] == info["G0s2"]["train_list_sha"], "S0s2 and G0s2 train lists differ"
    same_lines = lambda a, b: sorted(open(a, encoding="utf-8").read().splitlines()) == sorted(open(b, encoding="utf-8").read().splitlines())
    assert same_lines(info["S0"]["train_list"], info["S0s2"]["train_list"]), "seed 1 and seed 2 train on different lines"
    assert info["S0"]["seed"] == info["G0"]["seed"] and info["S0s2"]["seed"] == info["G0s2"]["seed"], info
    bv.log("  readers found: " + ", ".join(f"{r} (guide {v['use_guide']}, seed {v['seed']}, ckpt {v['checkpoint_sha'][:12]})"
                                           for r, v in info.items()))
    return info


def gather_sets():
    bv.set_paths(DATA_EXPERIMENT)
    sets = {}
    for s, key in SETS.items():
        f = bv.DATA / f"{key}.txt"
        if not f.is_file():
            raise SystemExit(f"missing list {f}")
        sets[s] = [[p.replace("/", "__"), str(bv.ROOT / p), t] for p, t in bv.load_set(key)]
        miss = [x for x in sets[s] if not Path(x[1]).is_file()]
        if miss:
            raise SystemExit(f"{s}: {len(miss)} images missing, e.g. {miss[0][1]}")
    lab = FRESH / "labels.tsv"
    if lab.is_file():
        sets["fresh boards"] = [[Path(p).name, str(FRESH / p), t] for p, t in bv.read_labels(lab)]
    if LIMIT:
        sets = {k: v[:LIMIT] for k, v in sets.items()}
    bv.log("  sets: " + ", ".join(f"{s} {len(v)}" for s, v in sets.items()))
    return sets


def saved_ctc(found):
    """Readings the original scoring saved (tools/infer_rec.py), to check our reading path against."""
    out = {}
    for r, d in found.items():
        f = d / "scores" / "predictions.csv"
        if f.is_file():
            for row in csv.DictReader(open(f, encoding="utf-8")):
                out[(r, row["set"], row["file"])] = row["pred"]
    return out


def summarise(out, sets, found, info):
    import statistics as st
    desc = ", ".join("%s (guidance %s, seed %s)" % (r, "on" if info[r]["use_guide"] else "off", info[r]["seed"]) for r in READERS)
    lines = ["# Where do the CTC and NRTR losses act? (server readers, no training)", "",
             f"Readers: {desc}.",
             "Same training data for all four. Pretrained = PP-OCRv5_server_rec before fine-tuning.", ""]
    res = {}
    if (out / "heads.csv").is_file():
        rows = list(csv.DictReader(open(out / "heads.csv", encoding="utf-8")))
        saved = saved_ctc(found)
        same = [row["ctc"] == saved[(row["model"], row["set"], row["file"])] for row in rows
                if (row["model"], row["set"], row["file"]) in saved]
        res["reading_check"] = dict(compared=len(same), identical=sum(same))
        lines += ["## 4. CTC head vs NRTR head (line CER %, lower is better)", "",
                  f"Reading check: {sum(same)} of {len(same)} CTC readings identical to the saved scoring readings.", "",
                  "| Reader | Set | CTC head | NRTR head |", "|---|---|---|---|"]
        tab = {}
        for row in rows:
            t = tab.setdefault((row["model"], row["set"]), [[], []])
            t[0].append(int(row["ctc_err"]) / int(row["chars"])); t[1].append(int(row["nrtr_err"]) / int(row["chars"]))
        for (r, s), (c, n) in tab.items():
            lines.append(f"| {r} | {s} | {100 * st.mean(c):.1f} | {100 * st.mean(n):.1f} |")
        res["heads"] = {f"{r}|{s}": dict(ctc=100 * st.mean(c), nrtr=100 * st.mean(n), n=len(c)) for (r, s), (c, n) in tab.items()}
        lines.append("")
        if same and sum(same) < 0.95 * len(same):
            lines.append("**WARNING: the reading path does not reproduce the saved readings; do not use part 4.**\n")
    if (out / "weights.json").is_file():
        rows = json.load(open(out / "weights.json"))
        res["weights"] = rows
        lines += ["## 1. How far each part moved from the pretrained weights (relative change, %)", "",
                  "guidance/seed = distance between guidance-on and guidance-off readers of the same seed, divided by the "
                  "distance between two seeds of the same recipe. About 1 = guidance changed this part no more than a "
                  "different seed does; clearly above 1 = guidance changed this part.",
                  "same direction = cosine between the guidance effect in seed 1 (G0 - S0) and in seed 2 (G0s2 - S0s2): "
                  "about 0 = the two seeds changed this part in unrelated ways (noise); clearly above 0 = turning guidance "
                  "off pushes this part the same way every time.", "",
                  "| Part | Params | S0 | S0s2 | G0 | G0s2 | guidance/seed | same direction |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['part']} | {r['params'] / 1e6:.2f} M | " + " | ".join(f"{100 * r[f'moved_{x}']:.2f}" for x in READERS)
                         + f" | {r['guidance_over_seed']:.2f} | {r['guidance_direction_cos']:.3f} |")
        lines.append("")
    if (out / "features.json").is_file():
        rows = json.load(open(out / "features.json"))
        res["features"] = rows
        lines += ["## 2. Stage features: similarity between readers (linear CKA, 1 = same)", "",
                  "Guidance effect = S0~G0 and S0s2~G0s2 (same seed). Seed noise = S0~S0s2 and G0~G0s2. If guidance "
                  "changed a stage, its guidance similarity is clearly lower than its seed similarity.", "",
                  "Control = S0 against G0 with the positions shuffled; it must be far below the others, or CKA is not "
                  "measuring anything.", "",
                  "| Images | Stage | S0~G0 | S0s2~G0s2 | S0~S0s2 | G0~G0s2 | pre~S0 | pre~G0 | control |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['images']} | {r['stage']} | " + " | ".join(f"{r[k]:.4f}" for k in
                         ("S0~G0", "S0s2~G0s2", "S0~S0s2", "G0~G0s2", "pretrained~S0", "pretrained~G0", "control_S0~shuffled_G0")) + " |")
        lines.append("")
    if (out / "gradients.json").is_file():
        g = json.load(open(out / "gradients.json"))
        res["gradients"] = g
        lines += ["## 3. CTC vs NRTR gradient per part (same fixed training batches)", "",
                  f"Sanity check: with guidance on, CTC gradient in the backbone = {g['sanity']['guided_ctc_backbone_grad']} "
                  f"(must be 0), in the CTC neck = {g['sanity']['guided_ctc_neck_grad']:.3g} (> 0).",
                  f"{g['batches']} batches of shape {g['batch_shape']}. CTC measured with guidance off for every reader.",
                  "CTC share = CTC gradient size / (CTC + NRTR). cos = direction agreement (+1 same way, 0 unrelated, "
                  "-1 opposite); per batch and for the gradient summed over all batches.", "",
                  "| Reader | Part | CTC grad | NRTR grad | CTC share | cos per batch | cos summed |", "|---|---|---|---|---|---|---|"]
        for r in g["rows"]:
            lines.append(f"| {r['model']} | {r['part']} | {r['ctc_grad']:.3g} | {r['nrtr_grad']:.3g} | {r['ctc_share']:.2f} | "
                         f"{r['cos_per_batch']:.2f} | {r['cos_summed']:.2f} |")
        lines.append("")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    bv.write_json(out / "summary.json", dict(readers=info, **res))
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["all", "check", "worker"])
    ap.add_argument("job", nargs="?")
    ap.add_argument("--parts", default="heads,weights,features,gradients")
    a = ap.parse_args()
    if a.step == "worker":
        worker(a.job); return
    found = find_runs()
    info = check_runs(found)
    sets = gather_sets()
    if a.step == "check":
        bv.log("check passed: all four readers, the pretrained weights and the crops are in place"); return
    bv.patch()
    out = bv.STORAGE / "experiments" / EXPERIMENT
    out.mkdir(parents=True, exist_ok=True)
    for f in ("REPORT.md", "summary.json", "heads.csv", "weights.json", "features.json", "gradients.json", "worker_log.txt"):
        (out / f).unlink(missing_ok=True)          # never mix results of an earlier run into this one
    job = dict(out=str(out), device="cpu" if bv.CPU else "gpu", parts=a.parts.split(","), sets=sets,
               weights={"pretrained": str(bv.PRETRAINED), **{r: str(d / "best_accuracy") for r, d in found.items()}},
               configs={r: str(d / "config.yml") for r, d in found.items()})
    bv.write_json(out / "job.json", job)
    bv.run([sys.executable, "-u", str(Path(__file__).resolve()), "worker", str(out / "job.json")],
           logfile=out / "worker_log.txt", show=re.compile(r"^\[\d\d:\d\d:\d\d\]|Traceback|Error|WORKER"),
           env={"B2N_POOL40": "0", "PYTHONPATH": str(bv.REPO)})
    summarise(out, sets, found, info)
    z = HERE / f"results_{EXPERIMENT}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in ("REPORT.md", "summary.json", "heads.csv", "weights.json", "features.json", "gradients.json",
                  "job.json", "worker_log.txt"):
            if (out / f).is_file():
                zf.write(out / f, f"results_{EXPERIMENT}/{f}")
    bv.log(f"DONE. Download {z.name} from {HERE}")


if __name__ == "__main__":
    main()
