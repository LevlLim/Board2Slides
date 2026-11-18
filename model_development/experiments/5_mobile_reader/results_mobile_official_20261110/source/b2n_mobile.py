#!/usr/bin/env python3
"""Board2Notes MOBILE reader experiments (PP-OCRv5_mobile_rec, PPLCNetV3 backbone) on RunPod.

Built on the unchanged server package `b2n_versions.py` (same data zips, label lists, scorers, export check, report),
so mobile and server results are produced by the same code. Only the mobile-specific parts live here.

Starting weights (B2N_MOBILE_START, fixed per experiment):
  official  PP-OCRv5_mobile_rec_pretrained.pdparams, official 18,383-character dictionary (as every server S/G run)
  t0ckpt    the teammate's TRAINING checkpoint (best_accuracy.pdparams: un-fused backbone + NRTR head) and its
            94-character ASCII dictionary. The supplied inference export cannot be used: it is fused (re-parameterised)
            and has no NRTR head. Refuses to start until T0_teammate/PROVENANCE.json is completed (see README).

Runs (all from the SAME starting weights; every run ~6,000 updates: epochs = round(6000 / updates per epoch)):
 Strategy batch (server S0 data lists, byte-identical; two seeds):
  MC0 / MC0s2  control: guidance on, CTC/NRTR weights 1.0/1.0 (explicit), uniform backbone LR, official SE placement,
               64x960 input (ZB, 10 Nov: the teammate's width; server runs used 1024), width/8 positions
  MG  / MGs2   MC0 + use_guide False (the CTC loss also trains the backbone)          (= server G0)
  ML  / MLs2   MC0 + six-group backbone LR multipliers [conv1, blocks2..6] = [0.1, 0.1, 0.1, 0.25, 0.5, 1.0]  (= server S1)
  ME  / MEs2   MC0 + an SE layer added to backbone block blocks6.3 (near-identity initialisation, gate 0.99)  (new)
  (focal CTC, server S2/S3, is not repeated)
 Data batch (server versions_2x2 lists, byte-identical; one seed, as on the server):
  MV1b MV2short MV3short MV2 MV3   IAM only / short only / + augmented copies / + long lines, adapted recipe
  MC1          V3 data, PaddleOCR's original recipe (48x320, 40 positions)
  MP40 / MP128 pooling-only pair: V3 lines that fit 40 positions, 64x960; only the TRAINING pooling differs
 Teammate recipe (optional, one seed):
  MT           the teammate's Colab settings on the SAME lines as MC1/MP40: 64x960 with training heights 32/48/64,
               original 40-position pooling, LR 5e-5 with a 2-epoch warm-up, RecConAug length limit 100, his RecAug
               keys (which this RecAug ignores, so augmentation is otherwise PaddleOCR's default, as for MP40). Kept: official
               dictionary (his drops characters our labels use) and the ~6,000-update budget (not his 20 epochs).
               A recipe comparison (several settings at once), not a single-factor test.
Primary rule (fixed before training, same as the server batches): whiteboard-validation mean line CER, mean of two
seeds; a change "helps" only if the paired 95% interval is below 0 AND both seeds improve against the matching MC0 seed.

PaddleOCR changes for PPLCNetV3 (rec_lcnetv3.py), added to the server patches already in the checkout:
  1 training pooling [1, 40] -> [1, width/2] (width/8 positions); B2N_POOL40=1 keeps 40 (P40 arm)
  2 reading pooling [3, 2] -> [feature height, 2] (at 64 px the original averaged 3 of the 4 feature rows)
  3 optional Backbone.b2n_extra_se: add SELayer to named blocks with conv2 weight 0 and bias 2.94
    (hardsigmoid(2.94) = 0.99, so the block starts almost unchanged and the gate can still learn)

Usage (through run_runpod_mobile.sh):  python b2n_mobile.py all --experiment mobile_official_20261110 [--smoke-only]
                                       python b2n_mobile.py check   (data and list checks only; no Paddle or GPU)
"""
from __future__ import annotations

import argparse, csv, hashlib, inspect, json, os, re, shutil, sys, time, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2n_versions as bv  # noqa: E402  (unchanged server package; its SHA-256 is checked below)

SERVER_SCRIPT_SHA256 = "a4f4ff156075772a99b4c9a6eaabeead3401442f4ad04f5a167cc9e26eaa19a8"   # board2notes_versions_runpod/b2n_versions.py (G0 batch)
START = os.environ.get("B2N_MOBILE_START", "official")
if START not in ("official", "t0ckpt"):
    raise SystemExit("B2N_MOBILE_START must be 'official' or 't0ckpt'")
OFFICIAL_PRETRAINED = Path(os.environ.get("B2N_MOBILE_PRETRAINED", bv.REPO / "PP-OCRv5_mobile_rec_pretrained"))
OFFICIAL_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams"
T0_DIR = Path(os.environ.get("B2N_T0_DIR", HERE.parent / "T0_teammate"))
T0_CHECKPOINT = Path(os.environ.get("B2N_T0_CHECKPOINT", T0_DIR / "training_checkpoint" / "best_accuracy"))
T0_DICT = T0_DIR / "reconstructed_dict.txt"
T0_DICT_SHA256 = "2178e24663ed540541d54c0310f78ef530121aa3fec2552ec054ccf7e9394605"   # 94 chars rebuilt from T0 inference.yml
T0_EXPORT = T0_DIR / "as_received"
SERVER_DICT = bv.REPO / "ppocr/utils/dict/ppocrv5_dict.txt"
MOBILE_CFG = bv.REPO / "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml"
CTC_W, NRTR_W = 1.0, 1.0                       # written explicitly as Loss.weight_1 / Loss.weight_2
LR_MULT6 = [0.1, 0.1, 0.1, 0.25, 0.5, 1.0]     # PPLCNetV3: conv1, blocks2, blocks3, blocks4, blocks5, blocks6
EXTRA_SE = "blocks6.3"                         # last backbone block (480 channels at scale 0.95); no SE officially
SE_GATE_BIAS = 2.94

bv.GEOMS["h64w960"] = dict(H=64, W=960)   # mobile recipe width (ZB decision, 10 Nov); server recipe "h64" is 64x1024
MOBILE_GEOM = "h64w960"
MOBILE_SPEC = {}
for _seed, _sfx in ((1, ""), (2, "s2")):
    _base = dict(geom=MOBILE_GEOM, pool40=False, positions=128, files=["wb_train", "long_train"], seed=_seed)
    MOBILE_SPEC["MC0" + _sfx] = dict(_base, what=f"mobile control C0: guidance on, CTC/NRTR 1.0/1.0, uniform LR (seed {_seed})")
    MOBILE_SPEC["MG" + _sfx] = dict(_base, guide=False, what=f"C0 + guidance off (seed {_seed})")
    MOBILE_SPEC["ML" + _sfx] = dict(_base, lr_mult6=LR_MULT6, what=f"C0 + six-group backbone LR {LR_MULT6} (seed {_seed})")
    MOBILE_SPEC["ME" + _sfx] = dict(_base, extra_se=EXTRA_SE, what=f"C0 + SE added to {EXTRA_SE} (seed {_seed})")
DATA_BATCH = ["V1b", "V2short", "V3short", "V2", "V3", "C1", "P40", "P128"]
for _v in DATA_BATCH:   # identical specs (files, length limits, geometry, pooling) to the server data batch
    MOBILE_SPEC["M" + _v] = dict(bv.RUN_SPEC[_v], seed=1, what="mobile " + bv.RUN_SPEC[_v]["what"])
    if MOBILE_SPEC["M" + _v]["geom"] == "h64":          # adapted-recipe runs move to 960; MC1 keeps PaddleOCR's 48x320
        MOBILE_SPEC["M" + _v]["geom"] = MOBILE_GEOM
MOBILE_SPEC["MT"] = dict(geom=MOBILE_GEOM, pool40=True, positions=40, files=bv.TRAIN_FILES_V3, seed=1, teammate_recipe=True,
                         what="teammate's Colab recipe (64x960, heights 32/48/64, 40 positions, LR 5e-5) on MC1/MP40's lines")
bv.RUN_SPEC.update(MOBILE_SPEC)
STRATEGY_RUNS = ["MC0", "MG", "ML", "ME", "MC0s2", "MGs2", "MLs2", "MEs2"]
DEFAULT_RUNS = STRATEGY_RUNS + ["M" + v for v in DATA_BATCH] + ["MT"]
TEAMMATE = dict(scales=[[960, 32], [960, 48], [960, 64]], lr=5e-5, warmup_epochs=2, conaug_max_text=100,
                recaug=dict(use_blur=True, use_gain=True, use_flip=False, prob=0.8))   # RecAug ignores these keys (**kwargs)
PARTNER = {r: ("MC0s2" if r.endswith("s2") else "MC0") for r in MOBILE_SPEC if r[:2] in ("MG", "ML", "ME")}
PARTNER["MP40"] = "MP128"
ALWAYS = {"Global.save_model_dir", "Global.save_res_path", "Train.dataset.label_file_list.0"}
ALLOWED = {"MG": {"Architecture.Head.head_list.0.CTCHead.Neck.use_guide"},
           "ML": {f"Architecture.Backbone.lr_mult_list.{i}" for i in range(6)},
           "ME": {"Architecture.Backbone.b2n_extra_se.0"}, "MP": set()}
COMPARISONS = [("MGavg", "MC0avg", "guidance off vs C0 (mean of 2 seeds)"),
               ("MLavg", "MC0avg", "six-group backbone LR vs C0 (mean of 2 seeds)"),
               ("MEavg", "MC0avg", "extra SE in blocks6.3 vs C0 (mean of 2 seeds)"),
               ("MG", "MC0", "guidance off, seed 1"), ("MGs2", "MC0s2", "guidance off, seed 2"),
               ("ML", "MC0", "six-group LR, seed 1"), ("MLs2", "MC0s2", "six-group LR, seed 2"),
               ("ME", "MC0", "extra SE, seed 1"), ("MEs2", "MC0s2", "extra SE, seed 2"),
               ("MC0s2", "MC0", "seed noise: control recipe, different seed"),
               ("MV1b", "start", "IAM-only adapted fine-tune vs no fine-tune"),
               ("MV2", "MV1b", "our whiteboard data vs IAM (replacement, not addition)"),
               ("MV2", "MV2short", "C vs A: adding long sentences, without our augmented copies"),
               ("MV3", "MV3short", "D vs B: adding long sentences, with our augmented copies"),
               ("MV3short", "MV2short", "B vs A: our augmented copies, short lines only"),
               ("MV3", "MV2", "D vs C: our augmented copies, short + long lines"),
               ("MV3", "MC1", "adapted recipe vs original recipe (several changes at once)"),
               ("MP128", "MP40", "training pooling ONLY (mobile; same lines, input, weights, budget)"),
               ("MP128", "MT", "our adapted recipe vs the teammate's recipe (same lines; several changes at once)"),
               ("MT", "MC1", "teammate's recipe vs PaddleOCR's original recipe (same lines)"),
               ("MC0avg", "start", "control vs its starting weights (extra training on our data)"),
               ("MC0avg", "T0export", "control vs the teammate's unchanged export")]
TEST_SETS = ("whiteboard val", "CNN test", "long sentences (held out)", bv.LONG_VIEW, "new writers", "IAM val")


def log(*a):
    bv.log(*a)


# ---------------------------------------------------------------- configuration of the shared server code
def configure():
    if bv.sha(bv.__file__) != SERVER_SCRIPT_SHA256:
        raise SystemExit("b2n_versions.py differs from the pinned server script; the mobile package reuses it unchanged")
    bv.BASE_CFG = MOBILE_CFG
    if START == "official":
        bv.PRETRAINED, bv.DICT = OFFICIAL_PRETRAINED, SERVER_DICT
    else:
        prov = T0_DIR / "PROVENANCE.json"
        p = json.loads(prov.read_text()) if prov.exists() else {}
        missing = [k for k in ("checkpoint_sha256", "dictionary_file_sha256", "train_list_supplied", "overlap_check_passed",
                               "source_commit_or_notebook", "train_log_supplied") if not p.get(k)]
        if missing:
            raise SystemExit(f"t0ckpt start refused: {prov} lacks {missing}. Use B2N_MOBILE_START=official, or complete "
                             "the provenance record (tools/check_t0_provenance.py) first.")
        if bv.sha(str(T0_CHECKPOINT) + ".pdparams") != p["checkpoint_sha256"]:
            raise SystemExit("T0 training checkpoint differs from PROVENANCE.json")
        if bv.sha(T0_DICT) != T0_DICT_SHA256 or p["dictionary_file_sha256"] != T0_DICT_SHA256:
            raise SystemExit("the teammate's dictionary file differs from the 94 characters in T0's inference.yml")
        bv.PRETRAINED, bv.DICT = T0_CHECKPOINT, T0_DICT


_server_make_cfg = bv.make_cfg


def sampler_batches(n, scales, first_bs, divided=(8, 16)):
    """Batches per epoch of PaddleOCR's MultiScaleSampler (fix_bs False), replicating its loop: every pass over the
    scales appends one batch per scale (short batches are padded), until n samples are covered. Checked against the
    real class: 4,269 lines at heights 32/48/64 -> batch sizes 16/10/8 -> 378 batches; one scale -> ceil(n/16)."""
    bw, bh = scales[0]
    base = bw * bh * first_bs
    sizes = [int(max(1, base / ((h // divided[1] * divided[1]) * (w // divided[0] * divided[0])))) for w, h in scales]
    passes = -(-n // sum(sizes))
    return passes * len(sizes), sizes


assert sampler_batches(4269, [[960, 32], [960, 48], [960, 64]], 16) == (378, [16, 10, 8])
assert sampler_batches(8356, [[960, 32], [960, 48], [960, 64]], 16)[0] == 738 and sampler_batches(1359, [[960, 64]], 16)[0] == 85


def make_cfg(save_dir, geom, train_file, eval_file="val_wb", steps=bv.STEPS, eval_step=bv.EVAL_STEP, opts=None):
    """The server make_cfg on the official MOBILE config, then: explicit loss weights, six-group LR, extra SE."""
    import yaml
    opts = dict(opts or {})
    path, plan = _server_make_cfg(save_dir, geom, train_file, eval_file, steps, eval_step,
                                  {k: v for k, v in opts.items() if k != "lr_mult"})
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    bb, loss = cfg["Architecture"]["Backbone"], cfg["Loss"]
    assert bb["name"] == "PPLCNetV3" and cfg["Architecture"]["algorithm"] == "SVTR_LCNet", bb
    assert loss["name"] == "MultiLoss", loss
    loss["weight_1"], loss["weight_2"] = CTC_W, NRTR_W      # nested 'CTCLoss: {weight: ..}' would be ignored
    if opts.get("lr_mult6"):
        assert len(opts["lr_mult6"]) == 6
        bb["lr_mult_list"] = list(opts["lr_mult6"])
    if opts.get("extra_se"):
        bb["b2n_extra_se"] = [opts["extra_se"]]
    if opts.get("teammate_recipe"):     # the teammate's Colab settings (9 Nov log), on top of the shared budget
        tr = cfg["Train"]
        assert tr["sampler"]["name"] == "MultiScaleSampler"
        tr["sampler"]["scales"] = [list(x) for x in TEAMMATE["scales"]]
        cfg["Optimizer"]["lr"]["learning_rate"] = TEAMMATE["lr"]
        cfg["Optimizer"]["lr"]["warmup_epoch"] = TEAMMATE["warmup_epochs"]
        ops = tr["dataset"]["transforms"]
        k = [i for i, op in enumerate(ops) if "RecAug" in op]
        assert len(k) == 1, ops
        ops[k[0]] = {"RecAug": dict(TEAMMATE["recaug"])}
        c = [i for i, op in enumerate(ops) if "RecConAug" in op]
        assert len(c) == 1, ops
        ops[c[0]]["RecConAug"]["max_text_length"] = TEAMMATE["conaug_max_text"]   # his yml anchors it to 100 (official: 25)
        # mixed heights give batch sizes 16/10/8, so an epoch has MORE updates than lines/16: recompute the budget
        per_epoch, sizes = sampler_batches(plan["lines"], tr["sampler"]["scales"], bv.BATCH)
        epochs = max(1, round(steps / per_epoch)); planned = epochs * per_epoch
        ev = eval_step if planned >= 2 * eval_step else max(1, planned // 3)
        cfg["Global"]["epoch_num"] = epochs
        cfg["Global"]["eval_batch_step"] = [0, ev]
        plan = dict(plan, steps_per_epoch=per_epoch, epochs=epochs, planned_updates=planned, eval_step=ev,
                    batch_sizes=sizes, note="multi-height sampler: matched UPDATES; images seen per update differ")
    cfg["Global"]["model_name"] = "PP-OCRv5_mobile_rec"
    yaml.safe_dump(cfg, open(path, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
    return path, plan


bv.make_cfg = make_cfg

# export: identical to the server export except the model name written into inference.yml
# and with a frame-level equivalence fallback. Export fuses LCNet's multi-branch convolutions (rep()), which adds small
# numeric differences (CPU: max |dp| ~0.003, 100% frame agreement). On an under-trained smoke model with many near-ties
# they can flip single characters (RunPod smoke, 10 Nov: 80% identical text, mean CER difference 0.0046). A real export
# bug (height, pooling, dictionary) changes many frames, so the fallback still requires >= 99% of output frames to pick
# the same class, plus the server's mean-CER tolerance. The measured numbers are written into verified.json.
LOGIT_CHECK = r"""
import sys, json, yaml, numpy as np, paddle, cv2
from pathlib import Path
sys.path.insert(0, ".")
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.data.imaug.rec_img_aug import resize_norm_img_chinese
import paddle.inference as pi
cfg_path, ckpt, export_dir, fixture, data_root, H, minW, device = sys.argv[1:9]; H, minW = int(H), int(minW)
paddle.set_device(device)
cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
post = build_post_process(cfg["PostProcess"], cfg["Global"]); n = len(post.character)
cfg["Architecture"]["Head"]["out_channels_list"] = {"CTCLabelDecode": n, "NRTRLabelDecode": n + 3}
m = build_model(cfg["Architecture"]); m.set_state_dict(paddle.load(ckpt + ".pdparams")); m.eval()
c = pi.Config(str(Path(export_dir) / "inference.json"), str(Path(export_dir) / "inference.pdiparams"))
c.enable_use_gpu(500, 0) if device == "gpu" else c.disable_gpu()
c.switch_ir_optim(False); c.enable_custom_passes([], True); pred = pi.create_predictor(c)
ih = pred.get_input_handle(pred.get_input_names()[0])
rows = [l.rstrip("\n").split("\t", 1) for l in open(fixture, encoding="utf-8") if "\t" in l]
maxd, frames, agree = 0.0, 0, 0
for p, lab in rows:
    x, _ = resize_norm_img_chinese(cv2.imread(str(Path(data_root) / p)), [3, H, minW]); x = x[None].astype("float32")
    a = m(paddle.to_tensor(x)); a = (a["ctc"] if isinstance(a, dict) else a).numpy()[0]
    ih.reshape(list(x.shape)); ih.copy_from_cpu(x); pred.run(); b = pred.get_output_handle(pred.get_output_names()[0]).copy_to_cpu()[0]
    assert a.shape == b.shape, (a.shape, b.shape)
    maxd = max(maxd, float(np.abs(a - b).max())); frames += a.shape[0]; agree += int((a.argmax(-1) == b.argmax(-1)).sum())
print("LOGITCHECK " + json.dumps(dict(crops=len(rows), frames=frames, frame_argmax_agreement=agree / frames, max_abs_prob_diff=maxd)))
"""
FRAME_AGREEMENT_MIN = 0.99
_LOGIT = {}


def _b2n_logit_ok(d, stage, G, dcer, same):
    lf = d / "export_logit_check.log"; lf.unlink(missing_ok=True)
    bv.run([sys.executable, "-c", LOGIT_CHECK, d / "config.yml", d / "best_accuracy", stage, bv.DATA / "export_fixture.txt",
            bv.ROOT, G["H"], G["W"], "cpu"], logfile=lf, show=re.compile("LOGITCHECK|Error"))
    line = [l for l in lf.read_text(errors="replace").splitlines() if l.startswith("LOGITCHECK ")]
    if not line:
        return False
    r = json.loads(line[-1][len("LOGITCHECK "):]); _LOGIT[d.name] = r
    ok = r["frame_argmax_agreement"] >= FRAME_AGREEMENT_MIN and dcer <= bv.EXPORT_CER_TOL
    log(f"  {d.name}: text identical {same:.0%} < {bv.EXPORT_SAME_MIN:.0%}; frame-level check: "
        f"{r['frame_argmax_agreement']:.4f} of {r['frames']} output frames agree (need {FRAME_AGREEMENT_MIN}), "
        f"max |prob diff| {r['max_abs_prob_diff']:.4f}, mean CER difference {dcer:.4f} -> {'EQUIVALENT' if ok else 'NOT equivalent'}")
    return ok


bv._b2n_logit_ok, bv._B2N_LOGIT = _b2n_logit_ok, _LOGIT
# All scoring and the export check run on CPU (RunPod smoke diagnostic, 10 Nov, smoke_MG): the UN-fused LCNet checkpoint
# read on GPU disagreed with every other variant (max |dp| 0.93, 67-70% identical text, CER 0.149 vs 0.131-0.137),
# while checkpoint and export on CPU agreed to max |dp| 0.0006 (100% identical text). CPU is also how the app reads.
# Training (and PaddleOCR's own in-training validation, which picks each run's best checkpoint) stays on the GPU.
_s = inspect.getsource(bv.score_checkpoint)
for _a, _b in (('meta = dict(weights=sha(str(weights) + ".pdparams"), config=sha(cfg),',
                'meta = dict(device="cpu", weights=sha(str(weights) + ".pdparams"), config=sha(cfg),'),
               ('f"Global.use_gpu={not CPU}"', '"Global.use_gpu=False"')):
    assert _s.count(_a) == 1, _a
    _s = _s.replace(_a, _b)
exec(_s, bv.__dict__)
_s = inspect.getsource(bv.score_static)
for _a, _b in (('meta = dict(runtime=STATIC_RUNTIME,', 'meta = dict(device="cpu", runtime=STATIC_RUNTIME,'),
               ('"cpu" if CPU else "gpu"]', '"cpu"]')):
    assert _s.count(_a) == 1, _a
    _s = _s.replace(_a, _b)
exec(_s, bv.__dict__)
_src = inspect.getsource(bv.export)
for _a, _b in (('"Global.model_name=PP-OCRv5_server_rec"', '"Global.model_name=PP-OCRv5_mobile_rec"'),
               ("    if same < EXPORT_SAME_MIN or dcer > EXPORT_CER_TOL:\n",
                "    if (same < EXPORT_SAME_MIN or dcer > EXPORT_CER_TOL) and not _b2n_logit_ok(d, stage, G, dcer, same):\n"),
               ("fixture_crops=len(a), differing=diffs))", "fixture_crops=len(a), differing=diffs, frame_check=_B2N_LOGIT.get(name)))"),
               ("ident = dict(runtime=STATIC_RUNTIME,", "ident = dict(score_device=\"cpu\", runtime=STATIC_RUNTIME,")):
    assert _src.count(_a) == 1, _a
    _src = _src.replace(_a, _b)
exec(_src, bv.__dict__)


def seed_average(results):
    """The server seed averaging, extended to the mobile run names."""
    for base in ("MC0", "MG", "ML", "ME"):
        if base in results and base + "s2" in results:
            tmp = {"S0": results[base], "S0s2": results[base + "s2"]}
            _server_seed_average(tmp)
            results[base + "avg"] = [dict(r, model=base + "avg") for r in tmp["S0avg"]]
    _server_seed_average(results)


_server_seed_average = bv.seed_average
bv.seed_average = seed_average


# ---------------------------------------------------------------- PaddleOCR patches for PPLCNetV3
SE_CODE = '''        for _b2n_spec in kwargs.get("b2n_extra_se") or []:  # board2notes mobile patch 3: SE added to a named block
            _b2n_stage, _b2n_i = _b2n_spec.split(".")
            _b2n_blk = getattr(self, _b2n_stage)[int(_b2n_i)]
            assert not _b2n_blk.use_se, f"{_b2n_spec} already has SE"
            _b2n_blk.se = SELayer(make_divisible(self.net_config[_b2n_stage][int(_b2n_i)][1] * scale),
                                  lr_mult=self.lr_mult_list[int(_b2n_stage[-1]) - 1])
            Constant(0.0)(_b2n_blk.se.conv2.weight)   # near-identity start: gate = hardsigmoid(2.94) = 0.99
            Constant(2.94)(_b2n_blk.se.conv2.bias)
            _b2n_blk.use_se = True
'''


def patch():
    bv.patch()                                   # server patches (HGNet pooling, export height): unchanged
    f = bv.REPO / "ppocr/modeling/backbones/rec_lcnetv3.py"
    s = f.read_text(encoding="utf-8")
    pairs = [("import paddle\n", "import os  # board2notes\nimport paddle\n"),
             ("x = F.adaptive_avg_pool2d(x, [1, 40])",
              "x = F.adaptive_avg_pool2d(x, [1, 40] if os.environ.get('B2N_POOL40') == '1' else [1, x.shape[3] // 2])"
              "  # board2notes mobile patch 1: width/8 training positions (B2N_POOL40=1 keeps the original 40)"),
             ("x = F.avg_pool2d(x, [3, 2])",
              "x = F.avg_pool2d(x, [x.shape[2], 2])  # board2notes mobile patch 2: average the full feature height"),
             ("        self.out_channels = make_divisible(512 * scale)\n",
              "        self.out_channels = make_divisible(512 * scale)\n" + SE_CODE)]
    for old, new in pairs:
        if new in s:
            continue
        if s.count(old) != 1:
            raise SystemExit(f"mobile patch: expected exactly one '{old.strip()[:50]}' in rec_lcnetv3.py")
        s = s.replace(old, new)
    f.write_text(s, encoding="utf-8")
    log("PaddleOCR mobile patches in place: LCNet training pooling, reading pooling, optional extra SE")


# ---------------------------------------------------------------- data
def train_file(name):
    return f"train_{name}_t0vocab" if START == "t0ckpt" else f"train_{name}"


def prepare_mobile(check_only=False):
    bv.DICT = SERVER_DICT if SERVER_DICT.is_file() else bv.DICT      # server data check (full dictionary)
    prepared = bv.prepare(check_only=check_only)
    D = bv.DATA
    eq = lambda a, b: bv.sha(D / f"train_{a}.txt") == bv.sha(D / f"train_{b}.txt")
    assert eq("MC0", "S0") and eq("MC0s2", "S0s2"), "MC0/MC0s2 must use server S0/S0s2's exact lists"
    for v in DATA_BATCH:
        assert eq("M" + v, v), f"M{v} must use server {v}'s exact list"
    assert eq("MT", "MC1") and eq("MT", "MP40"), "MT must train on MC1/MP40's exact lines"
    for r in MOBILE_SPEC:
        if r in PARTNER and r != "MP40":
            assert eq(r, PARTNER[r]), f"{r} must use {PARTNER[r]}'s exact list"
    assert eq("MP40", "MP128"), "MP40 and MP128 must train on identical lines"
    lens = {r: sum(1 for _ in open(D / f"train_{r}.txt", encoding="utf-8")) for r in ("MC0", "MP128", "MV3")}
    log(f"mobile lists: strategy runs = server S0 ({lens['MC0']} lines); data batch = server versions_2x2 lists "
        f"(MV3 {lens['MV3']}); MP40 = MP128 = MC1 = MT ({lens['MP128']} lines fitting 40 positions)")
    if START == "t0ckpt":
        vocab = set(T0_DICT.read_text(encoding="utf-8").split("\n")[:-1]) | {" "}
        for r in list(MOBILE_SPEC) + ["smoke_train_h64", "smoke_train_40"]:
            src = D / (f"train_{r}.txt" if r in MOBILE_SPEC else f"{r}.txt")
            keep = [l for l in open(src, encoding="utf-8") if all(c in vocab for c in l.rstrip("\n").split("\t", 1)[1])]
            (D / (f"train_{r}_t0vocab.txt" if r in MOBILE_SPEC else f"{r}_t0vocab.txt")).write_text("".join(keep), encoding="utf-8")
        log("t0ckpt: training lists filtered to the 95-character T0 vocabulary (CTCLabelEncode would silently drop "
            "unknown characters); validation labels unchanged (4 contain characters T0 cannot output)")
        bv.DICT = T0_DICT
    else:
        bv.DICT = SERVER_DICT
    return prepared


# ---------------------------------------------------------------- smoke: mobile-specific structural checks
MOBILE_CHECK = r"""
import json, os, sys, yaml, numpy as np, paddle
sys.path.insert(0, ".")
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.losses import build_loss
paddle.set_device(sys.argv[1]); start = sys.argv[2]; out = {}
for tag, path in zip(sys.argv[3::2], sys.argv[4::2]):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    n = len(build_post_process(cfg["PostProcess"], cfg["Global"]).character)
    cfg["Architecture"]["Head"]["out_channels_list"] = {"CTCLabelDecode": n, "NRTRLabelDecode": n + 3}
    paddle.seed(0); m = build_model(cfg["Architecture"])
    sd = m.state_dict(); pre = paddle.load(start + ".pdparams")
    same = [k for k in sd if k in pre and list(pre[k].shape) == list(sd[k].shape)]
    shape = [k for k in sd if k in pre and list(pre[k].shape) != list(sd[k].shape)]
    new = [k for k in sd if k not in pre]
    m.set_state_dict({k: pre[k] for k in same})
    r = dict(classes=n, loaded=len(same), shape_mismatch=shape[:8], n_shape_mismatch=len(shape), not_in_start=new[:12],
             n_not_in_start=len(new), unused_in_start=len([k for k in pre if k not in sd]))
    shapes = {}
    for pool40 in ("0", "1"):
        os.environ["B2N_POOL40"] = pool40; m.backbone.train(); a = m.backbone(paddle.randn([1, 3, 64, 960])).shape
        m.backbone.eval(); b = m.backbone(paddle.randn([1, 3, 64, 960])).shape; shapes[pool40] = dict(train=list(a), eval=list(b))
    os.environ["B2N_POOL40"] = "0"; r["positions_64x960"] = shapes
    blk = m.backbone.blocks6[3]
    if blk.use_se:
        x = paddle.randn([2, 480, 1, 64]); g = blk.se.hardsigmoid(blk.se.conv2(blk.se.relu(blk.se.conv1(blk.se.avg_pool(x)))))
        r["extra_se_gate"] = [float(g.min()), float(g.max())]
    r["lr_by_group"] = {k: sorted({round(float(p.optimize_attr["learning_rate"]), 4) for p in getattr(m.backbone, k).parameters()
                                   if not p.stop_gradient})           # trainable only (BN running stats are not)
                        for k in ("conv1", "blocks2", "blocks3", "blocks4", "blocks5", "blocks6")}
    L = build_loss(dict(cfg["Loss"])); r["loss_weights"] = [L.weight_1, L.weight_2]
    m.train(); x = paddle.randn([2, 3, 64, 256])
    y = m.head.ctc_head(m.head.ctc_encoder(m.backbone(x))); y = y[-1] if isinstance(y, (list, tuple)) else y
    (y.astype("float32") ** 2).mean().backward()
    r["ctc_only_backbone_grad"] = sum(float(p.grad.abs().sum()) for p in m.backbone.parameters() if p.grad is not None)
    r["use_guide"] = cfg["Architecture"]["Head"]["head_list"][0]["CTCHead"]["Neck"]["use_guide"]
    if blk.use_se:   # one real optimiser step on the full CTC+NRTR loss: the added SE gate must move
        from ppocr.data.imaug.label_ops import MultiLabelEncode
        enc = MultiLabelEncode(max_text_length=cfg["Global"]["max_text_length"], character_dict_path=cfg["Global"]["character_dict_path"],
                               use_space_char=True, gtc_encode="NRTRLabelEncode")
        it = [enc({"image": None, "label": t}) for t in ("hello world", "mesh of wires")]
        bt = [paddle.to_tensor(np.stack([i[k] for i in it]).astype("int64")) for k in ("label_ctc", "label_gtc")] + \
             [paddle.to_tensor(np.array([i["length"] for i in it], "int64")), paddle.to_tensor(np.ones(2, "float32"))]
        m.clear_gradients(); w0 = blk.se.conv2.weight.numpy().copy(); b0 = blk.se.conv2.bias.numpy().copy()
        opt = paddle.optimizer.Adam(learning_rate=1e-4, parameters=m.parameters())
        x = paddle.randn([2, 3, 64, 256]); loss = build_loss(dict(cfg["Loss"]))(m(x, bt), [x] + bt)["loss"]
        loss.backward(); opt.step()
        r["extra_se_moves_after_one_step"] = [float(np.abs(blk.se.conv2.weight.numpy() - w0).sum()),
                                              float(np.abs(blk.se.conv2.bias.numpy() - b0).sum())]
    out[tag] = r
print("MOBILECHECK " + json.dumps(out))
"""


def flat(x, k=""):
    if isinstance(x, dict):
        return {kk: vv for a, b in x.items() for kk, vv in flat(b, f"{k}.{a}" if k else str(a)).items()}
    if isinstance(x, list):
        return {kk: vv for i, b in enumerate(x) for kk, vv in flat(b, f"{k}.{i}").items()}
    return {k: x}


def opts_of(r):
    s = MOBILE_SPEC[r]
    return dict(seed=s.get("seed"), guide=s.get("guide"), lr_mult6=s.get("lr_mult6"), extra_se=s.get("extra_se"),
                teammate_recipe=s.get("teammate_recipe"))


def check_configs(d, runs):
    """Each changed arm's config differs from its control ONLY in the intended keys (plus its own paths/list)."""
    import yaml
    paths = {}
    need = set(runs) | {PARTNER[x] for x in runs if x in PARTNER} | {"MC0"} | ({"MP40"} if "MT" in runs else set())
    for r in sorted(need):
        paths[r], _ = make_cfg(d / f"cfgcheck_{r}", MOBILE_SPEC[r]["geom"], train_file(r), opts=opts_of(r))
    fs = {r: flat(yaml.safe_load(open(p, encoding="utf-8"))) for r, p in paths.items()}
    for r in runs:
        if r not in PARTNER:
            continue
        c = PARTNER[r]
        diff = {k for k in set(fs[r]) | set(fs[c]) if fs[r].get(k) != fs[c].get(k)}
        allowed = ALWAYS | ALLOWED[r[:2]]
        assert diff <= allowed and (diff - ALWAYS or r == "MP40"), f"{r} vs {c}: unexpected differences {sorted(diff - allowed)}"
        for k in ("Global.seed", "Global.pretrained_model", "Loss.weight_1", "Loss.weight_2", "Global.character_dict_path"):
            assert fs[r].get(k) == fs[c].get(k), (r, k)
    if "MT" in runs:   # the teammate-recipe arm differs from MP40 (same lines, 40 positions) only in his settings
        diff = {k for k in set(fs["MT"]) | set(fs["MP40"]) if fs["MT"].get(k) != fs["MP40"].get(k)} - ALWAYS
        ok = ("Train.sampler.scales.", "Global.d2s_train_image_shape.", "Optimizer.lr.learning_rate", "Optimizer.lr.warmup_epoch",
              "Train.dataset.transforms.", "Global.epoch_num", "Global.eval_batch_step.")   # epochs differ BY DESIGN (budget)
        bad = [k for k in diff if not k.startswith(ok) and not re.fullmatch(r"Eval\.dataset\.transforms\.\d+\.RecResizeImg\.image_shape\.\d", k)]
        assert not bad, f"MT vs MP40: unexpected differences {sorted(bad)}"
        aug = {k for k in diff if k.startswith("Train.dataset.transforms.")}
        assert all(".RecAug." in k or k.endswith(".RecConAug.max_text_length") for k in aug), \
            f"MT vs MP40: training transforms changed beyond RecAug / RecConAug length: {sorted(aug)}"
        assert fs["MT"]["Optimizer.lr.learning_rate"] == TEAMMATE["lr"] and fs["MT"]["Train.sampler.scales.0.1"] == 32
        n = sum(1 for _ in open(bv.DATA / f"{train_file('MT')}.txt", encoding="utf-8"))
        upd = lambda r: fs[r]["Global.epoch_num"] * (sampler_batches(n, TEAMMATE["scales"], bv.BATCH)[0] if r == "MT" else -(-n // bv.BATCH))
        assert abs(upd("MT") - upd("MP40")) <= 0.1 * upd("MP40"), f"MT budget {upd('MT')} vs MP40 {upd('MP40')} updates"
        log(f"  MT budget: {fs['MT']['Global.epoch_num']} epochs = {upd('MT')} updates (MP40: {fs['MP40']['Global.epoch_num']} epochs = {upd('MP40')})")
    log(f"  config check: every arm differs from its control only in its intended change ({', '.join(sorted(paths))})")
    return paths


def check_structure(d, paths):
    want = {t: paths[t] for t in ("MC0", "MG", "ML", "ME") if t in paths}
    lf = d / "mobile_check.log"; lf.unlink(missing_ok=True)
    args = [x for t, p in want.items() for x in (t, str(p))]
    bv.run([sys.executable, "-c", MOBILE_CHECK, "cpu", str(bv.PRETRAINED), *args], logfile=lf, show=re.compile("MOBILECHECK|Error"))
    line = [l for l in lf.read_text(errors="replace").splitlines() if l.startswith("MOBILECHECK ")]
    assert line, lf.read_text(errors="replace")[-3000:]
    mc = json.loads(line[-1][len("MOBILECHECK "):])
    c0 = mc["MC0"]
    assert c0["n_shape_mismatch"] == 0, f"starting weights do not fit the model (dictionary/head?): {c0['shape_mismatch']}"
    assert c0["n_not_in_start"] == 0, f"model parameters missing from the starting weights: {c0['not_in_start']}"
    assert c0["positions_64x960"]["0"] == dict(train=[1, 480, 1, 120], eval=[1, 480, 1, 120]), c0["positions_64x960"]
    assert c0["positions_64x960"]["1"]["train"] == [1, 480, 1, 40], c0["positions_64x960"]
    assert c0["loss_weights"] == [CTC_W, NRTR_W], c0["loss_weights"]
    assert c0["use_guide"] is True and c0["ctc_only_backbone_grad"] == 0.0, c0
    if "MG" in mc:
        assert mc["MG"]["use_guide"] is False and mc["MG"]["ctc_only_backbone_grad"] > 0, mc["MG"]
    if "ML" in mc:
        got = [mc["ML"]["lr_by_group"][k] for k in ("conv1", "blocks2", "blocks3", "blocks4", "blocks5", "blocks6")]
        # each group holds exactly its multiplier, plus 0.1x of it for the Learnable Affine Blocks (lab_lr = 0.1)
        assert all(set(g) <= {round(m_, 4), round(0.1 * m_, 4)} and round(m_, 4) in g for m_, g in zip(LR_MULT6, got)), got
    if "ME" in mc:
        e = mc["ME"]
        assert e["n_shape_mismatch"] == 0 and all(".se." in k and "blocks6.3" in k for k in e["not_in_start"]) \
            and e["n_not_in_start"] == 4, e["not_in_start"]
        assert abs(e["extra_se_gate"][0] - 0.99) < 1e-3 and abs(e["extra_se_gate"][1] - 0.99) < 1e-3, e["extra_se_gate"]
        assert min(e["extra_se_moves_after_one_step"]) > 0, f"the added SE does not train: {e['extra_se_moves_after_one_step']}"
    bv.write_json(d / "mobile_check.json", mc)
    log(f"  structure check: {c0['loaded']} tensors loaded from the starting weights, none missing or mis-shaped; "
        "64x960 gives 120 positions in training and reading (40 with B2N_POOL40=1); loss weights 1.0/1.0 explicit; "
        "guided CTC-only backbone gradient 0, unguided > 0; six LR groups; extra SE = 4 new tensors, gate 0.99, and it moves after one optimiser step")


def smoke(prepared, runs):
    d = bv.RUNS / "smoke"
    if (d / "OK").exists():
        log("SMOKE OK: passed earlier for these exact experiment settings; skipped"); return
    d.mkdir(parents=True, exist_ok=True)
    paths = check_configs(d, runs)
    check_structure(d, paths)
    sets = {s: f"smoke_{k}" for s, k in bv.EVAL_SETS.items()}
    results = score_baselines(sets, d)
    setups = {}
    for r in runs:
        s = MOBILE_SPEC[r]
        key = (s["geom"], s["pool40"], s.get("guide", True), tuple(s.get("lr_mult6") or ()), s.get("extra_se"), s["positions"],
               bool(s.get("teammate_recipe")))
        setups.setdefault(key, (f"smoke_{r.removesuffix('s2')}", r))
    sfx = "_t0vocab" if START == "t0ckpt" else ""
    cpu = ""
    if bv.TEST_MODE:   # local CPU plumbing test only: a few lines, so a 4 GB CPU machine can run every code path
        cpu = "_cpu"
        for f, k in (("smoke_train_h64" + sfx, 4), ("smoke_train_40" + sfx, 4), ("smoke_val", 2)):
            lines = open(bv.DATA / f"{f}.txt", encoding="utf-8").readlines()[:k]
            (bv.DATA / f"{f}_cpu.txt").write_text("".join(lines), encoding="utf-8")
    for (geom, pool40, guide, lr6, se, pos, tm), (name, r) in setups.items():
        lst = ("smoke_train_40" if pos == 40 else "smoke_train_h64") + sfx + cpu
        rd = bv.train(name, geom, pool40, lst, eval_file="smoke_val" + cpu, steps=24 if not bv.TEST_MODE else bv.STEPS,
                      eval_step=8, what="smoke", opts=opts_of(r))
        results[name] = bv.score_checkpoint(name, rd / "best_accuracy", rd / "config.yml", rd / "scores", sets)
        bv.export(name, geom, rd / "inference")
    names = [n for n, _ in setups.values()]
    bv.report(results, prepared, d / "report", comparisons=[(n, "start", "smoke") for n in names], test_sets=TEST_SETS)
    bv.write_json(d / "OK", dict(at=time.strftime("%Y-%m-%d %H:%M:%S"), models=sorted(results)))
    log(f"SMOKE OK: {len(setups)} mobile training setups, scoring, export checks and report work. Smoke CER means nothing.")


def score_baselines(sets, out_root):
    cfg, _ = make_cfg(out_root / "start", MOBILE_GEOM, train_file("MC0"))
    res = {"start": bv.score_checkpoint("start", bv.PRETRAINED, cfg, out_root / "start", sets)}
    res["T0export"] = bv.score_static("T0export", T0_EXPORT, 64, 640, out_root / "T0export", sets)
    return res


def mobile_sections(results, out_dir):
    """The verdict sections the shared server report only produces for server run names: the primary two-seed rule
    (MG/ML/ME vs MC0), the 2x2 augmentation x long-line interaction (MV runs) and the repetition diagnostic (MP40 vs
    MP128). Same bootstrap, same decision rules, appended to report.md and summary.json."""
    sets = list(dict.fromkeys(r["set"] for rows in results.values() for r in rows))
    def bs(a, b, st, field="cer"):
        ra = [r for r in results.get(a, []) if r["set"] == st]; rb = [r for r in results.get(b, []) if r["set"] == st]
        return bv.bootstrap(ra, rb, field) if ra and rb else None
    V, L = {}, ["", "## Mobile verdicts (rules fixed before training)", ""]
    P = []
    for t, what in (("MG", "guidance off"), ("ML", "six-group backbone LR"), ("ME", "extra SE in blocks6.3")):
        v, v1, v2 = bs(t + "avg", "MC0avg", "whiteboard val"), bs(t, "MC0", "whiteboard val"), bs(t + "s2", "MC0s2", "whiteboard val")
        if not (v and v1 and v2):
            continue
        ok = v[2] < 0 and v1[0] < 0 and v2[0] < 0
        bad = v[1] > 0 and v1[0] > 0 and v2[0] > 0
        dec = "helps" if ok else ("hurts" if bad else "improvement not established")
        V[f"primary {t}"] = dict(diff=v[0], lo=v[1], hi=v[2], seed1=v1[0], seed2=v2[0], verdict=dec)
        P.append(f"| {t} ({what}) | {v1[0]:+.4f} | {v2[0]:+.4f} | {v[0]:+.4f} | {v[1]:+.4f} .. {v[2]:+.4f} | **{dec}** |")
    if P:
        L += ["### Primary criterion: whiteboard validation, mean of 2 seeds", "",
              "A setting **helps** only if the seed-averaged 95% interval is below 0 AND both seeds improve on their own "
              "control (MC0 / MC0s2). Not met = 'improvement not established' (not proof of no effect); interval above 0 "
              "and both seeds worse = 'hurts'.", "",
              "| setting | seed 1 vs MC0 | seed 2 vs MC0s2 | mean of seeds | 95% interval | decision |", "|---|---|---|---|---|---|"] + P
        sn = bs("MC0s2", "MC0", "whiteboard val")
        if sn:
            L += ["", f"For scale: the control recipe with a different seed (MC0s2 vs MC0) differs by {sn[0]:+.4f}."]
    if all(m in results for m in ("MV2short", "MV3short", "MV2", "MV3")):
        L += ["", "### 2x2 interaction: do our augmented copies help MORE when long sentences are included? (mobile)", "",
              "Per crop: (MV3 - MV2) - (MV3short - MV2short) in CER. Negative = the copies help more with long sentences.", "",
              "| set | interaction | 95% interval | verdict |", "|---|---|---|---|"]
        for st in sets:
            q = {m: {r["file"]: r for r in results[m] if r["set"] == st} for m in ("MV2short", "MV3short", "MV2", "MV3")}
            if not q["MV2"]:
                continue
            diff = [dict(file=f, cluster=r["cluster"], cer=(float(q["MV3"][f]["cer"]) - float(r["cer"]))
                         - (float(q["MV3short"][f]["cer"]) - float(q["MV2short"][f]["cer"]))) for f, r in q["MV2"].items()]
            d, lo, hi, *_ = bv.bootstrap(diff, [dict(x, cer=0.0) for x in diff])
            v = "copies help more with long data" if hi < 0 else ("copies help less with long data" if lo > 0 else "no clear interaction")
            V[f"interaction {st}"] = dict(diff=d, lo=lo, hi=hi, verdict=v)
            L.append(f"| {st} | {d:+.4f} | {lo:+.4f} .. {hi:+.4f} | {v} |")
        L += ["", "Caution: short + long runs also have more original training lines, so 'adding long sentences' is partly "
              "'more data'."]
    if "MP40" in results and "MP128" in results:
        L += ["", "### Repetition diagnostic (MP40 vs MP128)", "",
              "Aligned insertions inside repeated substrings, per crop. Fewer flags with width/8 pooling are consistent "
              "with the train/read position-mismatch hypothesis but do not prove it.", "",
              "| set | repeated insertions MP40 - MP128 | 95% interval | diagnostic |", "|---|---|---|---|"]
        for st in sets:
            r = bs("MP40", "MP128", st, "extra_repeats")
            if r:
                v = "MP40 has more repetition flags" if r[1] > 0 else ("MP128 has more repetition flags" if r[2] < 0 else "no clear difference in flags")
                V[f"repeats {st}"] = dict(diff=r[0], lo=r[1], hi=r[2], verdict=v)
                L.append(f"| {st} | {r[0]:+.3f} | {r[1]:+.3f} .. {r[2]:+.3f} | {v} |")
    L += ["", "### Reading caveats", "",
          "- **Writer provenance:** the data batch (MV2short, MV3short, MV2, MV3, MC1, MP40, MP128) and MT train on the 64 "
          "teammates' crops of uncertain writer identity, exactly as the server data batch did. Their new-writer scores "
          "may be optimistic; the strategy batch (MC0/MG/ML/ME) excludes those crops.",
          "- **T0export** (the teammate's export) trained on our whiteboard-validation and CNN-test crops, all of Jy and "
          "101 of the 160 new-writer crops: its scores on those sets are 'seen in training', not held out.",
          "- **MT** is a recipe comparison (several settings at once). Its budget matches the others in UPDATES (multi-height "
          "batches of 16/10/8 images), so it sees fewer images per update than the fixed-batch runs.",
          "- Everything is mobile: 64x960 input (server runs used 64x1024), official mobile weights."]
    with open(out_dir / "report.md", "a", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    sm = out_dir / "summary.json"
    if sm.exists():
        j = bv.read_json(sm); j["mobile_verdicts"] = V; bv.write_json(sm, j)
    return V


def pack():
    bv.pack()
    z = bv.HERE / f"results_{bv.EXPERIMENT}.zip"
    with zipfile.ZipFile(z, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.write(Path(__file__), "source/b2n_mobile.py")
    log(f"results: {z}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["all", "check"])
    ap.add_argument("--experiment", default=f"mobile_{START}_20261110")
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS), help="comma list from " + ",".join(MOBILE_SPEC))
    ap.add_argument("--smoke-only", action="store_true")
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", a.experiment):
        ap.error("--experiment: letters, digits, _ or - only")
    runs = [r for r in a.runs.split(",") if r]
    if [r for r in runs if r not in MOBILE_SPEC]:
        ap.error(f"unknown runs in {runs}")
    if ("MP40" in runs) != ("MP128" in runs):
        ap.error("MP40 and MP128 are a matched pair: run both or neither")
    for r in runs:
        if r in PARTNER and r != "MP40" and PARTNER[r] not in runs:
            ap.error(f"{r} needs its control {PARTNER[r]} in the same experiment")
    configure()
    bv.set_paths(a.experiment)
    if a.step == "check":
        prepare_mobile(check_only=True); return
    patch()
    payload = bv.experiment_payload(runs)
    payload["mobile"] = dict(start=START, script_sha256=bv.sha(Path(__file__)), base_cfg_sha256=bv.sha(MOBILE_CFG),
                             dict_sha256=bv.sha(bv.DICT), t0_export_params_sha256=bv.sha(T0_EXPORT / "inference.pdiparams"),
                             loss_weights=[CTC_W, NRTR_W], lr_mult6=LR_MULT6, extra_se=EXTRA_SE, se_gate_bias=SE_GATE_BIAS)
    bv.initialize_experiment(payload)
    bv.RUNS.mkdir(parents=True, exist_ok=True)
    prepared = prepare_mobile()
    smoke(prepared, runs)
    if a.smoke_only:
        log("smoke test only: stopping here"); return
    results = score_baselines(bv.EVAL_SETS, bv.RUNS)
    for name in runs:
        s = MOBILE_SPEC[name]
        d = bv.train(name, s["geom"], s["pool40"], train_file(name), what=s["what"], opts=opts_of(name))
        results[name] = bv.score_checkpoint(name, d / "best_accuracy", d / "config.yml", d / "scores")
        cer = bv.summarize(results[name])[(name, "whiteboard val")]["mean_line_CER"]
        log(f"{name}: whiteboard val mean line CER {cer:.4f}")
    for name in runs:
        bv.export(name, MOBILE_SPEC[name]["geom"], bv.RESULTS / "models" / f"PP-OCRv5_mobile_rec_board2notes_{name}")
    bv.report(results, prepared, bv.RESULTS, comparisons=[c for c in COMPARISONS if all(
        m.removesuffix("avg") in runs or m in ("start", "T0export") for m in c[:2])], test_sets=TEST_SETS)
    mobile_sections(results, bv.RESULTS)
    pack()
    log("DONE")


if __name__ == "__main__":
    main()
