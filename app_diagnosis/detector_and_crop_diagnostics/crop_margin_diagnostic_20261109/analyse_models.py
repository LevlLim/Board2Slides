"""S0/S1/G0 (both seeds) on the frozen 160 original vs tightened JH/DS crops. Reads saved predictions only."""
import json, random, collections, hashlib, platform
from pathlib import Path
A = Path.cwd().parent
rows = json.load(open("boxes_final.json")); P = json.load(open("PROTOCOL.json"))
assert hashlib.sha256(open("boxes_final.json", "rb").read()).hexdigest() == P["boxes_sha256"]
po = {**json.load(open("pred_original.json")), **json.load(open("pred_original_models.json"))}
pt = {**json.load(open("pred_tight.json")), **json.load(open("pred_tight_models.json"))}
MODELS = ["S0", "S0s2", "S1", "S1s2", "G0", "G0s2"]; RECIPES = ["S0", "S1", "G0"]
assert all(m in po and m in pt and len(po[m]) == len(pt[m]) == 160 for m in MODELS)
cases = {c["file"]: c for c in json.load(open(A / "codex_generalisation_audit_20261109/cases.json")) if c["set"] == "new writers"}
norm = lambda s: " ".join(str(s).lower().split())
def ops(a, b):
    a, b = norm(a), norm(b); n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): D[i][0] = i
    for j in range(m + 1): D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i-1][j] + 1, D[i][j-1] + 1, D[i-1][j-1] + (a[i-1] != b[j-1]))
    i, j, S, Dl, I = n, m, 0, 0, 0
    while i or j:
        if i and j and D[i][j] == D[i-1][j-1] + (a[i-1] != b[j-1]): S += a[i-1] != b[j-1]; i -= 1; j -= 1
        elif i and D[i][j] == D[i-1][j] + 1: Dl += 1; i -= 1
        else: I += 1; j -= 1
    return dict(cer=D[n][m] / max(1, n), S=S, D=Dl, I=I)
crops = []
for r in rows:
    x = dict(name=r["name"], file=r["file"], cluster=r["cluster"], group=r["group"], kept_h=r["kept_h"], label=r["label"], m={})
    for m in MODELS:
        x["m"][m] = dict(pred_original=po[m][r["src"]], pred_tight=pt[m][r["tight_file"]],
                         o=ops(r["label"], po[m][r["src"]]), t=ops(r["label"], pt[m][r["tight_file"]]))
    for rc in RECIPES:   # per-crop mean of the two seeds (a statistic, not an ensemble)
        a, b = x["m"][rc], x["m"][rc + "s2"]
        x["m"][rc + "avg"] = {arm: {k: (a[arm][k] + b[arm][k]) / 2 for k in ("cer", "S", "D", "I")} for arm in ("o", "t")}
    crops.append(x)
ALL = crops; EX = [x for x in crops if x["group"] != "flagged"]
def boot_mean(xs, f, B=4000, seed=0):
    cl = collections.defaultdict(list)
    for x in xs: cl[x["cluster"]].append(x)
    ks = sorted(cl); rnd = random.Random(seed); v = []
    for _ in range(B):
        s = [x for k in (rnd.choice(ks) for _ in ks) for x in cl[k]]; v.append(sum(f(x) for x in s) / len(s))
    v.sort(); return [v[int(.025 * B)], v[int(.975 * B) - 1]]
def summ(xs, m):
    n = len(xs); g = lambda arm, k: sum(x["m"][m][arm][k] for x in xs)
    return dict(n=n, cer_original=g("o", "cer") / n, cer_tight=g("t", "cer") / n,
                change=(g("t", "cer") - g("o", "cer")) / n, ci=boot_mean(xs, lambda x: x["m"][m]["t"]["cer"] - x["m"][m]["o"]["cer"]),
                D=[g("o", "D"), g("t", "D")], S=[g("o", "S"), g("t", "S")], I=[g("o", "I"), g("t", "I")],
                better=sum(x["m"][m]["t"]["cer"] < x["m"][m]["o"]["cer"] - 1e-12 for x in xs),
                worse=sum(x["m"][m]["t"]["cer"] > x["m"][m]["o"]["cer"] + 1e-12 for x in xs))
def compare(xs, a, b, arm):
    d = lambda x: x["m"][a][arm]["cer"] - x["m"][b][arm]["cer"]
    return dict(diff=sum(d(x) for x in xs) / len(xs), ci=boot_mean(xs, d),
                better=sum(d(x) < -1e-12 for x in xs), worse=sum(d(x) > 1e-12 for x in xs))
out = dict(models={}, comparisons={}, groups={})
for scope, xs in (("all 160", ALL), ("156 excl. flagged", EX)):
    out["models"][scope] = {m: summ(xs, m) for m in MODELS + [r + "avg" for r in RECIPES]}
    out["comparisons"][scope] = {f"{a} vs {b} | {arm_name}": compare(xs, a, b, arm)
        for arm, arm_name in (("t", "tight"), ("o", "original"))
        for a, b in (("G0", "S0"), ("G0s2", "S0s2"), ("G0avg", "S0avg"), ("S1", "S0"), ("S1s2", "S0s2"), ("S1avg", "S0avg"),
                     ("G0avg", "S1avg"), ("S0s2", "S0"), ("G0s2", "G0"))}
for grp in ("large margin", "middle", "control", "flagged"):
    xs = [x for x in crops if x["group"] == grp]
    out["groups"][grp] = {m: summ(xs, m) for m in [r + "avg" for r in RECIPES] + MODELS}
# remaining errors on tight crops and regressions
out["persistent_tight"] = [x["name"] for x in crops if all(x["m"][m]["t"]["cer"] >= 0.15 for m in MODELS)]
out["regressions"] = {m: [x["name"] for x in crops if x["m"][m]["t"]["cer"] > x["m"][m]["o"]["cer"] + 1e-12] for m in MODELS}
# identities and reproducibility against the saved RunPod GPU readings
dirs = {m: (A / ("results_guide_g0" if m.startswith("G") else "results_strategies_s0s3") / "results/models" / f"PP-OCRv5_server_rec_board2notes_{m}") for m in MODELS}
ident = {}
for m, d in dirs.items():
    v = json.load(open(d / "verified.json"))
    ident[m] = dict(model_dir=str(d.relative_to(A)), inference_pdiparams_sha256=hashlib.sha256((d / "inference.pdiparams").read_bytes()).hexdigest(),
                    checkpoint_sha256=v["identity"]["checkpoint"], config_sha256=v["identity"]["config"], export_h=v["identity"]["export_h"],
                    runpod_export_check=f"{v['fixture_crops']} crops, identical {v['identical_predictions']:.0%}",
                    gpu_readings_identical=sum(po[m][r["src"]] == cases[r["file"]]["predictions"][m]["text"] for r in rows),
                    gpu_mean_cer=sum(cases[r["file"]]["predictions"][m]["cer"] for r in rows) / 160)
import paddle, cv2
out["identity"] = dict(models=ident, scorer="score_static_mac.py", scorer_sha256=hashlib.sha256(open("score_static_mac.py", "rb").read()).hexdigest(),
    runtime=dict(paddle=paddle.__version__, opencv=cv2.__version__, python=platform.python_version(), machine=platform.machine(),
                 device="CPU", ir_optim=False, custom_passes="none"),
    input=dict(height=64, min_width=1024, resize="resize_norm_img_chinese (verbatim from PaddleOCR 91279fee): height 64, width follows the aspect ratio, padded to >= 1024; normalise to [-1, 1]",
               original_arm="untouched source files", tight_arm="lossless PNG crops (boxes_final.json)", decoder="cv2.imread (BGR)"),
    labels="boxes_final.json labels = cases.json labels (unchanged)", boxes_sha256=P["boxes_sha256"])
out["crops"] = crops
json.dump(out, open("RESULTS_MODELS.json", "w"), indent=1)
for scope in ("all 160", "156 excl. flagged"):
    print(f"== {scope}")
    for m in MODELS + [r + "avg" for r in RECIPES]:
        s = out["models"][scope][m]
        print(f"  {m:6s} CER {s['cer_original']*100:5.2f} -> {s['cer_tight']*100:5.2f} ({s['change']*100:+.2f} [{s['ci'][0]*100:+.1f},{s['ci'][1]*100:+.1f}])  D {s['D'][0]:.0f}->{s['D'][1]:.0f}  S {s['S'][0]:.0f}->{s['S'][1]:.0f}  I {s['I'][0]:.0f}->{s['I'][1]:.0f}  better/worse {s['better']}/{s['worse']}")
    for k, v in out["comparisons"][scope].items():
        print(f"  {k:28s} {v['diff']*100:+.2f} [{v['ci'][0]*100:+.2f}, {v['ci'][1]*100:+.2f}] better/worse {v['better']}/{v['worse']}")
print("persistent on tight (>=15% CER in all 6):", len(out["persistent_tight"]))
print({m: len(v) for m, v in out["regressions"].items()})
print({m: (v["gpu_readings_identical"], round(v["gpu_mean_cer"] * 100, 2)) for m, v in ident.items()})
