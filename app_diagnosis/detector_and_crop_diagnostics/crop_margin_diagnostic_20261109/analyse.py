import json, random, collections
from pathlib import Path
A = Path.cwd().parent
rows = json.load(open("boxes_final.json")); P = json.load(open("PROTOCOL.json"))
po = json.load(open("pred_original.json"))["S1"]; pt = json.load(open("pred_tight.json"))["S1"]
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
    return dict(cer=D[n][m] / max(1, n), S=S, D=Dl, I=I, chars=n)
out = []
for r in rows:
    o, t = ops(r["label"], po[r["src"]]), ops(r["label"], pt[r["tight_file"]])
    out.append(dict(name=r["name"], cluster=r["cluster"], group=r["group"], kept_h=r["kept_h"], label=r["label"],
                    pred_original=po[r["src"]], pred_tight=pt[r["tight_file"]], o=o, t=t, d=t["cer"] - o["cer"]))
# reproducibility of the original arm against the saved RunPod GPU S1 predictions
same = sum(x["pred_original"] == cases[r["file"]]["predictions"]["S1"]["text"] for x, r in zip(out, rows))
gpu = sum(cases[r["file"]]["predictions"]["S1"]["cer"] for r in rows) / len(rows)
def boot(xs, key=lambda x: x["d"], B=4000, seed=0):
    cl = collections.defaultdict(list)
    for x in xs: cl[x["cluster"]].append(x)
    ks = sorted(cl); rnd = random.Random(seed); vals = []
    for _ in range(B):
        s = [x for k in (rnd.choice(ks) for _ in ks) for x in cl[k]]
        vals.append(sum(key(x) for x in s) / len(s))
    vals.sort(); return vals[int(.025 * B)], vals[int(.975 * B) - 1]
def summ(xs):
    n = len(xs)
    return dict(n=n, cer_original=sum(x["o"]["cer"] for x in xs) / n, cer_tight=sum(x["t"]["cer"] for x in xs) / n,
                change=sum(x["d"] for x in xs) / n, ci=boot(xs) if n > 1 else None,
                S=(sum(x["o"]["S"] for x in xs), sum(x["t"]["S"] for x in xs)), D=(sum(x["o"]["D"] for x in xs), sum(x["t"]["D"] for x in xs)),
                I=(sum(x["o"]["I"] for x in xs), sum(x["t"]["I"] for x in xs)),
                better=sum(x["d"] < -1e-12 for x in xs), worse=sum(x["d"] > 1e-12 for x in xs))
groups = {g: [x for x in out if x["group"] == g] for g in ("large margin", "middle", "control", "flagged")}
res = {g: summ(v) for g, v in groups.items()}
res["all 156 (excl. flagged)"] = summ([x for x in out if x["group"] != "flagged"])
# difference in differences (large margin minus control), cluster bootstrap over the union of their clusters
L, C = groups["large margin"], groups["control"]
cl = collections.defaultdict(list)
for x in L + C: cl[x["cluster"]].append(x)
ks = sorted(cl); rnd = random.Random(1); vals = []
for _ in range(4000):
    s = [x for k in (rnd.choice(ks) for _ in ks) for x in cl[k]]
    a = [x["d"] for x in s if x["group"] == "large margin"]; b = [x["d"] for x in s if x["group"] == "control"]
    if a and b: vals.append(sum(a) / len(a) - sum(b) / len(b))
vals.sort(); did = (res["large margin"]["change"] - res["control"]["change"], vals[int(.025 * len(vals))], vals[int(.975 * len(vals)) - 1])
lm = res["large margin"]
c1 = lm["ci"][1] < 0; c2 = lm["D"][1] < lm["D"][0]; c3 = did[2] < 0
verdict = "supported" if (c1 and c2 and c3) else "not supported"
json.dump(dict(protocol=P, reproducibility=dict(identical_to_saved_gpu_S1=same, n=len(out), gpu_mean_cer=gpu),
               groups=res, diff_in_diff=did, criteria=dict(ci_below_0=c1, deletions_down=c2, did_ci_below_0=c3), verdict=verdict,
               crops=out), open("RESULTS.json", "w"), indent=1)
print(f"original arm vs saved GPU S1 readings: {same}/{len(out)} identical; GPU mean CER {gpu:.4f}")
for g, s in res.items():
    ci = f"[{s['ci'][0]*100:+.2f}, {s['ci'][1]*100:+.2f}]" if s["ci"] else ""
    print(f"{g:24s} n={s['n']:3d}  CER {s['cer_original']*100:5.2f} -> {s['cer_tight']*100:5.2f}  change {s['change']*100:+.2f} {ci}  "
          f"S {s['S'][0]}->{s['S'][1]}  D {s['D'][0]}->{s['D'][1]}  I {s['I'][0]}->{s['I'][1]}  better/worse {s['better']}/{s['worse']}")
print(f"diff-in-diff (large - control): {did[0]*100:+.2f} [{did[1]*100:+.2f}, {did[2]*100:+.2f}]")
print("criteria:", dict(ci_below_0=c1, deletions_down=c2, did_ci_below_0=c3), "->", verdict)
