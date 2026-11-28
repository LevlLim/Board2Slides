"""Where in the server reader did turning guidance off (the CTC gradient reaching the backbone) change things?
Local, from the exported models, no pod. Parts:
  weights  how far each part moved from pretrained; guidance vs seed distance; same-direction cosine
  swap     transplant: S0 with one or more parts taken from G0 (and seed controls), read the crops, count error types
  cka      stage-output similarity between readers on the same crops
"""
import sys, csv, glob, collections, time
sys.path.insert(0, '/home/claude/sweep')
from layers_common import *
sys.path.insert(0, '/home/claude/fresh/pkg/b2n_ocr'); os.environ['B2N_PACKAGE_ROOT'] = '/home/claude/fresh/pkg'
import ocr_backend as ob, cv2
paddle.set_device('cpu')
OUT = '/home/claude/sweep/layers_out'; os.makedirs(OUT, exist_ok=True)
part = sys.argv[1]
W = {m: named(m) for m in ('pretrained', 'S0', 'S0s2', 'G0', 'G0s2')}
keys = [k for k in W['S0'] if group(k)]
stat = lambda k: k.endswith('._mean') or k.endswith('._variance')

# ---------------- crops
def crops():
    S = []
    nw = list(csv.DictReader(open('/home/claude/diag/nw/b2n_data/newwriter/manifest.csv', encoding='utf-8')))
    S += [('new writers', f"/home/claude/diag/nw/b2n_data/newwriter/{r['path']}", r['label']) for r in nw]
    FR = '/tmp/claude-0/-home-claude/02f3df2f-1dbb-533d-b187-71169cf31ef0/scratchpad/unf/board2notes_freshwriter_test_20261111'
    S += [('fresh boards', f"{FR}/{r['file']}", r['label']) for r in csv.DictReader(open(FR + '/manifest.csv', encoding='utf-8')) if r['included'] == 'yes']
    P = f'{U}/results_strategies_s0s3/results/predictions.csv'
    S += [('whiteboard val', f"{U}/board2notes_reader_dataset_20261105/images/wb/{r['file'].split('__')[-1]}", r['label'])
          for r in csv.DictReader(open(P, encoding='utf-8')) if r['set'] == 'whiteboard val' and r['model'] == 'S0']
    return S

def ops(a, b):
    # substitutions, deletions, insertions of reading b against label a
    n, m = len(a), len(b); D = np.zeros((n + 1, m + 1), int); D[:, 0] = range(n + 1); D[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    i, j, s, d, ins = n, m, 0, 0, 0
    while i or j:
        if i and j and D[i, j] == D[i - 1, j - 1] + (a[i - 1] != b[j - 1]): s += a[i - 1] != b[j - 1]; i -= 1; j -= 1
        elif i and D[i, j] == D[i - 1, j] + 1: d += 1; i -= 1
        else: ins += 1; j -= 1
    return s, d, ins

norm = lambda t: ' '.join(t.lower().split())

if part == 'weights':
    rows = []
    for g in GROUPS:
        ks = [k for k in keys if group(k) == g and not stat(k)]
        f = lambda m: np.concatenate([W[m][k].ravel().astype(np.float64) for k in ks])
        pre, s0, s0b, g0, g0b = (f(m) for m in ('pretrained', 'S0', 'S0s2', 'G0', 'G0s2'))
        nb = np.linalg.norm(pre); d = lambda a, b: np.linalg.norm(a - b) / nb
        e1, e2 = g0 - s0, g0b - s0b
        rows.append(dict(part=g, params=int(pre.size), moved_S0=d(s0, pre), moved_S0s2=d(s0b, pre), moved_G0=d(g0, pre), moved_G0s2=d(g0b, pre),
                         guidance=(d(s0, g0) + d(s0b, g0b)) / 2, seed=(d(s0, s0b) + d(g0, g0b)) / 2,
                         same_direction=float(e1 @ e2 / (np.linalg.norm(e1) * np.linalg.norm(e2))),
                         seed_direction=float((s0b - s0) @ (g0b - g0) / (np.linalg.norm(s0b - s0) * np.linalg.norm(g0b - g0)))))
    json.dump(rows, open(f'{OUT}/weights.json', 'w'), indent=1)
    for r in rows: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

if part == 'swap':
    S = crops(); print(collections.Counter(s for s, _, _ in S), flush=True)
    m, post = build('S0'); shape = [3, 64, 1024]
    X = [ob.resize_for_reader(cv2.imread(p), shape) for _, p, _ in S]
    def read(wd):
        m.set_state_dict({k: paddle.to_tensor(v) for k, v in wd.items()}); m.eval(); out = []
        with paddle.no_grad():
            for x in X: out.append(post(m(paddle.to_tensor(x[None])))[0][0])
        return out
    def hybrid(base, donor, parts):
        d = dict(W[base]); d.update({k: W[donor][k] for k in keys if group(k) in parts}); return d
    early = lambda n: GROUPS[:n]
    plan = [('S0', 'S0', []), ('G0', 'G0', [])]
    plan += [('S0', 'G0', [g]) for g in GROUPS]
    plan += [('S0', 'G0', early(n)) for n in (2, 3, 4, 5)]
    plan += [('S0', 'G0', GROUPS[n:]) for n in (1, 3, 5)]
    plan += [('S0', 'S0s2', [g]) for g in GROUPS]          # seed control: same swap from a different seed of S0
    plan += [('G0', 'S0', [g]) for g in GROUPS]            # reverse: put S0's part into G0
    plan += [('S0s2', 'G0s2', [g]) for g in GROUPS]        # the same guidance swap in seed 2
    res = json.load(open(f'{OUT}/swap.json')) if os.path.exists(f'{OUT}/swap.json') else []
    done = {(r['base'], r['donor'], tuple(r['parts'])) for r in res}
    for base, donor, parts in plan:
        if (base, donor, tuple(parts)) in done: continue
        t = time.time(); pr = read(hybrid(base, donor, parts))
        per = collections.defaultdict(lambda: [0, 0, 0, 0])
        for (s, _, lab), p in zip(S, pr):
            a = norm(lab); o = ops(a, norm(p)); c = per[s]
            c[0] += o[0]; c[1] += o[1]; c[2] += o[2]; c[3] += len(a)
        res.append(dict(base=base, donor=donor, parts=parts, sets={s: dict(sub=v[0], dele=v[1], ins=v[2], chars=v[3]) for s, v in per.items()}, preds=pr))
        json.dump(res, open(f'{OUT}/swap.json', 'w'))
        print(base, '<-', donor, parts, {s: round(100 * (v[0] + v[1] + v[2]) / v[3], 2) for s, v in per.items()}, f'{time.time() - t:.0f}s', flush=True)
    print('SWAP DONE', flush=True)

if part == 'cka':
    S = crops(); shape = [3, 64, 1024]; rng = np.random.default_rng(0)
    X = {s: [ob.resize_for_reader(cv2.imread(p), shape) for ss, p, _ in S if ss == s] for s in ('whiteboard val', 'new writers', 'fresh boards')}
    groups_img = {'familiar (whiteboard val)': X['whiteboard val'], 'unfamiliar (new writers + fresh)': X['new writers'] + X['fresh boards']}
    feats, idx = {}, {}
    for mid in ('pretrained', 'S0', 'S0s2', 'G0', 'G0s2'):
        m, _ = build(mid)
        for gname, xs in groups_img.items():
            st = [[] for _ in range(5)]
            with paddle.no_grad():
                for x in xs:
                    h = m.backbone.stem(paddle.to_tensor(x[None])); outs = [h]
                    for s in m.backbone.stages: h = s(h); outs.append(h)
                    for i, o in enumerate(outs): st[i].append(o.mean(axis=2)[0].transpose([1, 0]).numpy())
            for i in range(5):
                a = np.concatenate(st[i], 0)
                if (gname, i) not in idx: idx[(gname, i)] = np.sort(rng.choice(len(a), min(6000, len(a)), replace=False))
                feats[(mid, gname, i)] = a[idx[(gname, i)]].astype(np.float64)
        print(mid, 'features done', flush=True)
    def cka(x, y):
        x = x - x.mean(0); y = y - y.mean(0)
        return float(np.linalg.norm(x.T @ y) ** 2 / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))
    rows = []
    for gname in groups_img:
        for i, sname in enumerate(['stem', 'stage 1', 'stage 2', 'stage 3', 'stage 4']):
            r = dict(images=gname, stage=sname)
            for a, b in (('S0', 'G0'), ('S0s2', 'G0s2'), ('S0', 'S0s2'), ('G0', 'G0s2'), ('pretrained', 'S0'), ('pretrained', 'G0')):
                r[f'{a}~{b}'] = cka(feats[(a, gname, i)], feats[(b, gname, i)])
            sh = feats[('G0', gname, i)][rng.permutation(len(feats[('G0', gname, i)]))]
            r['control'] = cka(feats[('S0', gname, i)], sh)
            rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    json.dump(rows, open(f'{OUT}/cka.json', 'w'), indent=1)
