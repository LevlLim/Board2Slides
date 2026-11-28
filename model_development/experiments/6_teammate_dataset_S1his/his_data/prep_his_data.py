"""Download the teammate's Hugging Face dataset (TYW02/whiteboard-notes-v1, pinned revision), fix its train/test
leakage by grouping the same writing onto one side, and write the lists for V2his / V3his.

Rule: crops whose label text is the same within the same source (same board series, JH, Jy, phone photos) are one
group; a JH whole image (JHn) joins its sub-crops (JHn.x); every augmented copy follows its crop. Each group goes
wholly to train or wholly to test. Per source the test keeps about the same number of crops as the teammate's
split, preferring groups that are already mostly in test (fewest moves). Seeded, so the result is reproducible.

Default (--assign val-from-leaks): groups his split had on both sides of train/test become a validation set.
Outputs (OUT/lists): his_train_full.txt (V3his), his_train_orig.txt (V2his), his_val_orig.txt (checkpoint selection),
his_test_orig.txt (test, both runs), his_*_full.txt (all files, reference), moves.csv, prep_report.json.
Paths are relative to OUT/raw.
"""
import argparse, collections, csv, difflib, hashlib, json, os, random, re, shutil, sys
from pathlib import Path

REPO = 'TYW02/whiteboard-notes-v1'
REVISION = '75619db23eb950d86d18e87e557954ca9a48cf6f'  # his 10 Nov 16:20 SGT update (grouped by label)
KNOWN = {'75619db23eb950d86d18e87e557954ca9a48cf6f': {'train': 8327, 'test': 2119},
         '896d42ecd25681dbeb5976945e24109817a3b5bc': {'train': 8374, 'test': 2072}}
PIX_LINK = 0.85
# Reviewed by eye on 10 Nov (version 75619db2): these label pairs are different words on the board, not one writing
# cropped differently, so the near/contained-label rule must not join them.
DIFFERENT_WRITING = {frozenset(x) for x in [
    ('chapter1', 'chapter2'), ('chapter1', 'chapter3'), ('chapter1', 'chapter4'), ('chapter1', 'chapter5'),
    ('chapter2', 'chapter3'), ('chapter2', 'chapter4'), ('chapter2', 'chapter5'), ('chapter3', 'chapter4'),
    ('chapter3', 'chapter5'), ('chapter4', 'chapter5'),
    ('classification', '1classification'), ('classification', '2binaryclassification'),
    ('classification', '3multiclassclassification'), ('convolution', 'convolutionneuralnetworks'),
    ('convolution', 'convolutionneuralnetworkspadding'), ('kernel', 'kernelkl')]}
NO_NEAR_LINK_SOURCES = {'Jy'}  # one writer's separate word cards: similar words (Linoleic/Linolenic) are separate writing
AUG = r'_(dim|bright|blur|rotated\d*|blemish|zoomout|zoomin|scale|perspective\d*|glare|cropped|clean\d*)$'


def parse(p):
    """file path -> (crop key, variant). 'plain' and 'original' (an exact copy) are unaugmented; 'cropped'/'clean' are
    re-cropped unaugmented versions used only when a crop has no plain file; the rest are augmented copies."""
    b = re.sub(r'\.[A-Za-z]+$', '', p.split('/')[-1]); v = 'plain'
    m = re.match(r'^(lighting|original|geometry)_', b)
    if m: v = m.group(1); b = b[m.end():]
    if re.search(r'_[0-9a-f]{8}$', b):
        b = re.sub(r'_[0-9a-f]{8}$', '', b); v = v if v != 'plain' else 'hash'
    m = re.search(AUG, b)
    if m: v = m.group(1); b = b[:m.start()]
    return b, v


def source(k):
    if k.upper().startswith('JH'): return 'JH'
    if re.match(r'^\d+(\.\d+)+$', k): return 'Jy'
    if re.match(r'^\d{12,}', k): return 'phone'
    return re.sub(r'[_\d].*', '', k) or 'other'


norm = lambda s: ' '.join(s.lower().split())
loose = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())  # ignores spacing and punctuation: '- In-class variation.'
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()


def load(raw, k):
    out = []
    for i, l in enumerate(open(raw / f'rec_gt_{k}.txt', encoding='utf-8'), 1):
        l = l.rstrip('\r\n')
        if not l.strip(): continue
        parts = l.split('\t')
        if len(parts) != 2: sys.exit(f'rec_gt_{k}.txt:{i} malformed line: {l!r}')
        p = re.sub(r'/+', '/', parts[0].replace('\\', '/'))
        out.append((p, parts[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/workspace/b2n_hisdata')
    ap.add_argument('--raw', help='use an existing download instead of downloading (folder with train/, test/, lists)')
    ap.add_argument('--seed', type=int, default=20261110)
    ap.add_argument('--assign', choices=['stratified', 'val-from-leaks', 'majority', 'balance'], default='stratified',
                    help='stratified (default): whole groups of the same writing split train:val:test by --ratio within '
                    'every source x line-length band; val-from-leaks: every group of the same writing found on both sides of his split '
                    'moves wholly into a new validation set, the rest keep his side; majority: fewest moves between '
                    'train and test; balance: per-source test size (used for 896d42ec)')
    ap.add_argument('--ratio', default='80:10:10', help='train:val:test crop ratio for val-from-leaks; "none" keeps '
                    'his split apart from the leaked groups')
    ap.add_argument('--revision', default=REVISION, help='Hugging Face revision (default: his 10 Nov 16:20 update)')
    ap.add_argument('--fixed-copy', help='also write a copy of the dataset with images physically in their new '
                    'train/ or test/ folder and fixed rec_gt_train.txt / rec_gt_test.txt (his format, all copies)')
    a = ap.parse_args()
    out = Path(a.out); lists = out / 'lists'; lists.mkdir(parents=True, exist_ok=True)
    raw = Path(a.raw) if a.raw else out / 'raw'
    if not a.raw:
        from huggingface_hub import snapshot_download
        print(f'downloading {REPO} @ {a.revision[:8]} -> {raw}', flush=True)
        snapshot_download(repo_id=REPO, repo_type='dataset', revision=a.revision, local_dir=str(raw), max_workers=16,
                          allow_patterns=['train/*', 'test/*', 'rec_gt_train.txt', 'rec_gt_test.txt'])

    L = {k: load(raw, k) for k in ('train', 'test')}
    for k, n in KNOWN.get(a.revision, {}).items():
        if len(L[k]) != n: sys.exit(f'rec_gt_{k}.txt has {len(L[k])} lines, expected {n}: not revision {a.revision[:8]}?')
    missing = [p for k in L for p, _ in L[k] if not (raw / p).is_file()]
    if missing: sys.exit(f'{len(missing)} listed images are missing, e.g. {missing[:3]}')

    # crops: key -> its files, label, original side
    crop = {}
    for k in ('train', 'test'):
        for p, t in L[k]:
            key, v = parse(p)
            c = crop.setdefault(key, {'files': [], 'label': t, 'sides': set(), 'var': {}})
            c['files'].append((p, t)); c['sides'].add(k); c['var'].setdefault(v, p)
    # groups (union-find)
    par = {k: k for k in crop}
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    def u(x, y): par[f(x)] = f(y)
    by = collections.defaultdict(list)
    for k, c in crop.items(): by[(source(k), norm(c['label']))].append(k)
    for ks in by.values():
        for x in ks[1:]: u(ks[0], x)
    # near-identical labels in the same source are the same writing with a labelling slip ('so had.' / 'so hard.') or a
    # crop cut at the edge ('he badness of our model'); a label contained in another is a crop that holds that line
    bys = collections.defaultdict(dict)
    for k, c in crop.items(): bys[source(k)].setdefault(loose(c['label']), []).append(k)
    near = []
    for s_, d in bys.items():
        labs = sorted(d)
        for x in labs:
            for ks in d[x][1:]: u(d[x][0], ks)
        for i, x in enumerate(labs):
            if s_ in NO_NEAR_LINK_SOURCES: break
            for y in labs[i + 1:]:
                if frozenset((x, y)) in DIFFERENT_WRITING: continue
                close = min(len(x), len(y)) >= 5 and abs(len(x) - len(y)) <= 3 and difflib.SequenceMatcher(None, x, y).ratio() >= 0.85
                inside = min(len(x), len(y)) >= 10 and (x in y or y in x)  # one crop holds the other's line
                if close or inside:
                    u(d[x][0], d[y][0]); near.append((s_, crop[d[x][0]]['label'], crop[d[y][0]]['label']))
    for k in crop:
        m = re.match(r'^(JH\d+)\.\d+', k, re.I)
        if m and m.group(1) in crop: u(k, m.group(1))
    # same writing under a different (wrong) label: link crops of the same source whose unaugmented images are
    # near-identical (pixel similarity >= PIX_LINK). Different words score <= 0.84 on this data; the same writing under wrong labels 0.85-0.9.
    pix_links = []
    try:
        import numpy as np
        from PIL import Image
        def b0(c):
            v = c['var']; return v.get('plain') or v.get('original')
        for s_ in sorted({source(k) for k in crop}):
            ks = [k for k in sorted(crop) if source(k) == s_ and b0(crop[k])]
            if len(ks) < 2: continue
            X = np.stack([np.asarray(Image.open(raw / b0(crop[k])).convert('L').resize((128, 32)), np.float32).ravel() for k in ks])
            X -= X.mean(1, keepdims=True); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-6
            S = X @ X.T; np.fill_diagonal(S, 0)
            for i, j in zip(*np.where(np.triu(S) >= PIX_LINK)):
                if f(ks[i]) != f(ks[j]):
                    pix_links.append([s_, ks[i], crop[ks[i]]['label'], ks[j], crop[ks[j]]['label'], round(float(S[i, j]), 3)])
                    u(ks[i], ks[j])
    except ImportError:
        pix_links = 'skipped: numpy/pillow not installed'
    G = collections.defaultdict(list)
    for k in crop: G[f(k)].append(k)
    # side per group
    rng = random.Random(a.seed); side = {}
    if a.assign == 'stratified':
        stratified(a.ratio, G, crop, side, random.Random(a.seed))
    elif a.assign == 'val-from-leaks':
        for g in G.values():
            sides = set().union(*(crop[k]['sides'] for k in g))
            s_ = 'val' if len(sides) > 1 else next(iter(sides))
            for k in g: side[k] = s_
        if a.ratio != 'none':
            rebalance(a.ratio, G, crop, side, random.Random(a.seed))
    elif a.assign == 'majority':
        # fewest moves: a group already on one side stays there; a mixed group goes to the side holding most of it
        for g in G.values():
            nt = sum('test' in crop[k]['sides'] for k in g)
            s_ = 'test' if nt * 2 >= len(g) and nt > 0 else 'train'
            for k in g: side[k] = s_
    else:
        # balance: per source keep about his number of test crops, preferring groups already mostly in test
        for s in sorted({source(k) for k in crop}):
            gs = [sorted(g) for g in G.values() if source(g[0]) == s]; gs.sort()
            target = sum(1 for k in crop if source(k) == s and 'test' in crop[k]['sides'])
            rng.shuffle(gs)
            gs.sort(key=lambda g: -sum('test' in crop[k]['sides'] for k in g) / len(g))
            n = 0; chosen = set()
            for i, g in enumerate(gs):
                if n >= target: break
                if n + len(g) > target * 1.25 and n > 0: continue
                chosen.add(i); n += len(g)
            for i, g in enumerate(gs):
                for k in g: side[k] = 'test' if i in chosen else 'train'

    # lists
    def base(c):
        v = c['var']
        return v.get('plain') or v.get('original') or next((v[x] for x in sorted(v) if x.startswith(('cropped', 'clean'))), None)
    W = lambda name, rows: (lists / name).write_text(''.join(f'{p}\t{t}\n' for p, t in rows), encoding='utf-8')
    tr = sorted(k for k in crop if side[k] == 'train'); te = sorted(k for k in crop if side[k] == 'test')
    va = sorted(k for k in crop if side[k] == 'val')
    W('his_train_full.txt', [x for k in tr for x in sorted(crop[k]['files'])])
    W('his_train_orig.txt', [(base(crop[k]), crop[k]['label']) for k in tr if base(crop[k])])
    W('his_test_orig.txt', [(base(crop[k]), crop[k]['label']) for k in te if base(crop[k])])
    W('his_test_full.txt', [x for k in te for x in sorted(crop[k]['files'])])
    if va:
        W('his_val_orig.txt', [(base(crop[k]), crop[k]['label']) for k in va if base(crop[k])])
        W('his_val_full.txt', [x for k in va for x in sorted(crop[k]['files'])])
    gid = {k: f(k) for k in crop}
    with open(lists / 'moves.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(['crop', 'label', 'source', 'was', 'now', 'group', 'files'])
        for k in sorted(crop):
            was = '+'.join(sorted(crop[k]['sides']))
            if was != side[k]: w.writerow([k, crop[k]['label'], source(k), was, side[k], gid[k], len(crop[k]['files'])])

    # checks
    grp_both = sum(1 for g in G.values() if len({side[k] for k in g}) > 1)
    st = collections.defaultdict(set)
    for k in crop: st[(source(k), loose(crop[k]['label']))].add(side[k])
    text_both = sum(1 for v in st.values() if len(v) > 1)
    cross = collections.defaultdict(set)
    for k in crop: cross[norm(crop[k]['label'])].add((source(k), side[k]))
    cross_src = sorted(t for t, v in cross.items() if len({s for _, s in v}) > 1)
    no_base = sorted(k for k in crop if not base(crop[k]))
    tab = {}
    for s in sorted({source(k) for k in crop}):
        ks = [k for k in crop if source(k) == s]
        tab[s] = dict(test_before=sum('test' in crop[k]['sides'] for k in ks), test_after=sum(side[k] == 'test' for k in ks),
                      val=sum(side[k] == 'val' for k in ks),
                      moved_to_test=sum(side[k] == 'test' and crop[k]['sides'] != {'test'} for k in ks),
                      moved_to_train=sum(side[k] == 'train' and crop[k]['sides'] != {'train'} for k in ks))
    R = dict(repo=REPO, revision=a.revision, seed=a.seed, assign=a.assign, ratio=a.ratio,
             list_sha256={k: sha(raw / f'rec_gt_{k}.txt') for k in ('train', 'test')},
             files=dict(train=len(L['train']), test=len(L['test'])), crops=len(crop), groups=len(G),
             crops_on_both_sides_before=sum(len(c['sides']) > 1 for c in crop.values()),
             out=dict(train_crops=len(tr), test_crops=len(te), val_crops=len(va),
                      his_val_orig=sum(1 for k in va if base(crop[k])),
                      his_train_full=sum(len(crop[k]['files']) for k in tr), his_train_orig=sum(1 for k in tr if base(crop[k])),
                      his_test_orig=sum(1 for k in te if base(crop[k])), his_test_full=sum(len(crop[k]['files']) for k in te)),
             near_label_links=near, pixel_links=pix_links, per_source=tab, crops_without_unaugmented_original=no_base,
             checks=dict(groups_on_both_sides=grp_both, same_text_same_source_on_both_sides=text_both,
                         same_text_different_source_on_both_sides=cross_src))
    # pixel check: every test original against every train original; same writing would score high (about 0.9+)
    try:
        import numpy as np
        from PIL import Image
        def F(rows):
            X = np.stack([np.asarray(Image.open(raw / p).convert('L').resize((128, 32)), np.float32).ravel() for p, _ in rows])
            X -= X.mean(1, keepdims=True); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-6; return X
        rows_of = lambda ks: [(base(crop[k]), crop[k]['label']) for k in ks if base(crop[k])]
        sets = {'train': rows_of(tr), 'test': rows_of(te), 'val': rows_of(va)}
        X = {k: F(v) for k, v in sets.items() if v}
        R['pixel_check'] = {}; n9 = 0
        for x, y in (('test', 'train'), ('val', 'train'), ('test', 'val')):
            if x not in X or y not in X: continue
            S = X[x] @ X[y].T; j = S.argmax(1); m = S.max(1); n9 += int((m >= 0.9).sum())
            R['pixel_check'][f'{x}_vs_{y}'] = dict(median=round(float(np.median(m)), 3), max=round(float(m.max()), 3),
                n_at_least_0_9=int((m >= 0.9).sum()),
                top=[[round(float(m[i]), 3), sets[x][i][0], sets[x][i][1], sets[y][j[i]][0], sets[y][j[i]][1]] for i in np.argsort(-m)[:10]])
            print(f'PIXEL CHECK {x} vs {y}: median', R['pixel_check'][f'{x}_vs_{y}']['median'], 'max', R['pixel_check'][f'{x}_vs_{y}']['max'],
                  '| pairs >= 0.9:', R['pixel_check'][f'{x}_vs_{y}']['n_at_least_0_9'])
        R['pixel_check']['n_at_least_0_9'] = n9
    except ImportError:
        R['pixel_check'] = 'skipped: numpy/pillow not installed'
    (lists / 'prep_report.json').write_text(json.dumps(R, indent=1, ensure_ascii=False) + '\n')
    if a.fixed_copy:
        R['fixed_copy'] = write_fixed(Path(a.fixed_copy), raw, crop, side, lists, a.revision)
        (lists / 'prep_report.json').write_text(json.dumps(R, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({k: R[k] for k in ('revision', 'files', 'crops', 'groups', 'crops_on_both_sides_before', 'out')}, indent=1))
    print(f"{'source':11s} {'test before':>11s} {'test after':>10s} {'val':>5s} {'to test':>8s} {'to train':>9s}")
    for s, d in tab.items(): print(f"{s:11s} {d['test_before']:11d} {d['test_after']:10d} {d['val']:5d} {d['moved_to_test']:8d} {d['moved_to_train']:9d}")
    print('CHECKS', R['checks'])
    ok = grp_both == 0 and text_both == 0 and not (isinstance(R['pixel_check'], dict) and R['pixel_check']['n_at_least_0_9'])
    print('PREP OK' if ok else 'PREP FAILED: same writing is in more than one set (see prep_report.json)')
    sys.exit(0 if ok else 1)


def band(t):
    n = len(t)
    return 'short' if n <= 20 else ('medium' if n <= 40 else 'long')


def stratified(ratio, G, crop, side, rng):
    """Whole groups of the same writing go to train/val/test so that every (source, line-length band) stratum is
    split by the ratio. A group's stratum is its source and the band most of its crops fall in. Within a stratum the
    groups are shuffled (seeded) and each goes to the set furthest below its target share; ties prefer the set the
    teammate's split already had it in, so his arrangement is kept where the ratio allows."""
    w = [float(x) for x in ratio.split(':')]; names = ('train', 'val', 'test'); share = [x / sum(w) for x in w]
    groups = sorted((sorted(g) for g in G.values()), key=lambda g: g[0])
    strata = collections.defaultdict(list)
    for g in groups:
        b = collections.Counter(band(crop[k]['label']) for k in g).most_common(1)[0][0]
        strata[(source(g[0]), b)].append(g)
    for key in sorted(strata):
        gs = strata[key]; rng.shuffle(gs); gs.sort(key=len, reverse=True)  # big groups first, then fill
        tot = sum(len(g) for g in gs); have = dict.fromkeys(names, 0)
        for g in gs:
            was = collections.Counter(s for k in g for s in crop[k]['sides'])
            gap = {n: share[i] * tot - have[n] for i, n in enumerate(names)}
            best = max(names, key=lambda n: (round(gap[n] - len(g) / 2, 6), was.get(n, 0)))
            have[best] += len(g)
            for k in g: side[k] = best


def rebalance(ratio, G, crop, side, rng):
    """Reach the train:val:test ratio by moving whole TEST groups out of test (never into it), so no writing is split.
    Test keeps the same share of each source (stratified). Groups leaving test fill validation first, taking sources
    validation lacks (validation from the leaks is mostly oct17), then the rest go to train."""
    w = [float(x) for x in ratio.split(':')]; n = len(crop)
    t_val, t_test = round(n * w[1] / sum(w)), round(n * w[2] / sum(w))
    groups = sorted((sorted(g) for g in G.values()), key=lambda g: g[0])
    test_g = [g for g in groups if side[g[0]] == 'test']
    n_test = sum(len(g) for g in test_g)
    if n_test <= t_test: return
    out = []
    for s in sorted({source(g[0]) for g in test_g}):
        gs = [g for g in test_g if source(g[0]) == s]; rng.shuffle(gs)
        have = sum(len(g) for g in gs); keep = round(have * t_test / n_test)
        for g in sorted(gs, key=len):  # small groups first: finer steps toward the target
            if have - len(g) >= keep: out.append(g); have -= len(g)
    n_val = sum(side[k] == 'val' for k in crop)
    val_src = collections.Counter(source(k) for k in crop if side[k] == 'val')
    out.sort(key=lambda g: (val_src[source(g[0])], rng.random()))  # sources validation lacks go first
    for g in out:
        to = 'val' if n_val + len(g) <= t_val else 'train'
        if to == 'val': n_val += len(g); val_src[source(g[0])] += len(g)
        for k in g: side[k] = to


def write_fixed(dst, raw, crop, side, lists, revision):
    """Copy every image into dst/<new side>/<same file name>; write his two gt files for the new split. Resumable."""
    rows = {'train': [], 'test': [], 'val': []}; plan = {}
    for k in sorted(crop):
        for p, t in sorted(crop[k]['files']):
            new = f"{side[k]}/{p.split('/')[-1]}"
            if new in plan and plan[new] != p: sys.exit(f'name clash in fixed copy: {p} and {plan[new]} -> {new}')
            plan[new] = p; rows[side[k]].append((new, t))
    SETS = [d for d in ('train', 'test', 'val') if rows[d]]
    for d in SETS: (dst / d).mkdir(parents=True, exist_ok=True)
    n = 0
    for new, p in plan.items():
        q = dst / new
        if q.exists() and q.stat().st_size == (raw / p).stat().st_size: continue
        shutil.copyfile(raw / p, q); n += 1
    stray = [str(x.relative_to(dst)) for d in SETS for x in (dst / d).iterdir() if f'{d}/{x.name}' not in plan]
    if stray: sys.exit(f'{len(stray)} files in the fixed copy are not in the fixed lists, e.g. {stray[:3]}')
    for d in SETS:
        (dst / f'rec_gt_{d}.txt').write_text(''.join(f'{p}\t{t}\n' for p, t in rows[d]), encoding='utf-8')
    shutil.copyfile(lists / 'moves.csv', dst / 'moves.csv')
    (dst / 'README_FIX.md').write_text(
        f'# whiteboard-notes-v1 @ {revision[:8]}, leakage-fixed split\n\nSame images and labels as the original. '
        f'{sum(1 for k in crop if crop[k]["sides"] != {side[k]})} crops (with all their augmented copies) were moved so '
        'that each piece of writing (the same label in the same source, including repeat photos of the same board and '
        'a JH whole image with its sub-crops) is wholly in one set. Groups that his split had on both sides of train/test '
        'now form val/ (rec_gt_val.txt), a validation set for checkpoint selection. moves.csv lists every moved crop. '
        'Made by prep_his_data.py (seed 20261110).\n', encoding='utf-8')
    print(f'fixed copy: {dst} (' + ', '.join(f'{d} {len(rows[d])}' for d in SETS) + f' files; {n} copied this run)')
    return dict(path=str(dst), **{f'{d}_files': len(rows[d]) for d in SETS},
                **{f'rec_gt_{d}_sha256': sha(dst / f'rec_gt_{d}.txt') for d in SETS})


if __name__ == '__main__':
    main()
