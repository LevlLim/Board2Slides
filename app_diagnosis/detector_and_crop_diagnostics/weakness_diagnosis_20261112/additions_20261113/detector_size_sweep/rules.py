import json, collections, numpy as np, sys
from common import line_h
rows = json.load(open('rows_S1.json')); det = json.load(open('det.json'))
R = collections.defaultdict(dict)
for r in rows: R[(r['vid'], r['cfg'])][r['id']] = r
def h_at(vid, cfg, size):
    rec = det[vid]; b = rec['cfg'].get(cfg) or rec['cfg'].get('full') or rec['cfg'].get('4000')
    return float(np.median([line_h(q) for q in b])) * min(1, size / rec['long']) if b else 0.0
def cfgname(rec, size):
    if size >= rec['long']: return 'full'
    return str(size) if str(size) in rec['cfg'] or size in (640, 960, 1280, 1600, 2400) else 'full'
SIZES = [640, 960, 1280, 1600, 2400, 4000]
def nearest(rec, want):
    best = min(SIZES, key=lambda s: abs(np.log(s / want)))
    return 'full' if best >= rec['long'] or best == 4000 else str(best)
def rule(name, vid):
    rec = det[vid]; small_path = 'small text' in rec['info']['path']
    if name == 'app': return 'app'
    if name == '960': return '960' if rec['long'] > 960 else 'full'
    h = h_at(vid, '960' if '960' in rec['cfg'] else 'full', 960)
    if name.startswith('A'):                 # app structure, quick pass at 960, threshold T (px at 960)
        T = float(name[1:])
        if h >= T: return '960' if rec['long'] > 960 else 'full'
        return 'app' if small_path else 'full'
    if name == 'B':                          # aim lines at ~48 px: keep 960 inside 32-96, otherwise rescale
        if 32 <= h <= 96: return '960' if rec['long'] > 960 else 'full'
        return nearest(rec, 960 * 48 / max(h, 1))
groups = sorted({r['group'] for r in rows}, key=lambda g: ('zoomed' in g, g))
names = ['app', '960', 'A12', 'A16', 'A20', 'A24', 'A32', 'B']
out = {}
print('| Rule | ' + ' | '.join(groups) + ' | mean of groups |'); print('|---|' + '---|' * (len(groups) + 1))
for n in names:
    cells = []
    for g in groups:
        vids = sorted({r['vid'] for r in rows if r['group'] == g})
        x = [v['cer'] for vid in vids for v in R[(vid, rule(n, vid))].values() if v['group'] == g]
        cells.append(100 * np.mean(x))
    out[n] = cells
    print(f'| {n} | ' + ' | '.join(f'{c:.1f}' for c in cells) + f' | {np.mean(cells):.1f} |')
print()
for vid in sorted(det):
    if any(k in vid for k in ('IMG_5095@1', 'IMG_5122@1', 'IMG_5122@3', 'JH1@4', 'Jy8@1', 'Jy8@2')):
        print(vid, 'line height at 960:', round(h_at(vid, '960' if '960' in det[vid]['cfg'] else 'full', 960), 1), '| A24 ->', rule('A24', vid), '| app path:', det[vid]['info']['path'][:45])

print('\n## Rule C: keep the app path unless the lines are big at full resolution; then shrink so lines are ~TARGET px')
def ruleC(vid, tbig, target):
    rec = det[vid]
    h960 = h_at(vid, '960' if '960' in rec['cfg'] else 'full', 960)
    hfull = h960 * rec['long'] / min(960, rec['long'])
    if hfull <= tbig: return 'app'
    return nearest(rec, rec['long'] * target / hfull)
print('| Rule | ' + ' | '.join(groups) + ' | mean of groups |'); print('|---|' + '---|' * (len(groups) + 1))
for tbig in (96, 128, 160, 200):
    for target in (48, 64, 96):
        cells = []
        for g in groups:
            vids = sorted({r['vid'] for r in rows if r['group'] == g})
            x = [v['cer'] for vid in vids for v in R[(vid, ruleC(vid, tbig, target))].values() if v['group'] == g]
            cells.append(100 * np.mean(x))
        print(f'| C big>{tbig} target {target} | ' + ' | '.join(f'{c:.1f}' for c in cells) + f' | {np.mean(cells):.1f} |')
