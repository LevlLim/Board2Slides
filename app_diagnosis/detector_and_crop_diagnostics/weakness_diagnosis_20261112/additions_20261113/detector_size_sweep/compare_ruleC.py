import json, numpy as np
old = json.load(open('rows_S1.json'))
new = json.load(open('rows_S1_ruleC.json'))
rows = [r for r in old if r['cfg'] in ('app', '960')] + [r for r in new if r['cfg'] == 'ruleC']
app_new = {(r['vid'], r['id']): r for r in new if r['cfg'] == 'app'}
app_old = {(r['vid'], r['id']): r for r in old if r['cfg'] == 'app'}
same = sum(abs(app_new[k]['cer'] - app_old[k]['cer']) < 1e-9 for k in app_new if k in app_old)
print(f'app rows re-scored: {len(app_new)}, identical to the sweep: {same}\n')
CF = ['app', '960', 'ruleC']
GR = sorted({r['group'] for r in rows}, key=lambda g: (('zoomed' in g), g))
m = lambda x: 100 * np.mean(x) if len(x) else float('nan')
rng = np.random.default_rng(0)
TRIG = {v for v, x in json.load(open('det_ruleC.json')).items() if x['triggered']}


def boot(d, n=5000):
    d = np.asarray(d)
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)


print('| Photos | lines | app | 960 | rule C | rule C - app (95% interval) | photos where rule C triggered |')
print('|---|---|---|---|---|---|---|')
allc = {c: [] for c in CF}
for g in GR:
    x = {c: {(r['vid'], r['id']): r for r in rows if r['group'] == g and r['cfg'] == c} for c in CF}
    keys = sorted(set(x['app']) & set(x['ruleC']))
    d = [x['ruleC'][k]['cer'] - x['app'][k]['cer'] for k in keys]
    lo, hi = boot(d)
    trig = {k[0] for k in keys} & TRIG
    vids = {k[0] for k in keys}
    for c in CF:
        allc[c] += [x[c][k]['cer'] for k in keys if k in x[c]]
    print(f"| {g} | {len(keys)} | " + ' | '.join(f"{m([x[c][k]['cer'] for k in keys if k in x[c]]):.1f}" for c in CF)
          + f" | {m(d):+.1f} ({lo:+.1f} to {hi:+.1f}) | {len(trig)}/{len(vids)} |")
print('| mean of groups | | ' + ' | '.join(
    f"{np.mean([m([r['cer'] for r in rows if r['group'] == g and r['cfg'] == c]) for g in GR]):.1f}" for c in CF) + ' | | |')
print('\n## One whole box / missed (%)\n')
print('| Photos | ' + ' | '.join(f'{c} whole' for c in CF) + ' | ' + ' | '.join(f'{c} missed' for c in CF) + ' |')
print('|---|' + '---|' * 6)
for g in GR:
    y = lambda c: [r for r in rows if r['group'] == g and r['cfg'] == c]
    print(f'| {g} | ' + ' | '.join(f"{m([r['kind'] == 'one box' and not r['stacked_merge'] for r in y(c)]):.0f}" for c in CF)
          + ' | ' + ' | '.join(f"{m([r['kind'] == 'missed' for r in y(c)]):.0f}" for c in CF) + ' |')
