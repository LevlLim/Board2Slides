import json, collections, sys, numpy as np
READER = sys.argv[1] if len(sys.argv) > 1 else 'S1'
rows = json.load(open(f'rows_{READER}.json')); det = json.load(open('det.json'))
CF = ['app', '640', '960', '1280', '1600', '2400', 'full']
GR = sorted({r['group'] for r in rows}, key=lambda g: (('zoomed' in g), g))
def m(x): return 100 * np.mean(x) if len(x) else float('nan')
out = {}
print('## Mean line CER (%) by detector setting (reader', READER, ')\n')
print('| Photos | refs | median line height in photo px | ' + ' | '.join(CF) + ' |'); print('|---|---|---|' + '---|' * len(CF))
for g in GR:
    x = [r for r in rows if r['group'] == g]
    hp = np.median([r['line_h_photo'] for r in x if r['cfg'] == 'app'])
    print(f"| {g} | {len(x)//len(CF)} | {hp:.0f} | " + ' | '.join(f"{m([r['cer'] for r in x if r['cfg']==c]):.1f}" for c in CF) + ' |')
print('\n## Lines found as one whole box (%)\n')
print('| Photos | ' + ' | '.join(CF) + ' |'); print('|---|' + '---|' * len(CF))
for g in GR:
    x = [r for r in rows if r['group'] == g]
    print(f"| {g} | " + ' | '.join(f"{m([r['kind']=='one box' and not r['stacked_merge'] for r in x if r['cfg']==c]):.0f}" for c in CF) + ' |')
print('\n## Missed lines (%)\n')
print('| Photos | ' + ' | '.join(CF) + ' |'); print('|---|' + '---|' * len(CF))
for g in GR:
    x = [r for r in rows if r['group'] == g]
    print(f"| {g} | " + ' | '.join(f"{m([r['kind']=='missed' for r in x if r['cfg']==c]):.0f}" for c in CF) + ' |')
print('\n## Line height as the detector sees it (px) -> result, all fixed sizes pooled\n')
bins = [0, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 10**6]
x = [r for r in rows if r['cfg'] != 'app']
print('| line height in detector px | lines | CER % | one whole box % | split % | missed % |'); print('|---|---|---|---|---|---|')
for a, b in zip(bins, bins[1:]):
    y = [r for r in x if a <= r['line_h_det'] < b]
    if y: print(f"| {a}-{b if b < 10**6 else ''} | {len(y)} | {m([r['cer'] for r in y]):.1f} | {m([r['kind']=='one box' and not r['stacked_merge'] for r in y]):.0f} | {m([r['kind']=='split' for r in y]):.0f} | {m([r['kind']=='missed' for r in y]):.0f} |")
