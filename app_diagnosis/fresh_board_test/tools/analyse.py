import csv, random, json
from collections import defaultdict
D = '/tmp/claude-0/-home-claude/02f3df2f-1dbb-533d-b187-71169cf31ef0/scratchpad/unf/board2notes_freshwriter_test_20261111'
rows = list(csv.DictReader(open(D + '/scores_server_models.csv', encoding='utf-8')))
by = defaultdict(dict)
for r in rows: by[r['model']][r['id']] = r
ids = sorted(by['S0'])
def avg(a, b): return {i: (float(by[a][i]['cer']) + float(by[b][i]['cer'])) / 2 for i in ids}
cer = {m: {i: float(by[m][i]['cer']) for i in ids} for m in by}
cer['S0avg'] = avg('S0', 'S0s2'); cer['S1avg'] = avg('S1', 'S1s2'); cer['G0avg'] = avg('G0', 'G0s2')
meta = {i: by['S0'][i] for i in ids}
def boot(a, b, key, n=10000):
    g = defaultdict(list)
    for i in ids: g[meta[i][key]].append(cer[a][i] - cer[b][i])
    G = list(g.values()); mean = lambda gs: sum(map(sum, gs)) / sum(map(len, gs)); rng = random.Random(0)
    s = sorted(mean([G[rng.randrange(len(G))] for _ in G]) for _ in range(n))
    return mean(G), s[int(.025 * n)], s[int(.975 * n) - 1], len(G)
order = ['S0', 'S0s2', 'S0avg', 'S1', 'S1s2', 'S1avg', 'G0', 'G0s2', 'G0avg', 'V2', 'S1his']
boards = 'ABCDE'
print('model  all  ' + '  '.join(boards))
for m in order:
    line = f"{m:6s} {100*sum(cer[m].values())/len(ids):5.1f} " + ' '.join(f"{100*sum(cer[m][i] for i in ids if meta[i]['board']==b)/sum(1 for i in ids if meta[i]['board']==b):5.1f}" for b in boards)
    print(line)
# writing-weighted mean (each writing counts once)
for m in ['S0avg', 'S1avg', 'G0avg', 'V2', 'S1his']:
    g = defaultdict(list)
    for i in ids: g[meta[i]['writing']].append(cer[m][i])
    print('per-writing mean', m, round(100 * sum(sum(v)/len(v) for v in g.values()) / len(g), 2))
for a, b in [('G0avg', 'S0avg'), ('S1avg', 'S0avg'), ('G0', 'S0'), ('G0s2', 'S0s2'), ('S0s2', 'S0'), ('S1s2', 'S1'), ('G0s2', 'G0'), ('V2', 'S0avg'), ('S1his', 'S1avg'), ('S1his', 'G0avg')]:
    for key in ('writing', 'photo'):
        d, lo, hi, k = boot(a, b, key)
        print(f"{a:6s} - {b:6s} by {key:8s} ({k:2d} groups): {100*d:+6.2f} [{100*lo:+6.2f}, {100*hi:+6.2f}]")
# per board G0avg - S0avg
for b in boards:
    sub = [i for i in ids if meta[i]['board'] == b]
    print('board', b, 'G0avg-S0avg', round(100 * sum(cer['G0avg'][i] - cer['S0avg'][i] for i in sub) / len(sub), 2), 'V2-S0avg', round(100 * sum(cer['V2'][i] - cer['S0avg'][i] for i in sub) / len(sub), 2))
