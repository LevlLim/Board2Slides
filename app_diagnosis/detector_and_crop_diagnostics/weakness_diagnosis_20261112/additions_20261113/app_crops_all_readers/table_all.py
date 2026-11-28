import json, csv, collections, numpy as np
FR='/tmp/claude-0/-home-claude/02f3df2f-1dbb-533d-b187-71169cf31ef0/scratchpad/unf/board2notes_freshwriter_test_20261111'
man=[r for r in csv.DictReader(open(FR+'/manifest.csv',encoding='utf-8')) if r['included']=='yes']
reads=json.load(open('all_reads.json'))
hand={(r['model'],r['id']):r['pred'] for r in csv.DictReader(open(FR+'/scores_server_models.csv',encoding='utf-8'))}
M={'app':(json.load(open('match.json')),json.load(open('app_boxes.json')),'crops'),'app960':(json.load(open('match960.json')),json.load(open('app_boxes960.json')),'crops960')}
norm=lambda s:' '.join(s.lower().split())
def lev(a,b):
    p=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        c=[i]
        for j,cb in enumerate(b,1): c.append(min(p[j]+1,c[j-1]+1,p[j-1]+(ca!=cb)))
        p=c
    return p[-1]
rows=[]
models=sorted({k.split('|')[0] for k in reads}|{'S0','S0s2','S1','S1s2','G0','G0s2','V2','S1his'})
for m in models:
    for r in man:
        a=norm(r['label'])
        pr={'hand': reads[f'{m}|hand'][FR+'/'+r['file']] if m=='pretrained' else hand[(m,r['id'])]}
        for src,(mt,bx,cd) in M.items():
            b=sorted(mt[r['id']]['boxes'],key=lambda k: np.mean([q[0] for q in bx[r['photo']][k]]))
            pr[src]=' '.join(reads[f'{m}|{src}'][f"{cd}/{r['photo']}_{k:02d}.png"] for k in b)
        for src,t in pr.items(): rows.append(dict(model=m,src=src,id=r['id'],board=r['board'],writing=r['writing'],pred=t,cer=lev(a,norm(t))/len(a)))
json.dump(rows,open('all_rows.json','w'),ensure_ascii=False)
G={'pretrained':['pretrained'],'S0 (2 seeds)':['S0','S0s2'],'S1 (2 seeds)':['S1','S1s2'],'G0 (2 seeds)':['G0','G0s2'],'V2':['V2'],'S1his':['S1his']}
T=collections.defaultdict(list)
for x in rows: T[(x['model'],x['src'])].append(x)
rng=np.random.default_rng(0); W=sorted({r['writing'] for r in man})
def ci(members,s1,s2):
    d=collections.defaultdict(list)
    for m in members:
        A={x['id']:x for x in T[(m,s1)]}; B={x['id']:x for x in T[(m,s2)]}
        for i in A: d[A[i]['writing']].append(A[i]['cer']-B[i]['cer'])
    bs=[]
    for _ in range(10000):
        s=rng.choice(W,len(W)); v=[y for w in s for y in d[w]]; bs.append(np.mean(v))
    allv=[y for w in W for y in d[w]]
    return 100*np.mean(allv),100*np.percentile(bs,2.5),100*np.percentile(bs,97.5)
print('| Reader | Hand-cut crops | App as it is now | App detector at 960 px | 960 px minus hand-cut [95% interval] |')
print('|---|---|---|---|---|')
for g,mm in G.items():
    v={s:100*np.mean([x['cer'] for m in mm for x in T[(m,s)]]) for s in ('hand','app','app960')}
    d=ci(mm,'app960','hand')
    print(f"| {g} | {v['hand']:.1f} | {v['app']:.1f} | {v['app960']:.1f} | {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}] |")
print()
print('By board, G0 2 seeds and S1his (hand / app / 960):')
for b in 'ABCDE':
    out=[]
    for g,mm in (('G0',['G0','G0s2']),('S1his',['S1his']),('pretrained',['pretrained'])):
        v=[100*np.mean([x['cer'] for m in mm for x in T[(m,s)] if x['board']==b]) for s in ('hand','app','app960')]
        out.append(f"{g} {v[0]:.1f}/{v[1]:.1f}/{v[2]:.1f}")
    print(b, ' | '.join(out))
