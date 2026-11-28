import json, csv, collections, numpy as np
D='/tmp/claude-0/-home-claude/02f3df2f-1dbb-533d-b187-71169cf31ef0/scratchpad/unf/board2notes_freshwriter_test_20261111'
man={r['id']:r for r in csv.DictReader(open(D+'/manifest.csv',encoding='utf-8'))}
match=json.load(open('match960.json')); reads=json.load(open('app_reads960.json')); app=json.load(open('app_boxes960.json'))
tight={}
for r in json.load(open('/home/claude/diag/stretch_fresh_rows.json')):
    if r['s']==1.0: tight[(r['model'],r['id'])]=r['cer']
norm=lambda s:' '.join(s.lower().split())
def lev(a,b):
    p=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        c=[i]
        for j,cb in enumerate(b,1): c.append(min(p[j]+1,c[j-1]+1,p[j-1]+(ca!=cb)))
        p=c
    return p[-1]
rows=[]
for k,v in match.items():
    if v['included']!='yes': continue
    r=man[k]; ph=r['photo']; bx=v['boxes']
    bx=sorted(bx,key=lambda b: np.mean([q[0] for q in app[ph][b]]))
    kind='merged' if v['merged'] else 'missed' if not bx else 'one box' if len(bx)==1 else 'split'
    for m in reads:
        txt=' '.join(reads[m][f"crops960/{ph}_{b:02d}.png"] for b in bx)
        a=norm(r['label']); rows.append(dict(id=k,board=r['board'],writing=r['writing'],kind=kind,model=m,app=lev(a,norm(txt))/len(a),tight=tight[(m,k)],pred=txt,label=r['label']))
json.dump(rows,open('app_rows960.json','w'),ensure_ascii=False)
grp=lambda m:'S0' if m.startswith('S0') else 'G0' if m.startswith('G0') else m
T=collections.defaultdict(lambda:[[],[]])
for x in rows:
    for kk in ('all',x['kind']): T[(grp(x['model']),kk)][0].append(x['app']); T[(grp(x['model']),kk)][1].append(x['tight'])
for g in ('S0','G0','S1his'):
    for kk in ('all','one box','split','merged'):
        a,t=T[(g,kk)]
        if a: print(f"{g:6s} {kk:8s} n={len(a)//(2 if g!='S1his' else 1):3d}  app {100*np.mean(a):5.1f}   tight {100*np.mean(t):5.1f}")
