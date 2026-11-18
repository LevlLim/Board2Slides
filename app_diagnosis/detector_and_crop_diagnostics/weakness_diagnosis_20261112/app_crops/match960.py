import json, csv, cv2, numpy as np, collections
D='/tmp/claude-0/-home-claude/02f3df2f-1dbb-533d-b187-71169cf31ef0/scratchpad/unf/board2notes_freshwriter_test_20261111'
man=list(csv.DictReader(open(D+'/manifest.csv',encoding='utf-8')))
app=json.load(open('app_boxes960.json'))
S=0.25
def mask(q,shape):
    m=np.zeros(shape,np.uint8); cv2.fillPoly(m,[np.round(np.asarray(q)*S).astype(np.int32)],1); return m
out={}; stats=collections.Counter()
for ph,boxes in app.items():
    lines=[r for r in man if r['photo']==ph and r['quad']]
    shape=(1100,1100)
    lm=[mask(json.loads(r['quad']),shape) for r in lines]
    assign=collections.defaultdict(list); merges=set()
    for k,b in enumerate(boxes):
        bm=mask(b,shape); a=bm.sum()
        if a==0: continue
        ov=[(bm&m).sum()/a for m in lm]
        cov=[(bm&m).sum()/max(m.sum(),1) for m in lm]
        hits=[i for i,(o,c) in enumerate(zip(ov,cov)) if o>0.25 or c>0.25]
        if not hits: stats['app box not on a labelled line']+=1; continue
        best=max(hits,key=lambda i:ov[i]); assign[best].append(k)
        if len(hits)>1:
            for i in hits: merges.add(i)
    for i,r in enumerate(lines):
        out[r['id']]=dict(boxes=assign.get(i,[]),merged=i in merges,included=r['included'])
        if r['included']=='yes':
            n=len(assign.get(i,[])); stats['merged with another line' if i in merges else ('missed' if n==0 else 'one box' if n==1 else 'split into %d'%min(n,4))]+=1
json.dump(out,open('match960.json','w'),indent=0)
print(dict(stats)); print('labelled lines missing from manifest photos:', set(app)-{r['photo'] for r in man})
