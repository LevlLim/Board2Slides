import json, csv, math, cv2, numpy as np
D='/tmp/claude-0/-home-claude/02f3df2f-1dbb-533d-b187-71169cf31ef0/scratchpad/unf/board2notes_freshwriter_test_20261111'
man={r['id']:r for r in csv.DictReader(open(D+'/manifest.csv',encoding='utf-8'))}
match=json.load(open('match.json'))
need=lambda t:(lambda u:len(u)+sum(a==b for a,b in zip(u,u[1:])))(' '.join(t.split()))
def g(im,lab):
    gr=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); _,b=cv2.threshold(gr,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ys=np.where(b.mean(1)>0.02)[0]; H,W=im.shape[:2]
    return math.ceil(math.ceil(64*W/H)/8)/need(lab), ((ys[-1]-ys[0]+1)/H if len(ys) else 1), H
rows=[]
for k,v in match.items():
    if v['included']!='yes' or len(v['boxes'])!=1 or v['merged']: continue
    r=man[k]; ph=r['photo']
    a=g(cv2.imread(f"crops/{ph}_{v['boxes'][0]:02d}.png"),r['label']); t=g(cv2.imread(D+'/'+r['file']),r['label'])
    rows.append((k,)+a+t)
A=np.array([x[1:] for x in rows])
print(len(rows),'one-box lines')
for nm,i in (('room',0),('ink height/crop height',1),('crop height px',2)):
    print(f"{nm:24s} app median {np.median(A[:,i]):.2f} (p10 {np.percentile(A[:,i],10):.2f})   tight median {np.median(A[:,i+3]):.2f} (p10 {np.percentile(A[:,i+3],10):.2f})")
print('app crops with room<1:', int((A[:,0]<1).sum()), ' room<1.5:', int((A[:,0]<1.5).sum()), '| tight room<1.5:', int((A[:,3]<1.5).sum()))
json.dump(rows,open('geom.json','w'))
