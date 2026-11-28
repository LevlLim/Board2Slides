import json, re, sys, statistics as st
from pathlib import Path
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from refs import refs, ignored
J=Path("/home/claude/fresh/jobs")
def norm(s): return re.sub(r"\s+"," ",(s or "")).strip().lower()
def cer(p,r): r=norm(r); return Levenshtein.distance(norm(p),r)/max(1,len(r))
def x0(ev):
    b=ev.get("box_xyxy") or [0]; return b[0]
def score(n, R, IG=()):
    regs=n["ocr"]["regions"]; evs=n.get("region_evidence",[{}]*len(regs))
    assign=[]
    for g in regs:
        p=norm(g["text"]); best,bj=-1,None
        for j,r in enumerate(list(R)+list(IG)):
            r=norm(r); s=fuzz.partial_ratio(p,r)*min(1.0,len(r)/max(1,len(p))) if p else 0
            if s>best: best,bj=s,j
        assign.append(bj if best>=60 and len(p)>=2 else None)
    lines=[]
    for j,r in enumerate(R):
        idx=sorted([i for i,a in enumerate(assign) if a==j], key=lambda i:x0(evs[i]))
        pred=" ".join(regs[i]["text"] for i in idx)
        lines.append(dict(ref=r,pred=pred,found=bool(idx),cer=cer(pred,r) if idx else 1.0))
    extra=[regs[i]["text"] for i,a in enumerate(assign) if a is None]
    assign=[a if (a is None or a<len(R)) else -1 for a in assign]
    chars=sum(len(norm(r)) for r in R)
    e2e=(sum(Levenshtein.distance(norm(l["pred"]),norm(l["ref"])) if l["found"] else len(norm(l["ref"])) for l in lines)+sum(len(norm(t)) for t in extra))/chars
    return lines, extra, e2e
rows=[]
for d in sorted(J.iterdir()):
    b,R=refs(d.name[4:])
    for cond in ("original","auto","auto_f35","det2400","det4000"):
        for m in ("downloaded","ft40","p40","baseline","tuned"):
            f=d/f"ocr_{cond}_{m}.json"
            if not f.exists(): continue
            n=json.loads(f.read_text()); lines,extra,e2e=score(n,R,ignored(d.name[4:]))
            found=[l for l in lines if l["found"]]
            rows.append(dict(photo=d.name,board=b,cond=cond,model=m,lines=len(R),found=len(found),
                read=st.mean(l["cer"] for l in found) if found else None,e2e=e2e,extra=len(extra),lines_detail=lines,extra_text=extra))
json.dump(rows,open("/home/claude/fresh/scores.json","w"),indent=1,ensure_ascii=False)
def summ(sel):
    f=sum(r["found"] for r in sel); L=sum(r["lines"] for r in sel)
    reads=[l["cer"] for r in sel for l in r["lines_detail"] if l["found"]]
    allc=[l["cer"] for r in sel for l in r["lines_detail"]]
    return f"photos {len(sel):2d} | lines found {f}/{L} | reading CER (found lines) {100*st.mean(reads):5.1f}% | line CER, missed=100% {100*st.mean(allc):5.1f}% | page CER {100*st.mean(r['e2e'] for r in sel):5.1f}% | extra boxes {sum(r['extra'] for r in sel)}"
for cond in ("original","auto","auto_f35","det2400","det4000"):
    for m in ("downloaded","ft40","p40","baseline","tuned"):
        sel=[r for r in rows if r["cond"]==cond and r["model"]==m]
        if sel: print(f"{cond:9s} {m:8s}", summ(sel))
