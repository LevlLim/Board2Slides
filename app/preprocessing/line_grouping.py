"""Join detector fragments in text coordinates, preserving separate rows.

Axis-aligned bounding rectangles overlap vertically for long tilted sentences
even when their actual rows are well separated. Compare distances normal to
the writing direction instead. No OCR strings or labels enter this function.
"""
from __future__ import annotations
import cv2
import numpy as np
from . import text_lines as T

_legacy_group = T.group_into_lines


def group_into_lines(polys,gap_factor=1.2,min_v_overlap=.5):
    boxes=[T.order_tl_tr_br_bl(p) for p in polys]
    directions=[]
    for p in boxes:
        d=(p[1]-p[0]+p[2]-p[3]).astype(float)
        if d[0]<0:d=-d
        directions.append(d/max(float(np.linalg.norm(d)),1e-6))
    heights=[T.line_height(p) for p in boxes]
    groups=[[i] for i in range(len(boxes))]

    def same_row(a,b):
        if min(heights[a],heights[b])<=0 or max(heights[a],heights[b])>2*min(heights[a],heights[b]):return False
        if np.dot(directions[a],directions[b])<np.cos(np.radians(5)):return False
        axis=directions[a]+directions[b];axis/=max(float(np.linalg.norm(axis)),1e-6)
        normal=np.array([-axis[1],axis[0]])
        av,bv=boxes[a]@normal,boxes[b]@normal
        overlap=min(av.max(),bv.max())-max(av.min(),bv.min())
        if overlap<min_v_overlap*min(float(np.ptp(av)),float(np.ptp(bv))):return False
        au,bu=boxes[a]@axis,boxes[b]@axis
        gap=max(bu.min()-au.max(),au.min()-bu.max())
        return gap<gap_factor*max(heights[a],heights[b])

    edges=[(a,b) for a in range(len(boxes)) for b in range(a+1,len(boxes)) if same_row(a,b)]
    # A chain of slightly displaced fragments must not connect distinct rows.
    for a,b in edges:
        ga=next(g for g in groups if a in g);gb=next(g for g in groups if b in g)
        if ga is gb:continue
        joined=ga+gb;axis=np.mean([directions[i] for i in joined],axis=0)
        axis/=max(float(np.linalg.norm(axis)),1e-6);normal=np.array([-axis[1],axis[0]])
        offsets=[float(boxes[i].mean(0)@normal) for i in joined]
        if np.ptp(offsets)>(1-min_v_overlap)*min(heights[i] for i in joined):continue
        ga.extend(gb);groups.remove(gb)
    output=[]
    for group in groups:
        p=boxes[group[0]] if len(group)==1 else T.order_tl_tr_br_bl(cv2.boxPoints(cv2.minAreaRect(np.vstack([boxes[i] for i in group]))))
        output.append((p,sorted(group)))
    return [p for p,_ in output],[g for _,g in output]


def group_with_row_guard(polys,gap_factor=1.2,min_v_overlap=.5):
    """Preserve old fragment joins unless a group spans distinct text rows.

    The first candidate unnecessarily split some useful word fragments. This
    guard requires overlapping along-line spans AND separated baselines
    before replacing an old group. It is a geometric check, not an OCR vote.
    """
    boxes=[T.order_tl_tr_br_bl(p) for p in polys]
    old,groups=_legacy_group(boxes,gap_factor=gap_factor,min_v_overlap=min_v_overlap)
    output=[];members=[]
    for old_box,group in zip(old,groups):
        separated=False
        for k,a in enumerate(group):
            for b in group[k+1:]:
                qa,qb=boxes[a],boxes[b]
                da=(qa[1]-qa[0]+qa[2]-qa[3]).astype(float);da/=max(float(np.linalg.norm(da)),1e-6)
                db=(qb[1]-qb[0]+qb[2]-qb[3]).astype(float);db/=max(float(np.linalg.norm(db)),1e-6)
                if np.dot(da,db)<np.cos(np.radians(12)):continue
                axis=da+db;axis/=max(float(np.linalg.norm(axis)),1e-6)
                normal=np.array([-axis[1],axis[0]])
                au,bu=qa@axis,qb@axis
                overlap=min(au.max(),bu.max())-max(au.min(),bu.min())
                separation=abs(float((qa.mean(0)-qb.mean(0))@normal))
                if (overlap>.3*min(float(np.ptp(au)),float(np.ptp(bu))) and
                        separation>.65*min(T.line_height(qa),T.line_height(qb))):
                    separated=True
        if not separated:
            output.append(old_box);members.append(group)
        else:
            pieces,indices=group_into_lines([boxes[i] for i in group],gap_factor,min_v_overlap)
            output.extend(pieces);members.extend([[group[i] for i in subset] for subset in indices])
    return output,members