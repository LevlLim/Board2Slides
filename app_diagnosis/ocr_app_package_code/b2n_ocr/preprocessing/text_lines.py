"""Text-line detection, cutting and enlarging (hurdle slide 8 "Zoom in / Enlarge").

Taken from Assignment/section8_auto_rectify_cells.ipynb section 9 (smart_ocr steps 1-4) with the globals removed.

  Detector.detect(img)      -> text-line boxes, the same way the app finds them (smart_ocr):
                               1. quick detection on the photo shrunk to 1600 px;
                               2. if the writing is smaller than DET_MIN_H px as the detector saw it,
                                  re-detect each text region at full resolution.
  group_into_lines(boxes)   -> boxes on the same line joined (the detector sometimes splits a line at a big gap)
  reading_order(boxes)      -> top-to-bottom, left-to-right
  quad_crop(img, box)       -> the line cut along its own (possibly tilted) box, like PaddleOCR does
  prepare_for_reader(crop)  -> lines shorter than REC_MIN_H px are enlarged to TARGET_H with bicubic + sharpen
                               (the reader works at 48 px; we tested ESPCN/TPGSR: no better, see
                               docs/SUPER_RESOLUTION_RUNS.md)

PaddleOCR is imported only when a Detector is created, so the geometry helpers work without it.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DET_MIN_H = 20        # writing smaller than this (px, as the 1600 px detector sees it) -> re-detect at full resolution
REC_MIN_H = 48        # lines shorter than this (px) are enlarged before reading (recognizer input height = 48 px)
TARGET_H = 64         # ... to about this height
MAX_SCALE = 4.0
QUICK_LIMIT = 1600
REGION_MAX = 4000     # PaddleOCR shrinks anything larger than this

# the detection preset of paddleocr_baseline.ipynb ("baseline"), used everywhere in the project
DET_ARGS = dict(thresh=0.20, box_thresh=0.40, unclip_ratio=2.00)


# ---------------------------------------------------------------- box helpers
def as_poly(p):
    return np.asarray(p, float).reshape(-1, 2)


def box_area(p):
    return float(cv2.contourArea(np.float32(as_poly(p))))


def line_height(poly):
    # height of a (roughly horizontal) text-line box: mean of its two short sides
    p = as_poly(poly)
    if len(p) != 4:
        return float(np.ptp(p[:, 1]))
    return float((np.linalg.norm(p[0] - p[3]) + np.linalg.norm(p[1] - p[2])) / 2)


def line_width(poly):
    p = as_poly(poly)
    if len(p) != 4:
        return float(np.ptp(p[:, 0]))
    return float((np.linalg.norm(p[0] - p[1]) + np.linalg.norm(p[3] - p[2])) / 2)


def order_tl_tr_br_bl(p):
    p = np.float32(as_poly(p))
    if len(p) != 4:
        p = np.float32(cv2.boxPoints(cv2.minAreaRect(p)))
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.float32([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]])


def merge_boxes(polys, overlap=0.5):
    # drop boxes that lie mostly inside a bigger one (same line found twice in overlapping regions)
    polys = sorted((np.float32(as_poly(p)) for p in polys), key=lambda p: -box_area(p))
    kept = []
    for a in polys:
        area = max(box_area(a), 1)
        if all(cv2.intersectConvexConvex(a, k)[0] / area <= overlap for k in kept):
            kept.append(a)
    return kept


def text_regions(polys, img_shape, grow=1.5, margin=0.6):
    # group nearby boxes into clusters; one full-resolution region (with margin) per cluster
    h, w = img_shape[:2]
    rects = []
    for p in polys:
        p = as_poly(p); lh = line_height(p)
        rects.append([*(p.min(0) - grow * lh), *(p.max(0) + grow * lh), lh])
    parent = list(range(len(rects)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = rects[i], rects[j]
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(rects)):
        groups.setdefault(find(i), []).append(i)
    regions = []
    for idx in groups.values():
        pts = np.vstack([as_poly(polys[i]) for i in idx])
        lh = np.median([rects[i][4] for i in idx])
        (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
        mx, my = max(margin * (x1 - x0), 4 * lh), max(margin * (y1 - y0), 4 * lh)
        regions.append([int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(w, x1 + mx)), int(min(h, y1 + my))])
    merged = True
    while merged:
        merged = False
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                a, b = regions[i], regions[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    regions[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    regions.pop(j); merged = True
                    break
            if merged:
                break
    return [tuple(r) for r in regions]


def on_mask(polys, mask):
    # boxes whose centre is on the board (all boxes if there is no board mask or none are on it)
    if mask is None:
        return list(polys)
    h, w = mask.shape
    keep = [p for p in polys if mask[int(np.clip(as_poly(p)[:, 1].mean(), 0, h - 1)),
                                     int(np.clip(as_poly(p)[:, 0].mean(), 0, w - 1))] > 0]
    return keep or list(polys)


# ---------------------------------------------------------------- lines: grouping and order
def group_into_lines(polys, gap_factor=1.2, min_v_overlap=0.5):
    """Join boxes that sit on the same text line (vertical overlap >= min_v_overlap of the smaller box, and a
    horizontal gap below gap_factor x line height). Returns a list of 4-point boxes (minAreaRect of the parts)
    and, for each, the indices of the boxes it was made from."""
    P = [order_tl_tr_br_bl(p) for p in polys]
    n = len(P)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    ext = [(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max(), line_height(p)) for p in P]
    for i in range(n):
        for j in range(i + 1, n):
            ax0, ay0, ax1, ay1, ah = ext[i]
            bx0, by0, bx1, by1, bh = ext[j]
            v_ov = min(ay1, by1) - max(ay0, by0)
            if v_ov < min_v_overlap * min(ay1 - ay0, by1 - by0):
                continue
            if max(ah, bh) > 2.0 * min(ah, bh):           # very different sizes: a title next to body text
                continue
            gap = max(bx0 - ax1, ax0 - bx1)
            if gap < gap_factor * max(ah, bh):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for idx in groups.values():
        if len(idx) == 1:
            out.append((P[idx[0]], idx))
        else:
            pts = np.vstack([P[i] for i in idx])
            out.append((order_tl_tr_br_bl(cv2.boxPoints(cv2.minAreaRect(np.float32(pts)))), sorted(idx)))
    return [o[0] for o in out], [o[1] for o in out]


def reading_order(polys):
    """Indices of the boxes top-to-bottom, then left-to-right within a row (a row = boxes whose vertical centres
    are within half a line height of each other)."""
    if not polys:
        return []
    c = [(as_poly(p)[:, 0].mean(), as_poly(p)[:, 1].mean(), line_height(p)) for p in polys]
    idx = sorted(range(len(polys)), key=lambda i: c[i][1])
    rows, cur = [], [idx[0]]
    for i in idx[1:]:
        ref = np.mean([c[k][1] for k in cur]); lh = np.median([c[k][2] for k in cur])
        if abs(c[i][1] - ref) < 0.5 * lh:
            cur.append(i)
        else:
            rows.append(cur); cur = [i]
    rows.append(cur)
    return [i for r in rows for i in sorted(r, key=lambda k: c[k][0])]


# ---------------------------------------------------------------- cutting and enlarging
def quad_crop(img, poly, pad=0.0):
    """Cut a text line along its own (possibly tilted) box, like PaddleOCR does internally.
    pad: grow the box by this fraction of its height on every side first (0 = exactly the box)."""
    p = order_tl_tr_br_bl(poly)
    if pad:
        lh = line_height(p)
        ux = (p[1] - p[0]) / max(np.linalg.norm(p[1] - p[0]), 1e-6)     # along the line
        uy = (p[3] - p[0]) / max(np.linalg.norm(p[3] - p[0]), 1e-6)     # across the line
        p = (p + np.float32([(-ux - uy), (ux - uy), (ux + uy), (-ux + uy)]) * pad * lh).astype(np.float32)
    w = int(round(max(np.linalg.norm(p[0] - p[1]), np.linalg.norm(p[3] - p[2]))))
    h = int(round(max(np.linalg.norm(p[0] - p[3]), np.linalg.norm(p[1] - p[2]))))
    M = cv2.getPerspectiveTransform(p, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    return cv2.warpPerspective(img, M, (max(w, 1), max(h, 1)), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def enlarge(img, s, method="bicubic+sharpen"):
    # enlarge an image by factor s (no learned super-resolution: it did not help, see SUPER_RESOLUTION_RUNS.md)
    if s <= 1.05 or method == "none":
        return img
    size = (int(round(img.shape[1] * s)), int(round(img.shape[0] * s)))
    if method == "bicubic":
        return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    if method == "lanczos":
        return cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    if method == "bicubic+sharpen":
        up = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(up, (0, 0), sigmaX=0.8 * s)
        return cv2.addWeighted(up, 1.6, blur, -0.6, 0)           # unsharp mask: strengthens stroke edges
    raise ValueError(f"unknown enlargement method {method!r}")


def prepare_for_reader(crop, height=None, method="bicubic+sharpen"):
    """What the app feeds the reader: lines shorter than REC_MIN_H are enlarged to ~TARGET_H, others unchanged.
    height: the line's box height (px); the crop height if not given."""
    lh = float(height if height is not None else crop.shape[0])
    if lh >= REC_MIN_H:
        return crop
    return enlarge(crop, float(np.clip(TARGET_H / max(lh, 1), 1, MAX_SCALE)), method)


def ink_rows(crop):
    """Number of text rows in a line crop (1 = one line). Horizontal ink profile after dividing out the board
    background; bands shorter than 12 % of the crop or 40 % of the tallest band (underlines, the dot of an i, an
    arrow tip) do not count. Checked by eye on 64 crops (23 Oct): it found the two-line crops and none of the
    single lines with underlines or arrows."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) if crop.ndim == 3 else crop.astype(np.float32)
    h = g.shape[0]
    if h < 8:
        return 1
    bg = cv2.GaussianBlur(g, (0, 0), max(3.0, h / 2))
    ink = (np.clip(g / np.maximum(bg, 1), 0, 1.5) < 0.8).astype(np.float32)
    k = max(1, int(round(h / 20)))
    prof = np.convolve(ink.mean(1), np.ones(k) / k, mode="same")
    if prof.max() <= 0:
        return 0
    on = list(prof > 0.3 * prof.max()) + [False]
    bands, start = [], None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        if not v and start is not None:
            bands.append([start, y]); start = None
    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] < 0.06 * h:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    big = [b for b in merged if b[1] - b[0] >= 0.12 * h]
    if not big:
        return 1
    tall = max(b[1] - b[0] for b in big)
    return len([b for b in big if b[1] - b[0] >= 0.4 * tall])


# ---------------------------------------------------------------- the detector (PaddleOCR, lazy)
class Detector:
    """PaddleOCR text detection with the app's settings (PP-OCRv5_mobile_det, the baseline preset).
    The mobile detector was ~6x faster than PP-OCRv6_medium_det on a 1600 px tile and found the same lines
    (section8 notebook, section 9.6)."""

    def __init__(self, model_name="PP-OCRv5_mobile_det", det_args=None, model_dir=None, runtime=None):
        import os
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import TextDetection
        self.model_name = model_name
        extra = dict(runtime or {})
        if model_dir is not None:
            if not (Path(model_dir) / "inference.pdiparams").is_file():
                raise FileNotFoundError(f"Packaged detector weights missing: {model_dir}")
            extra["model_dir"] = str(model_dir)
        self.model = TextDetection(model_name=model_name, **extra)
        self.args = dict(DET_ARGS, **(det_args or {}))

    def boxes(self, img_bgr, limit=None, unclip=None):
        # text boxes only. limit=1600 shrinks the image to at most 1600 px; None = full resolution (<= REGION_MAX)
        # unclip: override of the box growth (2.0 in the baseline preset; large values join neighbouring lines)
        kw = dict(self.args, limit_side_len=limit or REGION_MAX, limit_type="max")
        if unclip is not None:
            kw["unclip_ratio"] = unclip
        return [as_poly(p) for p in self.model.predict(img_bgr, **kw)[0]["dt_polys"]]

    def detect(self, img_bgr, board_mask_fn=None, unclip=None):
        """smart_ocr steps 1-3. board_mask_fn(img, boxes) -> mask or None (board_geometry.segment_board_floodfill)
        keeps the work on the board. Returns (boxes, info)."""
        h, w = img_bgr.shape[:2]
        shrink = min(1.0, QUICK_LIMIT / max(h, w))
        info = dict(path="", writing_px_detector=None, regions=0)
        quick = self.boxes(img_bgr, limit=QUICK_LIMIT, unclip=unclip)
        if not quick:
            info["path"] = "nothing found in the quick pass"
            return [], info
        mask = board_mask_fn(img_bgr, quick) if board_mask_fn else None
        writing = on_mask(quick, mask)
        det_h = float(np.median([line_height(p) for p in writing])) * shrink
        info["writing_px_detector"] = round(det_h, 1)
        if det_h >= DET_MIN_H:
            info["path"] = f"text large enough ({det_h:.0f} px) -> quick boxes kept"
            return writing, info
        regions = text_regions(writing, img_bgr.shape)
        polys = []
        for x0, y0, x1, y1 in regions:
            polys += [p + [x0, y0] for p in self.boxes(img_bgr[y0:y1, x0:x1], unclip=unclip)]
        info["regions"] = len(regions)
        info["path"] = f"small text ({det_h:.0f} px) -> {len(regions)} region(s) re-detected at full resolution"
        return merge_boxes(polys) or writing, info


class Reader:
    """PaddleOCR text-line recognition (the app's reader, PP-OCRv6_medium_rec unless a model_dir is given)."""

    def __init__(self, model_name="PP-OCRv6_medium_rec", model_dir=None):
        import os
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import TextRecognition
        self.model = TextRecognition(model_name=model_name, **({"model_dir": str(model_dir)} if model_dir else {}))

    def read(self, images, batch=16):
        out = []
        for i in range(0, len(images), batch):
            out += [(r["rec_text"], float(r["rec_score"])) for r in self.model.predict(images[i:i + batch])]
        return out
