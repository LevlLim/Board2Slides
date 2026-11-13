"""Perspective correction of a whiteboard photo (hurdle slide 7).

Taken from Assignment/section8_auto_rectify_cells.ipynb (sections 8.1-8.3) with the notebook globals removed:
the PaddleOCR text boxes are PASSED IN (from preprocessing.text_lines.Detector), so this module needs only
OpenCV + NumPy + Pillow and can be tested without PaddleOCR.

How it works (strict frame-evidence gate added after the notebook experiments):
  1. find the board: flood-fill from the writing, stopped by strong edges (frame, tray)       segment_board_floodfill
  2. find long straight lines (Hough + merging of collinear fragments)                          detect_long_lines
  3. pick two board edges that are parallel in reality, around the writing                      choose_plane_edges
  4. their vanishing point + the camera focal length (EXIF) give the board's rotation R;
     the homography K R K^-1 turns the board to face the camera                                  rotation_from_edges
  5. warp, keeping the resolution of the writing                                                fit_to_canvas_at

Changes from the notebook (the notebook rectifies around ONE target line; a dataset needs every line):
  * the close-up fallback (rectify the single largest text box) is OFF by default: it keeps one box only;
  * the edge pair is anchored at the median of all writing and must enclose >= MIN_INSIDE of the text boxes;
  * the result keeps all the writing on the board (not only the band between the two edges), and a warp is
    rejected if it would push text out of the image (MIN_TEXT_KEPT) or turns the board > MAX_BOARD_ANGLE.
  Failures found in the first order-check run (22 Oct): oct17_040 was cut down to the lone word "output";
  oct17_002 / oct17_015 were cut to a strip or over-warped. Without usable edges the photo stays unrectified.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MAX_SIDE = 4000          # max width/height of a rectified image (px)
DEFAULT_F35 = 24         # iPhone 14 Pro main camera (the EXIF of our PNGs says 24 mm); used if EXIF is missing

# ---------------------------------------------------------------- camera
def focal35_from_exif(path) -> float | None:
    """35 mm-equivalent focal length from EXIF (our PNGs keep the iPhone EXIF), or None."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            f35 = im.getexif().get_ifd(0x8769).get(0xA405)
        return float(f35) if f35 else None
    except Exception:
        return None


def camera_matrix(w, h, f35=DEFAULT_F35):
    # K from the 35 mm-equivalent focal length (a 35 mm frame is 36 mm wide)
    f = f35 / 36.0 * max(w, h)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def line_through(p, q):
    # homogeneous line through two image points
    return np.cross([*p, 1.0], [*q, 1.0])


# ---------------------------------------------------------------- 1. board region (flood fill, no model)
FLOOD_WORK_SIDE = 1000   # long side (px) of the downscaled copy used for flood fill
FLOOD_DIFF = 3           # max grey-level step between neighbouring pixels (gradual lighting passes, hard edges stop)


def segment_board_floodfill(img_bgr, boxes):
    """Board mask (uint8 0/255, full res) grown from the writing, stopped by strong edges; None if no boxes."""
    h, w = img_bgr.shape[:2]
    if not boxes:
        return None
    s = FLOOD_WORK_SIDE / max(h, w)
    small = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    sh, sw = gray.shape

    centres = [b.mean(0) for b in boxes]
    k = min(range(len(boxes)), key=lambda i: np.hypot(centres[i][0] - w / 2, centres[i][1] - h / 2))
    x0, y0 = (boxes[k].min(0) * s).astype(int)
    x1, y1 = (boxes[k].max(0) * s).astype(int)

    barrier = cv2.dilate(cv2.Canny(gray, 20, 60), np.ones((3, 3), np.uint8))
    barrier[max(0, y0):y1, max(0, x0):x1] = 0
    ff_mask = np.zeros((sh + 2, sw + 2), np.uint8)
    ff_mask[1:-1, 1:-1] = (barrier > 0).astype(np.uint8)

    pad = max(4, int(0.3 * (y1 - y0)))
    seeds = [(x0 - pad, (y0 + y1) // 2), (x1 + pad, (y0 + y1) // 2), ((x0 + x1) // 2, y0 - pad),
             ((x0 + x1) // 2, y1 + pad), ((x0 + x1) // 2, (y0 + y1) // 2)]
    filled = ff_mask.copy()
    for sx, sy in seeds:
        sx, sy = int(np.clip(sx, 0, sw - 1)), int(np.clip(sy, 0, sh - 1))
        if filled[sy + 1, sx + 1] == 0:
            cv2.floodFill(gray.copy(), filled, (sx, sy), 255, FLOOD_DIFF, FLOOD_DIFF,
                          4 | cv2.FLOODFILL_MASK_ONLY | (2 << 8))
    region = (filled[1:-1, 1:-1] == 2).astype(np.uint8) * 255

    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab = cv2.connectedComponents(region)
    cy, cx = int(np.clip((y0 + y1) / 2, 0, sh - 1)), int(np.clip((x0 + x1) / 2, 0, sw - 1))
    keep = lab[cy, cx] if lab[cy, cx] else (np.bincount(lab.ravel())[1:].argmax() + 1 if n > 1 else 0)
    region = (lab == keep).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = cv2.drawContours(np.zeros_like(region), cnts, -1, 255, cv2.FILLED)
    return cv2.resize(region, (w, h), interpolation=cv2.INTER_NEAREST)


BOUNDARY_BAND = 0.015    # a line counts as a board edge if it runs within this fraction of the long side of the mask outline
BOUNDARY_FRAC = 0.6      # ... for at least this fraction of its length


def keep_boundary_lines(lines, mask):
    # only lines that run along the outline of the board mask (the real board edges), not the photo border
    h, w = mask.shape
    s = 1000 / max(h, w)
    small = cv2.resize(mask, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    outline = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    outline[:2, :] = 0; outline[-2:, :] = 0; outline[:, :2] = 0; outline[:, -2:] = 0
    dist = cv2.distanceTransform(255 - outline, cv2.DIST_L2, 3)
    band = BOUNDARY_BAND * 1000
    kept = []
    for e in lines:
        t = np.linspace(0, 1, 40)
        xs = (e["p"][0] + t * (e["q"][0] - e["p"][0])) * s
        ys = (e["p"][1] + t * (e["q"][1] - e["p"][1])) * s
        ok = (xs >= 0) & (xs < small.shape[1]) & (ys >= 0) & (ys < small.shape[0])
        if ok.sum() < 5:
            continue
        if (dist[ys[ok].astype(int), xs[ok].astype(int)] < band).mean() >= BOUNDARY_FRAC:
            kept.append(e)
    return kept


# ---------------------------------------------------------------- 2-3. long lines and the two board edges
LINE_WORK_SIDE = 1600   # long side (px) of the downscaled copy used for line detection
MIN_LINE_FRAC = 0.12    # minimum merged line length, as a fraction of the long side
VERTICAL_TOL = 25       # degrees from image-vertical that still counts as a vertical line
MAX_GAP_FRAC = 0.05     # max gap between line fragments that are merged into one line
MIN_PAIR_FRAC = 0.15    # the two plane edges must each be at least this long (fraction of the long side)
MIN_REF_FRAC = 0.25     # a reference line must be at least this long


def detect_long_lines(img_bgr):
    # long straight lines in full-res coords, with nearly-collinear Hough fragments merged
    h, w = img_bgr.shape[:2]
    s = LINE_WORK_SIDE / max(h, w)
    g = cv2.cvtColor(cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 15, 45)
    segs = cv2.HoughLinesP(e, 1, np.pi / 720, 40, minLineLength=int(0.04 * LINE_WORK_SIDE), maxLineGap=12)
    if segs is None:
        return []
    segs = np.asarray(segs).reshape(-1, 4).astype(float) / s     # (N,1,4) in OpenCV 4, (N,4) in OpenCV 5
    diag = np.hypot(w, h)
    info = []
    for x1, y1, x2, y2 in segs:
        th = np.arctan2(y2 - y1, x2 - x1) % np.pi
        n = np.array([-np.sin(th), np.cos(th)])
        info.append((th, n @ [x1, y1], np.hypot(x2 - x1, y2 - y1), (x1, y1, x2, y2)))
    used, lines = [False] * len(info), []
    for i in np.argsort([-l for _, _, l, _ in info]):
        if used[i]:
            continue
        th0, r0 = info[i][0], info[i][1]
        group = []
        for j, (th, r, l, seg) in enumerate(info):
            dth = min(abs(th - th0), np.pi - abs(th - th0))
            r_adj = r if abs(th - th0) < np.pi / 2 else -r
            if not used[j] and dth < np.radians(1.5) and abs(r_adj - r0) < 0.005 * diag:
                used[j] = True
                group.append((l, seg))
        d = np.array([np.cos(th0), np.sin(th0)])
        spans = sorted((min(d @ seg[:2], d @ seg[2:]), max(d @ seg[:2], d @ seg[2:]), l, seg) for l, seg in group)
        runs, cur, end = [], [spans[0]], spans[0][1]
        for sp in spans[1:]:
            if sp[0] - end > MAX_GAP_FRAC * max(h, w):
                runs.append(cur); cur = []
            cur.append(sp); end = max(end, sp[1])
        runs.append(cur)
        best_run = max(runs, key=lambda r: sum(x[2] for x in r))
        support = sum(x[2] for x in best_run)
        if support < MIN_LINE_FRAC * max(h, w):
            continue
        pts = np.float32([p for *_, seg in best_run for p in (seg[:2], seg[2:])])
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        t = (pts - [x0, y0]) @ [vx, vy]
        p, q = (x0 + vx * t.min(), y0 + vy * t.min()), (x0 + vx * t.max(), y0 + vy * t.max())
        deg_from_vertical = np.degrees(np.arccos(abs(vy)))
        extent = float(np.hypot(q[0] - p[0], q[1] - p[1]))
        lines.append(dict(p=p, q=q, length=extent, support=support, kind="v" if deg_from_vertical < VERTICAL_TOL else "h"))
    return lines


def y_at_x(e, x):
    (x1, y1), (x2, y2) = e["p"], e["q"]
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def x_at_y(e, y):
    (x1, y1), (x2, y2) = e["p"], e["q"]
    return x1 + (x2 - x1) * (y - y1) / (y2 - y1)


def _mid(e):
    return ((e["p"][0] + e["q"][0]) / 2, (e["p"][1] + e["q"][1]) / 2)


def _cross_inside(a, b, w, h):
    # True if the two (extended) lines intersect inside the photo: real parallel board edges meet only outside it
    x, y, z = np.cross(line_through(a["p"], a["q"]), line_through(b["p"], b["q"]))
    return abs(z) > 1e-9 and 0 <= x / z <= w and 0 <= y / z <= h


MIN_INSIDE = 0.5        # a chosen edge pair must have at least this fraction of the writing between it


def choose_plane_edges(lines, boxes, mask, min_inside=None):
    """Pair of non-crossing board lines around the writing, both LONG and CLOSE to it.

    Changed from the notebook (which aims at ONE target line, the box nearest the photo centre):
      * anchor = the median centre of all writing on the board, not the box nearest the photo centre
        (on oct17_040 that box was a lone word, "output", and everything else was cropped away);
      * every candidate pair is ranked by the notebook's score, and the best one with at least
        `min_inside` of the text boxes between its two lines is taken (on close-ups, Hough lines along the
        handwriting itself were paired around one row of text).
    """
    min_inside = MIN_INSIDE if min_inside is None else min_inside   # read at call time, so a changed setting counts
    h, w = mask.shape
    grow = cv2.dilate(mask, np.ones((int(0.03 * max(h, w)),) * 2, np.uint8))
    on_board = lambda e: grow[int(np.clip(_mid(e)[1], 0, h - 1)), int(np.clip(_mid(e)[0], 0, w - 1))] > 0
    centres = [b.mean(0) for b in boxes]
    in_mask = [k for k, (x, y) in enumerate(centres) if grow[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))] > 0]
    pool = in_mask or list(range(len(boxes)))
    ax_, ay_ = np.median(np.array([centres[k] for k in pool]), axis=0)
    for axis in ("h", "v"):                      # prefer top/bottom edges; fall back to left/right
        cand = [e for e in lines if e["kind"] == axis and on_board(e) and e["length"] >= MIN_PAIR_FRAC * max(h, w)]
        pos = (lambda e: y_at_x(e, ax_)) if axis == "h" else (lambda e: x_at_y(e, ay_))
        anchor = ay_ if axis == "h" else ax_
        before = [e for e in cand if pos(e) < anchor]
        after = [e for e in cand if pos(e) > anchor]
        scored = []
        for lo in before:
            for hi in after:
                if not _cross_inside(lo, hi, w, h):
                    scored.append((min(lo["length"], hi["length"]) / (pos(hi) - pos(lo)), lo, hi))
        for _, lo, hi in sorted(scored, key=lambda t: -t[0]):
            if axis == "h":
                inside = [k for k in pool if y_at_x(lo, centres[k][0]) < centres[k][1] < y_at_x(hi, centres[k][0])]
            else:
                inside = [k for k in pool if x_at_y(lo, centres[k][1]) < centres[k][0] < x_at_y(hi, centres[k][1])]
            if len(inside) < min_inside * len(pool):
                continue
            choice = dict(axis=axis, pair=(lo, hi), boxes=inside)
            other = [e for e in lines if e["kind"] != axis and e["length"] >= MIN_REF_FRAC * max(h, w)]
            choice["ref"] = max(other, key=lambda e: e["length"]) if other else None
            return choice
    return None


def refine_on_image(img_bgr, e):
    # snap a line to the strongest parallel edge in a band around it (full resolution)
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    band = 0.008 * max(h, w)
    p, q = np.array(e["p"]), np.array(e["q"])
    m = np.zeros_like(gray)
    cv2.line(m, tuple(map(int, p)), tuple(map(int, q)), 255, int(2 * band))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 10, 40) & m
    ys, xs = np.nonzero(edges)
    if len(xs) < 50:
        return e
    d = np.float32(np.c_[xs, ys])
    vx, vy, x0, y0 = cv2.fitLine(d, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    proj = lambda r: (x0 + vx * ((r[0] - x0) * vx + (r[1] - y0) * vy), y0 + vy * ((r[0] - x0) * vx + (r[1] - y0) * vy))
    return dict(e, p=proj(p), q=proj(q))


# ---------------------------------------------------------------- 4-5. rotation, homography, warp
def rotation_from_edges(K, choice, img_shape):
    # board rotation R (rows: board x, y, normal in camera coords) from an opposite edge pair + optional reference
    h, w = img_shape[:2]
    Ki = np.linalg.inv(K)
    a, b = choice["pair"]
    vp = np.cross(line_through(a["p"], a["q"]), line_through(b["p"], b["q"]))
    d1 = Ki @ vp
    d1 /= np.linalg.norm(d1)
    ref = choice["ref"]
    if choice["axis"] == "h":
        l_ref = line_through(ref["p"], ref["q"]) if ref else line_through((K[0, 2], 0), (K[0, 2], h))
        d_h = d1 * np.sign(d1[0] or 1)
        d_v = np.cross(d_h, K.T @ l_ref)
    else:
        l_ref = line_through(ref["p"], ref["q"]) if ref else line_through((0, K[1, 2]), (w, K[1, 2]))
        d_v = d1 * np.sign(d1[1] or 1)
        d_h = np.cross(d_v, K.T @ l_ref)
        d_h /= np.linalg.norm(d_h)
        d_h *= np.sign(d_h[0] or 1)
    d_v /= np.linalg.norm(d_v)
    d_v *= np.sign(d_v[1] or 1)
    return np.stack([d_h, d_v, np.cross(d_h, d_v)])


def fit_to_canvas_at(img, H, roi, anchor, max_side=None, keep_pts=None):
    """Warp so that the resolution is preserved at `anchor` (the writing).
    keep_pts (new): points that must stay inside the output (all the writing). The canvas is centred on their
    bounding box instead of the anchor, and if they span more than max_side px the whole output is scaled down
    until they fit (the far end of a long, steeply angled board then loses some resolution instead of being
    cut off)."""
    max_side = MAX_SIDE if max_side is None else max_side
    x0, y0, x1, y1 = roi
    ax, ay, d = anchor[0], anchor[1], 5.0
    p = cv2.perspectiveTransform(np.float32([[[ax, ay]], [[ax + d, ay]], [[ax, ay + d]]]), H).reshape(-1, 2)
    J = np.stack([(p[1] - p[0]) / d, (p[2] - p[0]) / d], axis=1)
    s = 1.0 / np.sqrt(abs(np.linalg.det(J)))
    corners = cv2.perspectiveTransform(np.float32([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]]), H).reshape(-1, 2)
    if keep_pts is not None:
        kp = cv2.perspectiveTransform(np.float32(keep_pts).reshape(1, -1, 2), H)[0]
        lo, hi = kp.min(0), kp.max(0)
        pad = 0.05 * (hi - lo).max()
        span = float((hi - lo).max() + 2 * pad)
        s = min(s, max_side / max(span, 1e-6))
        centre = (lo + hi) / 2 * s
    else:
        centre = p[0] * s
    corners = corners * s
    bx0, by0 = np.maximum(corners.min(0), centre - max_side / 2)
    bx1, by1 = np.minimum(corners.max(0), centre + max_side / 2)
    H_out = np.array([[s, 0, -bx0], [0, s, -by0], [0, 0, 1.0]]) @ H
    out = cv2.warpPerspective(img, H_out, (max(1, int(bx1 - bx0)), max(1, int(by1 - by0))),
                              flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    return out, H_out


def crop_box(shape, pts, margin=0.6, min_px=150):
    """(x0, y0, x1, y1) around the points, grown by `margin` x their size (at least min_px)."""
    h, w = shape[:2]
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    mx, my = max(margin * (x1 - x0), min_px), max(margin * (y1 - y0), min_px)
    return (int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(w, x1 + mx)), int(min(h, y1 + my)))


def crop_to_text(img, pts, margin=0.6):
    x0, y0, x1, y1 = crop_box(img.shape, pts, margin)
    return img[y0:y1, x0:x1]


def order_quad(points):
    # top-left, top-right, bottom-right, bottom-left
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) != 4:
        pts = cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    ordered[0], ordered[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    ordered[1], ordered[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return ordered


def rectify_from_text_box(img_bgr, boxes, margin=1.15):
    # close-up fallback of the notebook: rectify the largest text quadrilateral only
    valid = [np.asarray(b, dtype=np.float32).reshape(-1, 2) for b in boxes if np.asarray(b).size >= 8]
    if not valid:
        return None, None, None
    src = order_quad(max(valid, key=lambda b: abs(cv2.contourArea(b))))
    centre = src.mean(axis=0)
    src = centre + margin * (src - centre)
    tl, tr, br, bl = src
    out_w = max(2, int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))))
    out_h = max(2, int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))))
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    H = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img_bgr, H, (out_w, out_h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return out, H, src


def between_mask(shape, choice):
    # pixels between the two chosen edges (the band of the board that holds the writing)
    h, w = shape[:2]
    a, b = choice["pair"]
    if choice["axis"] == "h":
        poly = [(0, y_at_x(a, 0)), (w, y_at_x(a, w)), (w, y_at_x(b, w)), (0, y_at_x(b, 0))]
    else:
        poly = [(x_at_y(a, 0), 0), (x_at_y(b, 0), 0), (x_at_y(b, h), h), (x_at_y(a, h), h)]
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.int32(np.clip(poly, -10 * max(h, w), 10 * max(h, w)))], 255)
    return m


MAX_BOARD_ANGLE = 75.0   # a board turned further than this is almost certainly a wrong edge pair
MIN_TEXT_KEPT = 0.9      # a rectification must keep at least this fraction of the text-box corners in the image


# Conservative, label-free proposal thresholds. Missing evidence means ABSTAIN,
# not 'frontal'. These are not a calibrated side-view classifier or accuracy claim.
SIDE_MIN_CONVERGENCE = 4.0   # degrees between top and bottom; roll alone stays parallel
SIDE_MIN_TAPER = 1.15        # board height must change >=15% over supported edge overlap
SIDE_MIN_EDGE = 0.25         # each observed frame segment / image long side
SIDE_MIN_OVERLAP = 0.25      # observed horizontal overlap / image width


def side_view_evidence(img_shape, boxes, choice, mask, f35):
    """Permit automatic yaw rectification only with supported board-frame evidence.

    Two mask-boundary top/bottom edges must converge, have substantial observed
    overlap and bracket ALL detected writing. A third boundary segment must join
    both as a real observed frame corner. Text boxes are NOT board corners.
    """
    metrics = {}
    def verdict(ok, reason):
        return dict(approved=bool(ok), reason=reason, metrics=metrics)
    if f35 is None or not np.isfinite(f35) or f35 <= 0:
        return verdict(False, "camera focal length unknown: abstain")
    if choice is None or choice.get("axis") != "h":
        return verdict(False, "need top and bottom board-frame edges: abstain")
    a, b = choice["pair"]
    ref = choice.get("ref")
    if ref is None:
        return verdict(False, "no observed board-frame corner/reference: abstain")
    if len(keep_boundary_lines([a, b, ref], mask)) != 3:
        return verdict(False, "edge/reference is not supported by board boundary: abstain")
    h, w = img_shape[:2]
    if min(float(e["length"]) for e in (a, b, ref)) < SIDE_MIN_EDGE * max(h, w):
        return verdict(False, "observed board-frame segments too short: abstain")
    dirs = [np.asarray(e["q"], float) - e["p"] for e in (a, b)]
    if min(abs(d[0]) for d in dirs) < 1e-6:
        return verdict(False, "invalid top/bottom directions: abstain")
    cosine = np.clip(abs(np.dot(*dirs)) / (np.linalg.norm(dirs[0]) * np.linalg.norm(dirs[1])), 0, 1)
    convergence = float(np.degrees(np.arccos(cosine)))
    metrics["convergence_deg"] = round(convergence, 3)
    if convergence < SIDE_MIN_CONVERGENCE:
        return verdict(False, "frame edges nearly parallel: no strong side-view evidence")
    left = max(min(e["p"][0], e["q"][0]) for e in (a, b))
    right = min(max(e["p"][0], e["q"][0]) for e in (a, b))
    overlap = (right - left) / w
    metrics["edge_overlap_fraction"] = round(float(overlap), 3)
    if overlap < SIDE_MIN_OVERLAP:
        return verdict(False, "too little observed edge overlap: abstain")
    gaps = [float(y_at_x(b, x) - y_at_x(a, x)) for x in (left, right)]
    if min(gaps) <= 0 or _cross_inside(a, b, w, h):
        return verdict(False, "frame edges cross or reverse: abstain")
    taper = max(gaps) / min(gaps)
    metrics["height_taper"] = round(taper, 3)
    if taper < SIDE_MIN_TAPER:
        return verdict(False, "insufficient frame taper: no strong side-view evidence")
    # The reference must actually span the two observed junctions, not just be
    # a vertical letter or a room line whose extension intersects somewhere.
    rp, rq = np.asarray(ref["p"], float), np.asarray(ref["q"], float)
    rd = rq - rp
    denom = float(rd @ rd)
    if denom < 1:
        return verdict(False, "invalid reference segment: abstain")
    for edge in (a, b):
        cross = np.cross(line_through(edge["p"], edge["q"]), line_through(rp, rq))
        if abs(cross[2]) < 1e-9:
            return verdict(False, "reference does not join both board edges: abstain")
        point = cross[:2] / cross[2]
        t = float((point - rp) @ rd / denom)
        ep, eq = np.asarray(edge["p"], float), np.asarray(edge["q"], float)
        ed = eq - ep
        u = float((point - ep) @ ed / max(float(ed @ ed), 1))
        if not (-0.1 <= t <= 1.1 and -0.1 <= u <= 1.1):
            return verdict(False, "frame junction lies beyond observed segments: abstain")
    text = np.vstack(boxes)
    inside = [(y_at_x(a, p[0]) - 0.01 * h <= p[1] <= y_at_x(b, p[0]) + 0.01 * h) for p in text]
    metrics["writing_inside_fraction"] = float(np.mean(inside))
    if not all(inside):
        return verdict(False, "frame pair does not bracket every detected text box: abstain")
    return verdict(True, "supported converging board frame: side-view candidate")


def auto_rectify(img_bgr, boxes, f35=None, text_box_fallback=False, strict=True, proposal_search=False):
    """Rectify the board the photo is aimed at.

    img_bgr : upright photo.   boxes : text boxes [(4,2) arrays, full-res coords] from the detector.
    f35     : 35 mm-equivalent focal length (focal35_from_exif(path)); DEFAULT_F35 if None.
    strict  : default True requires known focal length, converging supported frame
              edges and an observed corner before any warp. Missing/cropped frame
              means abstain, not a reliable declaration that the photo is frontal.
              False is an explicit legacy opt-out, not used by the normal pipeline.
    Returns info dict: status, rectified (or None), H (photo -> rectified), angle (board turn, deg),
    text_rect (the text points in the rectified image), mask, choice.
    """
    if proposal_search:
        return auto_rectify_supported(img_bgr, boxes, f35)
    h, w = img_bgr.shape[:2]
    info = dict(status="", rectified=None, H=None, angle=np.nan, mask=None, choice=None, text_rect=None)
    if not boxes:
        info["status"] = "no text found"
        return info
    info["mask"] = mask = segment_board_floodfill(img_bgr, boxes)
    if mask is None:
        info["status"] = "no whiteboard found"
        return info
    lines = detect_long_lines(img_bgr)
    boundary_lines = keep_boundary_lines(lines, mask)
    choice = choose_plane_edges(boundary_lines, boxes, mask)
    if choice is None and not strict:
        choice = choose_plane_edges(lines, boxes, mask)  # historical, explicit opt-out only
    if choice is None:
        if text_box_fallback and not strict:
            out, H, _ = rectify_from_text_box(img_bgr, boxes)
            if out is not None:
                info.update(rectified=out, H=H, status="largest text box only (no board edges)",
                            text_rect=cv2.perspectiveTransform(np.float32(np.vstack(boxes))[None], H)[0])
                return info
        info["status"] = "no board edges around the writing"
        return info
    info["choice"] = choice
    if strict:
        gate = side_view_evidence(img_bgr.shape, boxes, choice, mask, f35)
        info["side_view_gate"] = gate
        if not gate["approved"]:
            info["status"] = gate["reason"]
            return info
    # all the writing on the board (not only the part between the two edges): the crop must keep every line
    on = [b for b in boxes if mask[int(np.clip(b[:, 1].mean(), 0, h - 1)), int(np.clip(b[:, 0].mean(), 0, w - 1))] > 0]
    text = np.vstack(on or boxes)
    choice["pair"] = tuple(refine_on_image(img_bgr, e) for e in choice["pair"])
    if choice["ref"] is not None:
        choice["ref"] = refine_on_image(img_bgr, choice["ref"])
    info["choice"] = choice
    if strict:
        gate = side_view_evidence(img_bgr.shape, boxes, choice, mask, f35)
        info["side_view_gate"] = gate
        if not gate["approved"]:
            info["status"] = "refined edges rejected: " + gate["reason"]
            return info
    K = camera_matrix(w, h, f35 or DEFAULT_F35)
    R = rotation_from_edges(K, choice, img_bgr.shape)
    info["angle"] = float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))
    if not np.isfinite(info["angle"]) or info["angle"] > MAX_BOARD_ANGLE:
        info["status"] = f"implausible board angle {info['angle']:.0f} deg: left unrectified"
        return info
    H = K @ R @ np.linalg.inv(K)
    # the plane = the segmented board (the notebook also cut it to the band between the two edges; that band can
    # leave out part of the writing now that the pair only needs MIN_INSIDE of it, so the band is not used)
    plane = cv2.dilate(mask, np.ones((int(0.01 * max(h, w)),) * 2, np.uint8))
    board_only = img_bgr.copy()
    board_only[plane == 0] = 255
    ys, xs = np.nonzero(plane)
    roi = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    info["rectified"], info["H"] = fit_to_canvas_at(board_only, H, roi, anchor=text.mean(0), keep_pts=text)
    info["text_rect"] = cv2.perspectiveTransform(np.float32(text)[None], info["H"])[0]
    rh, rw = info["rectified"].shape[:2]
    tr = info["text_rect"]
    kept = float(np.mean((tr[:, 0] >= 0) & (tr[:, 0] < rw) & (tr[:, 1] >= 0) & (tr[:, 1] < rh)))
    required_kept = 1.0 if strict else MIN_TEXT_KEPT
    if kept < required_kept:
        info.update(rectified=None, H=None, text_rect=None,
                    status=f"rectification would cut off the writing ({kept:.0%} kept): left unrectified")
        return info
    info["status"] = (f"rectified from {'top/bottom' if choice['axis'] == 'h' else 'left/right'} edges + "
                      + ("reference line" if choice["ref"] else "no-roll assumption"))
    return info


# Revised, label-free proposal search. The historical strict route above stays
# available for reproduction. Neither manual annotations nor filenames are used.
FRAME_MIN_TARGET_AREA = 0.90
FRAME_MIN_VERTICAL = 0.12
TEXT_SMALL_HEIGHT = 16.0       # provisional ink-height proxy in native source pixels
TEXT_KEEP_SMALL = 0.95         # small writing must retain this share of ink height


def _intersection(a, b):
    p = np.cross(line_through(a['p'],a['q']),line_through(b['p'],b['q']))
    return None if abs(p[2])<1e-8 else p[:2]/p[2]


def _segment_parameter(e, point):
    p,q=np.asarray(e['p']),np.asarray(e['q']);d=q-p
    return float((point-p)@d/max(d@d,1))


def _frame_reference(top, bottom, ref, shape, edge_distance=None, edge_scale=1.):
    """Two observed joins, or one join plus an edge clipped at the image border.

    The latter is explicit geometric extrapolation, not an invented observed
    corner. Interior letters/room lines without a supported join cannot qualify.
    """
    h,w=shape[:2];points=[_intersection(e,ref) for e in (top,bottom)]
    if any(p is None for p in points):return False
    observed=[]
    for e,p in zip((top,bottom),points):
        on_edge=-.12<=_segment_parameter(e,p)<=1.12
        on_ref=-.12<=_segment_parameter(ref,p)<=1.12
        # A detector may stop a real frame segment early. Extend it only when
        # the intervening image pixels supply continuous edge evidence.
        if not on_edge and on_ref and edge_distance is not None:
            endpoint=min((np.asarray(e[k]) for k in ('p','q')),key=lambda q:np.linalg.norm(q-p))
            if np.linalg.norm(endpoint-p)<.15*max(h,w):
                samples=(endpoint+np.linspace(0,1,30)[:,None]*(p-endpoint))*edge_scale
                valid=(samples[:,0]>=0)&(samples[:,0]<edge_distance.shape[1])&(samples[:,1]>=0)&(samples[:,1]<edge_distance.shape[0])
                if valid.all():
                    ix=samples.astype(int);on_edge=np.mean(edge_distance[ix[:,1],ix[:,0]]<6)>=.8
        observed.append(on_edge and on_ref)
    if all(observed):return True
    if sum(observed)!=1:return False
    j=observed.index(False);p=points[j]
    outside=p[0]<0 or p[0]>=w or p[1]<0 or p[1]>=h
    near=np.min([np.linalg.norm(p-np.asarray(e[k])) for e in ((top,bottom)[j],ref) for k in ('p','q')])
    extrap=max(abs(_segment_parameter(ref,p)-.5)-.5,0)
    return bool(outside and near<.15*max(h,w) and extrap<.35)


def _box_panel_fraction(box, polygon):
    area=abs(cv2.contourArea(np.float32(box)))
    return float(cv2.intersectConvexConvex(np.float32(box),np.float32(polygon))[0]/max(area,1))


def detect_frame_lines(image):
    """Independent LSD proposals plus Hough; merge by endpoint distance to the
    seed line, retaining every supported run instead of discarding all but one.
    """
    h,w=image.shape[:2];scale=LINE_WORK_SIDE/max(h,w)
    gray=cv2.cvtColor(cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
    found=cv2.createLineSegmentDetector().detect(gray)[0]
    segments=[]
    if found is not None:
        for row in found.reshape(-1,4):
            p,q=row[:2]/scale,row[2:]/scale
            if np.linalg.norm(q-p)>=.04*max(h,w):segments.append((p,q))
    segments += [(np.asarray(e['p']),np.asarray(e['q'])) for e in detect_long_lines(image)]
    used=set();result=[];long=max(h,w)
    for i in sorted(range(len(segments)),key=lambda i:-np.linalg.norm(segments[i][1]-segments[i][0])):
        if i in used:continue
        p,q=segments[i];direction=(q-p)/np.linalg.norm(q-p);normal=np.array([-direction[1],direction[0]])
        group=[]
        for j,(a,b) in enumerate(segments):
            if j in used:continue
            d=(b-a)/max(np.linalg.norm(b-a),1)
            if abs(direction@d)<np.cos(np.radians(1.2)):continue
            if max(abs((a-p)@normal),abs((b-p)@normal))>.004*long:continue
            u,v=sorted((float(a@direction),float(b@direction)));group.append((u,v,j,a,b));used.add(j)
        runs=[];end=None
        for span in sorted(group,key=lambda s:s[0]):
            if end is None or span[0]-end>MAX_GAP_FRAC*long:runs.append([])
            runs[-1].append(span);end=max(end or span[1],span[1])
        for run in runs:
            extent=max(s[1] for s in run)-min(s[0] for s in run)
            if extent<MIN_LINE_FRAC*long:continue
            pts=np.float32([r[k] for r in run for k in (3,4)])
            vx,vy,x,y=cv2.fitLine(pts,cv2.DIST_HUBER,0,.01,.01).ravel();t=(pts-[x,y])@[vx,vy]
            a=np.array([x,y])+t.min()*np.array([vx,vy]);b=np.array([x,y])+t.max()*np.array([vx,vy])
            result.append(dict(p=a,q=b,length=float(np.linalg.norm(b-a)),kind='v' if abs(vy)>np.cos(np.radians(VERTICAL_TOL)) else 'h'))
    return result


def frame_proposals(image, boxes, mask, lines=None):
    """Search physical boundary triplets, rather than accepting the first pair
    and the globally longest vertical line. Permit a long internal panel joint
    only with an outline-supported lower edge and vertical frame reference.
    """
    h,w=image.shape[:2];long=max(h,w)
    lines=detect_frame_lines(image) if lines is None else lines
    boundary=keep_boundary_lines(lines,mask)
    edge_scale=1000/long
    small=cv2.resize(image,None,fx=edge_scale,fy=edge_scale,interpolation=cv2.INTER_AREA)
    edges=cv2.Canny(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),20,60)
    distance=cv2.distanceTransform(255-edges,cv2.DIST_L2,3)
    boundary_ids={id(e) for e in boundary}
    grown=cv2.dilate(mask,np.ones((max(3,int(.015*long)),)*2,np.uint8))
    def on_board(e):
        points=np.asarray(e['p'])+np.linspace(0,1,30)[:,None]*(np.asarray(e['q'])-e['p'])
        points=np.clip(points,[0,0],[w-1,h-1]).astype(int)
        return np.mean(grown[points[:,1],points[:,0]]>0)>=.60
    horizontals=[e for e in lines if e['kind']=='h' and e['length']>=SIDE_MIN_EDGE*long and on_board(e)]
    verticals=[e for e in lines if e['kind']=='v' and e['length']>=FRAME_MIN_VERTICAL*long]
    areas=np.array([abs(cv2.contourArea(np.float32(b))) for b in boxes]);total=max(areas.sum(),1)
    ax=float(np.median(np.vstack(boxes)[:,0]));proposals=[]
    for top in horizontals:
        for bottom in horizontals:
            if top is bottom or y_at_x(top,ax)>=y_at_x(bottom,ax):continue
            if _cross_inside(top,bottom,w,h):continue
            if id(bottom) not in boundary_ids:continue
            # Only long grey physical segments may act as an internal joint.
            for edge in (top,bottom):
                ts=np.linspace(.02,.98,60);ps=np.asarray(edge['p'])+ts[:,None]*(np.asarray(edge['q'])-edge['p'])
                ps=np.clip(ps,[0,0],[w-1,h-1]).astype(int)
                sat=cv2.cvtColor(image[ps[:,1],ps[:,0]][None],cv2.COLOR_BGR2HSV)[0,:,1]
                if np.mean(sat>70)>.25:break
            else:
                d1=np.asarray(top['q'])-top['p'];d2=np.asarray(bottom['q'])-bottom['p']
                angle=np.degrees(np.arccos(np.clip(abs(d1@d2)/(np.linalg.norm(d1)*np.linalg.norm(d2)),0,1)))
                if angle<SIDE_MIN_CONVERGENCE:continue
                left=max(min(top['p'][0],top['q'][0]),min(bottom['p'][0],bottom['q'][0]))
                right=min(max(top['p'][0],top['q'][0]),max(bottom['p'][0],bottom['q'][0]))
                if right-left<SIDE_MIN_OVERLAP*w:continue
                gaps=[y_at_x(bottom,x)-y_at_x(top,x) for x in (left,right)]
                if min(gaps)<=0 or max(gaps)/min(gaps)<SIDE_MIN_TAPER:continue
                refs=[]
                for r in verticals:
                    if not _frame_reference(top,bottom,r,image.shape,distance,edge_scale):continue
                    a,b=_intersection(top,r),_intersection(bottom,r)
                    clipped=dict(r,p=a,q=b,length=float(np.linalg.norm(b-a)))
                    if keep_boundary_lines([clipped],mask):refs.append(r)
                if not refs:continue
                # Enumerate left/right support independently; don't assume the
                # longest room vertical is the board side.
                for ref in refs:
                    xt=float(x_at_y(ref,float(np.median(np.vstack(boxes)[:,1]))))
                    candidates_left=[r for r in refs if x_at_y(r,h/2)<ax]
                    candidates_right=[r for r in refs if x_at_y(r,h/2)>ax]
                    le=max(candidates_left,key=lambda r:x_at_y(r,h/2),default=None)
                    re=min(candidates_right,key=lambda r:x_at_y(r,h/2),default=None)
                    corners=[_intersection(top,le) if le else [0,y_at_x(top,0)],
                             _intersection(top,re) if re else [w-1,y_at_x(top,w-1)],
                             _intersection(bottom,re) if re else [w-1,y_at_x(bottom,w-1)],
                             _intersection(bottom,le) if le else [0,y_at_x(bottom,0)]]
                    if any(p is None for p in corners):continue
                    polygon=np.float32(corners)
                    if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon):continue
                    inside=np.array([_box_panel_fraction(b,polygon)>=.90 for b in boxes])
                    area_kept=float(areas[inside].sum()/total)
                    if inside.sum()<3 or area_kept<FRAME_MIN_TARGET_AREA:continue
                    # A panel must agree with the flood-fill estimate. A room
                    # rectangle is not validated merely by containing text.
                    canvas=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(canvas,np.int32(polygon),255)
                    overlap=float(np.count_nonzero((canvas>0)&(mask>0))/max(np.count_nonzero(canvas),1))
                    if overlap<.55:continue
                    rank=area_kept+min(top['length'],bottom['length'])/long+.2*(le is not None and re is not None)+.1*overlap+.35*(id(top) in boundary_ids)
                    proposals.append(dict(axis='h',pair=(top,bottom),ref=ref,polygon=polygon,
                        target_indices=np.flatnonzero(inside).tolist(),rank=float(rank),
                        evidence=dict(convergence_deg=float(angle),height_taper=float(max(gaps)/min(gaps)),
                            retained_box_area=area_kept,mask_overlap=overlap,observed_sides=int(le is not None)+int(re is not None))))
    proposals.sort(key=lambda p:-p['rank'])
    unique=[]
    for p in proposals:
        if not any(np.max(np.abs(p['polygon']-q['polygon']))<.015*long for q in unique):unique.append(p)
    return unique[:8]


def ink_height_proxy(image, box):
    """Connected ink-component height; a readability proxy, not character truth."""
    q=np.float32(box);width=max(8,int(max(np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3]))))
    height=max(8,int(max(np.linalg.norm(q[3]-q[0]),np.linalg.norm(q[2]-q[1]))))
    s=min(1.,900/width,200/height);cw,ch=max(8,int(width*s)),max(8,int(height*s))
    m=cv2.getPerspectiveTransform(q,np.float32([[0,0],[cw-1,0],[cw-1,ch-1],[0,ch-1]]))
    crop=cv2.warpPerspective(image,m,(cw,ch),borderValue=(255,255,255))
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg=cv2.GaussianBlur(gray,(0,0),max(3,ch/2));ink=(gray/np.maximum(bg,1)<.80).astype(np.uint8)
    _,_,stats,_=cv2.connectedComponentsWithStats(ink)
    heights=[r[cv2.CC_STAT_HEIGHT] for r in stats[1:] if r[cv2.CC_STAT_AREA]>=3 and .15*ch<=r[cv2.CC_STAT_HEIGHT]<=.98*ch]
    return float(np.percentile(heights,75)/s) if heights else None


def preservation_gate(image, boxes, H, shape):
    rows=[];required_boost=1.
    for box in boxes:
        b=np.float32(box);t=cv2.perspectiveTransform(b[None],H)[0]
        old=(np.linalg.norm(b[0]-b[3])+np.linalg.norm(b[1]-b[2]))/2
        new=(np.linalg.norm(t[0]-t[3])+np.linalg.norm(t[1]-t[2]))/2
        source=ink_height_proxy(image,b)
        unknown=source is None
        # No reliable glyph estimate: retain the original box height instead
        # of treating an uncertain/faint line as permission to shrink it.
        if unknown:source=float(old)
        target=source*new/max(old,1);floor=(source if unknown else min(source,TEXT_SMALL_HEIGHT))*TEXT_KEEP_SMALL
        required_boost=max(required_boost,floor/max(target,1e-6))
        rows.append(dict(source_ink_px=None if unknown else source,proposed_ink_px=float(target),required_ink_px=float(floor),
                         box_height_fallback=unknown,height_scale=float(new/max(old,1))))
    fits=required_boost*max(shape[:2])<=MAX_SIDE
    return dict(approved=bool(fits),reason='ink-height floor retained' if fits else 'readability would exceed output canvas limit',
                canvas_scale=float(required_boost),rows=rows)


def fit_panel_canvas(source, H, polygon, points):
    """Bound the observed panel itself, not its axis-aligned source rectangle.
    Bounding-rectangle corners can lie far outside a slanted board and near the
    homography horizon, wasting the canvas on an enormous white triangle.
    """
    h,w=source.shape[:2];frame=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    area,visible=cv2.intersectConvexConvex(np.float32(polygon),frame)
    if area<=0 or visible is None:raise ValueError('No observed panel inside photo')
    transformed=cv2.perspectiveTransform(np.float32(np.vstack([visible.reshape(-1,2),points]))[None],H)[0]
    if not np.isfinite(transformed).all():raise ValueError('Non-finite panel projection')
    anchor=points.mean(0);rays=np.float32([[anchor,anchor+[5,0],anchor+[0,5]]])
    projected=cv2.perspectiveTransform(rays,H)[0]
    jacobian=np.stack([(projected[1]-projected[0])/5,(projected[2]-projected[0])/5],axis=1)
    scale=1/max(np.sqrt(abs(np.linalg.det(jacobian))),1e-6)
    lo,hi=transformed.min(0),transformed.max(0);pad=max(5.,.02*float((hi-lo).max()))
    lo-=pad;hi+=pad;scale=min(scale,(MAX_SIDE-2)/max(float((hi-lo).max()),1))
    output=np.array([[scale,0,-lo[0]*scale],[0,scale,-lo[1]*scale],[0,0,1.]])@H
    size=np.maximum(1,np.ceil((hi-lo)*scale).astype(int))
    image=cv2.warpPerspective(source,output,tuple(size),flags=cv2.INTER_CUBIC,borderValue=(255,255,255))
    return image,output


def auto_rectify_supported(image, boxes, f35):
    info=dict(status='no supported frame proposal',rectified=None,H=None,angle=np.nan,mask=None,choice=None,text_rect=None)
    if not boxes or f35 is None or not np.isfinite(f35) or f35<=0:
        info['status']='text or camera information unavailable: abstain';return info
    mask=segment_board_floodfill(image,boxes);info['mask']=mask
    if mask is None:return info
    proposals=frame_proposals(image,boxes,mask)
    info['proposal_count']=len(proposals);rejections=[]
    for choice in proposals:
        k=camera_matrix(image.shape[1],image.shape[0],f35)
        r=rotation_from_edges(k,choice,image.shape)
        angle=float(np.degrees(np.arccos(np.clip(abs(r[2,2]),0,1))))
        if not np.isfinite(r).all() or not np.isfinite(angle) or not 0<=angle<=MAX_BOARD_ANGLE:continue
        target=[boxes[i] for i in choice['target_indices']];points=np.vstack(target)
        plane=np.zeros(image.shape[:2],np.uint8);cv2.fillConvexPoly(plane,np.int32(choice['polygon']),255)
        plane=cv2.dilate(plane,np.ones((7,7),np.uint8));source=image.copy();source[plane==0]=255
        result,H=fit_panel_canvas(source,k@r@np.linalg.inv(k),choice['polygon'],points)
        gate=preservation_gate(image,target,H,result.shape)
        if not gate['approved']:rejections.append(gate['reason']);continue
        boost=gate['canvas_scale']
        if boost>1.001:
            H=np.diag([boost,boost,1.])@H
            result=cv2.warpPerspective(source,H,(int(np.ceil(result.shape[1]*boost)),int(np.ceil(result.shape[0]*boost))),flags=cv2.INTER_CUBIC,borderValue=(255,255,255))
        tp=cv2.perspectiveTransform(np.float32(points)[None],H)[0]
        if not ((tp>=0).all() and (tp[:,0]<result.shape[1]).all() and (tp[:,1]<result.shape[0]).all()):
            rejections.append('writing outside output canvas');continue
        info.update(rectified=result,H=H,angle=angle,text_rect=tp,choice=choice,
            target_indices=choice['target_indices'],excluded_indices=[i for i in range(len(boxes)) if i not in choice['target_indices']],
            readability_gate=gate,side_view_gate=dict(approved=True,reason='supported physical frame/joint proposal',metrics=choice['evidence']),
            status='supported frame and ink-height preservation passed')
        return info
    info['rejections']=rejections
    return info