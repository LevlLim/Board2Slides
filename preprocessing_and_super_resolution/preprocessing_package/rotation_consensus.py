"""Conservative small-roll correction from independent detected text rows.

No OCR text/confidence is used. A box direction must agree with a robust fit
of glyph-component centres in its pixels. Three spatially independent rows
must agree before a single expanded-canvas rotation is allowed. Ambiguous
cursive/short/faint text is deliberately left unchanged.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

MIN_LINES = 3
MIN_AGREEMENT = 0.70
ANGLE_TOL = 2.0
PIXEL_BOX_TOL = 2.5
MIN_COMPONENTS = 5
MIN_ASPECT = 3.0


def _pixel_direction(image, quad, line_h):
    """Robust centre-to-centre slope, not the slant of a letter's strokes."""
    h, w = image.shape[:2]
    lo = np.floor(quad.min(0) - .15 * line_h).astype(int)
    hi = np.ceil(quad.max(0) + .15 * line_h).astype(int)
    x0, y0 = np.maximum(lo, 0)
    x1, y1 = np.minimum(hi, [w, h])
    if x1 <= x0 or y1 <= y0:
        return None, "empty text region"
    crop = image[y0:y1, x0:x1]
    scale = min(1., 1200 / crop.shape[1], 400 / crop.shape[0])
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lh = line_h * scale
    background = cv2.GaussianBlur(gray, (0, 0), max(3., lh))
    ink = ((gray / np.maximum(background, 1)) < .80).astype(np.uint8)
    # Restrict evidence to the detector's actual polygon: outside dividers and
    # room edges cannot vote even if they fall inside its axis-aligned bounds.
    region = np.zeros_like(ink)
    cv2.fillConvexPoly(region, np.int32((quad - [x0, y0]) * scale), 1)
    ink &= region
    _, _, stats, centres = cv2.connectedComponentsWithStats(ink)
    valid = [i for i, (x, y, cw, ch, area) in enumerate(stats[1:], 1)
             if area >= max(3, .015 * lh * lh) and .18 * lh <= ch <= 1.6 * lh
             and .04 * lh <= cw <= 2.5 * lh]
    points = centres[valid]
    if len(points) < MIN_COMPONENTS:
        return None, "too few independent glyph components"
    span = float(np.ptp(points[:, 0]))
    if span < .55 * (x1 - x0) * scale:
        return None, "glyph evidence covers too little of the line"
    slopes = [(points[j, 1] - points[i, 1]) / (points[j, 0] - points[i, 0])
              for i in range(len(points)) for j in range(i + 1, len(points))
              if abs(points[j, 0] - points[i, 0]) >= .25 * span]
    if not slopes:
        return None, "insufficient component separation"
    slope = float(np.median(slopes))
    intercept = float(np.median(points[:, 1] - slope * points[:, 0]))
    residual = float(np.median(np.abs(points[:, 1] - slope * points[:, 0] - intercept)))
    if residual > .25 * lh:
        return None, "components do not form a straight text row"
    return float(np.degrees(np.arctan(slope))), "pixel baseline supported"


def estimate(image, boxes, min_angle=2., max_angle=15.):
    """Return an auditable decision; confidence is agreement, NOT a probability."""
    info = dict(angle=0., confidence=0., used=False, applied=False,
                method="independent-text-row-consensus", reason="insufficient independent text rows",
                candidates=[], independent_rows=0, agreeing_rows=0, stacked_rows=0)
    if boxes is None:
        info["reason"] = "no detected text boxes supplied: abstain"
        return info
    candidates = []
    for index, box in enumerate(boxes):
        try:
            q = np.asarray(box, np.float32).reshape(-1, 2)
        except (ValueError, TypeError):
            info["reason"] = "invalid detected geometry: abstain"
            return info
        if q.shape != (4, 2) or not np.isfinite(q).all():
            info["reason"] = "invalid detected geometry: abstain"
            return info
        direction = ((q[1] - q[0]) + (q[2] - q[3])) / 2
        if direction[0] < 0:
            direction = -direction
        width = float(np.linalg.norm(direction))
        height = float((np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2)
        angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
        if height < 8 or width < 80 or width < MIN_ASPECT * height or abs(angle) > 45:
            continue
        pixel, reason = _pixel_direction(image, q, height)
        # Keep large/conflicting horizontal rows in the denominator. Dropping
        # them would let a small tilted minority manufacture a consensus.
        small_roll = abs(angle) <= max_angle
        agreed = small_roll and pixel is not None and abs(pixel - angle) <= PIXEL_BOX_TOL
        if not small_roll:
            reason = "box angle outside small-roll range"
        elif pixel is not None and not agreed:
            reason = "pixel/box direction disagreement"
        row = dict(index=index, box_angle=round(angle, 3), pixel_angle=pixel,
                   reliable=bool(agreed), reason=reason,
                   height=height, width=width, centre=q.mean(0).tolist())
        info["candidates"].append(row)
        candidates.append(row)
    if not candidates:
        return info
    # Fragments/duplicate boxes from the SAME physical row must not manufacture
    # a majority. Compare row intercepts under the median box direction.
    reference = float(np.median([r["box_angle"] for r in candidates]))
    slope = math.tan(math.radians(reference))
    groups = []
    for row in sorted(candidates, key=lambda r: -r["width"]):
        intercept = (row["centre"][1] - slope * row["centre"][0]) * math.cos(math.radians(reference))
        if any(abs(intercept - other["intercept"]) < .7 * min(row["height"], other["height"])
               for other in groups):
            continue
        groups.append(dict(row, intercept=intercept))
    info["independent_rows"] = len(groups)
    reliable = [r for r in groups if r["reliable"]]
    if len(groups) < MIN_LINES or len(reliable) < MIN_LINES:
        return info
    clusters = [[r for r in reliable if abs(r["pixel_angle"] - seed["pixel_angle"]) <= ANGLE_TOL]
                for seed in reliable]
    inliers = max(clusters, key=len)
    angle = float(np.median([r["pixel_angle"] for r in inliers]))
    # Re-evaluate around the final estimate, not just the winning seed.
    inliers = [r for r in reliable if abs(r["pixel_angle"] - angle) <= ANGLE_TOL]
    agreement = len(inliers) / len(groups)
    info.update(angle=round(angle, 3), confidence=round(agreement, 3), agreeing_rows=len(inliers))
    if len(inliers) < MIN_LINES or agreement < MIN_AGREEMENT:
        info["reason"] = "independent rows disagree: abstain"
        return info
    if np.ptp([r["intercept"] for r in inliers]) < 2 * np.median([r["height"] for r in inliers]):
        info["reason"] = "insufficient separation between text rows: abstain"
        return info
    # Three short captions across the board must not masquerade as three
    # stacked paragraph rows merely because their handwritten baselines differ.
    # Require a common horizontal span within at least one column of writing.
    # Use the baseline axis so this check is invariant to small camera roll.
    radians = math.radians(angle)
    axis = np.array([math.cos(radians), math.sin(radians)])
    intervals = [(float(np.dot(r["centre"], axis)) - .375 * r["width"],
                  float(np.dot(r["centre"], axis)) + .375 * r["width"], r) for r in inliers]
    stacked = max(([r for lo, hi, r in intervals if lo <= pivot <= hi]
                   for pivot in [v for lo, hi, _ in intervals for v in (lo, hi)]), key=len)
    info["stacked_rows"] = len(stacked)
    if (len(stacked) < MIN_LINES or
            np.ptp([r["intercept"] for r in stacked]) < 2 * np.median([r["height"] for r in stacked])):
        info["reason"] = "no column with enough separated text rows: abstain"
        return info
    if not min_angle <= abs(angle) <= max_angle:
        info["reason"] = "tilt below trigger or outside small-roll range: unchanged"
        return info
    info.update(used=True, reason="independent text rows and pixel baselines agree")
    return info


def correct(image, boxes, min_angle=2., max_angle=15.):
    info = estimate(image, boxes, min_angle, max_angle)
    if not info["used"]:
        return image, info
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), info["angle"], 1.)
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    projected = cv2.transform(corners[None], matrix)[0]
    lo, hi = np.floor(projected.min(0)) - 1, np.ceil(projected.max(0)) + 1
    matrix[:, 2] -= lo
    nw, nh = np.int32(hi - lo)
    text = np.vstack([np.asarray(b, np.float32).reshape(4, 2) for b in boxes])
    mapped = cv2.transform(text[None], matrix)[0]
    if not np.all((mapped[:, 0] >= 0) & (mapped[:, 0] < nw) &
                  (mapped[:, 1] >= 0) & (mapped[:, 1] < nh)):
        info.update(used=False, reason="rotation would clip detected writing: unchanged")
        return image, info
    border = np.median(np.vstack([image[0], image[-1], image[:, 0], image[:, -1]]), axis=0)
    rotated = cv2.warpAffine(image, matrix, (int(nw), int(nh)), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(float(v) for v in border))
    info.update(applied=True, transform=matrix.tolist(), output_shape=[int(nh), int(nw)])
    return rotated, info
