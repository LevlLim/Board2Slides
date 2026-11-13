"""Embedded single-step preprocessing: same implementation as the audit."""
from __future__ import annotations
import time
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from . import board_geometry as geo, text_lines as tl
from . import fix_lighting, rotation_consensus, line_grouping

MAX_ROLL = 15.0

REGION_MARGIN = 0.6

DIRECTIONAL_GROUPING = False

@dataclass(frozen=True)
class Steps:
    lighting_light: bool = False
    perspective: bool = False
    rotation: bool = False
    lighting_strong: bool = False
    blemish: bool = False
    crop_region: bool = True      # cut out the text region before step 5 (needed for rotation/strong lighting)
    unclip: float = 2.0           # detector box growth; 2.0 = the baseline preset (app); smaller keeps lines apart
    min_board_angle: float = 0.0  # perspective is applied only if the board is turned at least this much (deg)
    auto: bool = False            # decide per photo which steps to run (Part 1 checks); the step flags above are
                                  # then ignored
    safety_net: bool = True       # auto only: keep the corrected version only if it keeps the text and the reader is
                                  # surer of it (Part 2)
    auto_blemish: bool = False    # auto only: allow blemish removal (off in the first test: it hurt in round 1 and its
                                  # check did not pick out the blemish photos; the check is still measured and logged)
    isolated: str = ""            # round 4: exactly one optional operation, or 'baseline'

    def name(self):
        if self.isolated:
            return f"isolated {self.isolated}, unclip {self.unclip}"
        if self.auto:
            return (f"auto: per-photo checks{' + safety net' if self.safety_net else ''}"
                    f"{' (blemish allowed)' if self.auto_blemish else ''}, unclip {self.unclip}")
        on = [k for k, v in asdict(self).items()
              if v is True and k not in ("crop_region", "safety_net", "auto_blemish")]
        if self.unclip != 2.0:
            on.append(f"unclip {self.unclip}")
        if self.perspective and self.min_board_angle:
            on.append(f"perspective only if board angle >= {self.min_board_angle:g}")
        return " + ".join(on) if on else "none"

PERSP_MIN_ANGLE = 9.0

PERSP_MIN_TEXT_SCALE = 0.9

LIGHT_ISSUES = ("dim", "low_contrast")

LIGHT_MIN_ILLUM = 0.45

ROT_MIN_ANGLE = 2.0

def _gated_rotation(img, boxes=None):
    # No global projection fallback: board rulings/room edges cannot vote.
    # "confidence" in this result is a row-agreement fraction, not OCR certainty.
    return rotation_consensus.correct(img, boxes, min_angle=ROT_MIN_ANGLE, max_angle=MAX_ROLL)

def _ordered_lines(boxes):
    if DIRECTIONAL_GROUPING:
        lines, _ = line_grouping.group_into_lines(boxes, gap_factor=getattr(tl, 'GROUP_GAP', 1.2))
    else:
        lines, _ = tl.group_into_lines(boxes)
    return [lines[i] for i in tl.reading_order(lines)]

def preprocess_isolated(img_bgr, detector, steps, f35=None):
    """Round 4: a shared ROI policy plus at most ONE gated preprocessing step.

    Gates are provisional geometry/photometry heuristics, not OCR-confidence
    selection. Skipped corrections retain exactly the baseline image path.
    Region detection at native resolution is common to every arm, not counted
    as enlargement. Ordinary quadrilateral line cropping remains common too.
    """
    t0 = time.time()
    mode = steps.isolated
    if mode not in {"baseline", "perspective", "lighting", "rotation", "enlargement"}:
        raise ValueError(f"Unknown isolated step: {mode}")
    boxes, _ = detector.detect(img_bgr, board_mask_fn=geo.segment_board_floodfill, unclip=steps.unclip)
    log = dict(steps=steps.name(), isolated=mode, applied=False,
               reason="baseline: no optional correction", signals={}, rectified=False)
    if not boxes:
        log.update(reason="no initial text: no optional correction", boxes=0, lines=0,
                   seconds=round(time.time() - t0, 3))
        return dict(image=img_bgr, read_image=img_bgr, lines=[], log=log, enlarge_method="none")
    x, pts = img_bgr, np.vstack(boxes)
    if mode == "perspective":
        r = geo.auto_rectify(img_bgr, boxes, f35=f35, proposal_search=True)
        angle = float(r["angle"])
        log["signals"]["board_angle"] = angle if np.isfinite(angle) else None
        log["signals"]["side_view_gate"] = r.get("side_view_gate")
        log["signals"]["proposal_count"] = r.get("proposal_count", 0)
        log["signals"]["proposal_rejections"] = r.get("rejections", [])
        log["signals"]["excluded_box_indices"] = r.get("excluded_indices", [])
        log["signals"]["readability_gate"] = r.get("readability_gate")
        log["signals"]["initial_boxes"] = [np.asarray(b).tolist() for b in boxes]
        if r.get("choice") is not None:
            choice = r["choice"]
            log["signals"]["panel_polygon"] = np.asarray(choice["polygon"]).tolist()
            log["signals"]["frame_segments"] = [[np.asarray(e["p"]).tolist(),np.asarray(e["q"]).tolist()]
                                                   for e in (*choice["pair"],choice["ref"])]
        log["reason"] = r["status"]
        if r["rectified"] is not None and np.isfinite(angle):
            scale = float(np.median([
                tl.line_height(cv2.perspectiveTransform(np.float32(b)[None], r["H"])[0]) /
                max(tl.line_height(b), 1) for b in boxes]))
            log["signals"]["text_scale"] = scale
            preserved = r.get("readability_gate", {}).get("approved", scale >= PERSP_MIN_TEXT_SCALE)
            if angle >= PERSP_MIN_ANGLE and preserved:
                x, pts = r["rectified"], r["text_rect"]
                log.update(applied=True, rectified=True, reason="supported board frame and ink-height checks passed")
            else:
                log["reason"] = "angle or text-size gate failed: original retained"
    # Same writing-region crop policy in ALL arms, including the baseline.
    x0, y0, x1, y1 = geo.crop_box(x.shape, pts, REGION_MARGIN)
    work = x[y0:y1, x0:x1].copy()
    log["region"] = [int(v) for v in (x0, y0, x1, y1)]
    if mode == "lighting":
        rep = fix_lighting.analyze(work)
        issues = [i for i in rep["issues"] if i in LIGHT_ISSUES]
        uneven = float(rep["metrics"]["illum_ratio"])
        log["signals"] = dict(lighting_issues=issues, illum_ratio=uneven)
        if issues or uneven >= LIGHT_MIN_ILLUM:
            work, _ = fix_lighting.fix(work, strength="light")
            log.update(applied=True, reason="lighting gate passed: gentle correction")
        else:
            log["reason"] = "lighting gate not met: original retained"
    elif mode == "rotation":
        rotated, info = _gated_rotation(work, [np.float32(b) - [x0, y0] for b in boxes])
        log["signals"] = info
        if info["used"] and abs(info["angle"]) >= ROT_MIN_ANGLE:
            work = rotated
            log.update(applied=True, reason=info.get("reason", "text-row consensus passed"))
        else:
            log["reason"] = info.get("reason", "rotation gate not met: original retained")
    final_boxes, dinfo = detector.detect(work, board_mask_fn=None, unclip=steps.unclip)
    lines = _ordered_lines(final_boxes)
    if mode == "enlargement":
        n = sum(tl.line_height(p) < tl.REC_MIN_H for p in lines)
        log.update(applied=bool(n), small_lines=int(n),
                   reason=f"bicubic enlargement requested for {n} small line crops")
    log.update(boxes=len(final_boxes), lines=len(lines), detect=dinfo["path"],
               seconds=round(time.time() - t0, 3))
    return dict(image=work, read_image=work, lines=lines, log=log,
                enlarge_method="bicubic" if mode == "enlargement" else "none")

def preprocess_auto(img_bgr, detector, steps, f35=None):
    """Apply compatible gated corrections to one photo, then detect text again.

    Geometry is assessed on the full photo. The writing crop is made once, and
    rotation and lighting are assessed on that same crop. If the final detector
    loses most initially detected lines, keep the baseline crop instead.
    """
    t0 = time.time()
    boxes, _ = detector.detect(img_bgr, board_mask_fn=geo.segment_board_floodfill,
                               unclip=steps.unclip)
    decisions = []
    if not boxes:
        for name in ("perspective", "crop", "rotation", "lighting"):
            decisions.append(dict(name=name, applied=False, reason="No writing detected; original retained."))
        return dict(image=img_bgr, read_image=img_bgr, lines=[],
                    log=dict(mode="auto", applied=False, reason="No writing detected; original retained.",
                             decisions=decisions, boxes=0, lines=0,
                             seconds=round(time.time() - t0, 3)))

    initial_boxes = list(boxes)
    image = img_bgr
    try:
        rectification = geo.auto_rectify(img_bgr, boxes, f35=f35, proposal_search=True)
        angle = float(rectification["angle"])
        if rectification["rectified"] is not None and np.isfinite(angle):
            scale = float(np.median([
                tl.line_height(cv2.perspectiveTransform(np.float32(b)[None], rectification["H"])[0]) /
                max(tl.line_height(b), 1) for b in boxes]))
            readable = (rectification.get("readability_gate") or {}).get(
                "approved", scale >= PERSP_MIN_TEXT_SCALE)
            if angle >= PERSP_MIN_ANGLE and readable:
                candidate_boxes = [np.float32(box) for box in rectification["text_rect"]]
                if not candidate_boxes:
                    raise ValueError("Perspective produced no text boxes")
                image = rectification["rectified"]
                boxes = candidate_boxes
                decisions.append(dict(name="perspective", applied=True,
                                      reason="Board frame and text-size checks passed."))
            else:
                decisions.append(dict(name="perspective", applied=False,
                                      reason="Board angle or text-size check did not pass."))
        else:
            decisions.append(dict(name="perspective", applied=False,
                                  reason=rectification.get("status", "No supported board perspective found.")))
    except (ValueError, TypeError, cv2.error, KeyError) as exc:
        decisions.append(dict(name="perspective", applied=False,
                              reason=f"Geometry check could not support a correction: {type(exc).__name__}."))

    x0, y0, x1, y1 = geo.crop_box(image.shape, np.vstack(boxes), REGION_MARGIN)
    baseline = image[y0:y1, x0:x1].copy()
    work = baseline
    cropped = (x0, y0, x1, y1) != (0, 0, image.shape[1], image.shape[0])
    decisions.append(dict(name="crop", applied=cropped,
                          reason="Cropped around detected writing." if cropped else "Writing already fills the photo."))
    local_boxes = [np.float32(box) - [x0, y0] for box in boxes]

    rotated, rotation = _gated_rotation(work, local_boxes)
    if rotation.get("applied"):
        work = rotated
    decisions.append(dict(name="rotation", applied=bool(rotation.get("applied")),
                          reason=rotation.get("reason", "Rotation check did not pass.")))

    lighting = fix_lighting.analyze(work)
    issues = [issue for issue in lighting["issues"] if issue in LIGHT_ISSUES]
    uneven = float(lighting["metrics"]["illum_ratio"])
    if min(work.shape[:2]) < 120:
        decisions.append(dict(name="lighting", applied=False,
                              reason="A narrow text crop is too small to judge lighting safely."))
    elif issues or uneven >= LIGHT_MIN_ILLUM:
        work, _ = fix_lighting.fix(work, strength="light")
        decisions.append(dict(name="lighting", applied=True,
                              reason="Gentle correction for low contrast, dimness, or uneven light."))
    else:
        decisions.append(dict(name="lighting", applied=False,
                              reason="Lighting check did not find a supported correction."))

    final_boxes, detection = detector.detect(work, board_mask_fn=None, unclip=steps.unclip)
    original_lines = _ordered_lines(initial_boxes)
    final_lines = _ordered_lines(final_boxes)
    if len(final_lines) < max(1, (len(original_lines) + 1) // 2):
        work = img_bgr
        x0, y0, x1, y1 = 0, 0, img_bgr.shape[1], img_bgr.shape[0]
        final_boxes = initial_boxes
        final_lines = original_lines
        detection = {"path": "initial detection; corrections reverted"}
        for decision in decisions:
            if decision["applied"]:
                decision.update(applied=False, reason="Reverted because too much detected writing was lost.")
    selected = [item["name"] for item in decisions if item["applied"]]
    reason = ("Applied " + ", ".join(selected) + "." if selected else
              "No correction was needed; original writing retained.")
    log = dict(mode="auto", applied=bool(selected), reason=reason, decisions=decisions,
               region=[int(v) for v in (x0, y0, x1, y1)],
               boxes=len(final_boxes), lines=len(final_lines), detect=detection["path"],
               seconds=round(time.time() - t0, 3))
    return dict(image=work, read_image=work, lines=final_lines, log=log)


def line_crops(result, prepare=True, method="bicubic+sharpen"):
    """Crops of every line of a preprocess_photo result, plus box heights/widths.
    prepare=True applies the app rule (lines < REC_MIN_H px enlarged to TARGET_H) = what the reader sees."""
    out = []
    method = result.get("enlarge_method", method)
    for p in result["lines"]:
        c = tl.quad_crop(result["read_image"], p)
        h, w = tl.line_height(p), tl.line_width(p)
        out.append(dict(crop=tl.prepare_for_reader(c, h, method) if prepare else c, height=h, width=w, box=p))
    return out

def preprocess_photo(img_bgr, detector, steps, f35=None, reader=None):
    return preprocess_isolated(img_bgr, detector, steps, f35)
