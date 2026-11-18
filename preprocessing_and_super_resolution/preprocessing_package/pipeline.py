"""The preprocessing chain for one photo, in the order the tests support (see notebook 07, section 2).

    photo --1 lighting(light)--> --2 detect (quick)--> --3 perspective--> --4 crop to the writing-->
          --5 rotation (residual, gated)--> --6 lighting(strong)--> --7 detect lines--> [8 blemish, optional]
          --> line crops --> enlarge small lines (app rule) --> reader / training crops

Why this order ([tested] = checked on our photos while building this; notebook 07 section 2 measures the
whole chain against the hurdle metrics):
  1 before 2:  lighting(light) only evens out the illumination; it helps the quick detection find faint writing
               and never binarises. 'strong' is NOT used on the whole photo: it is tuned for the board surface,
               and the walls/floor/ceiling would dominate its statistics.
  3 needs 2:   the perspective correction needs text boxes (where the writing is) and the board edges around
               them, so one quick detection must come first.
  5 after 3+4: estimate small roll from independent detected text rows and their ink pixels, not room edges.
               Require at least three separated rows, >=70% agreement within 2 degrees and box/pixel agreement.
               Short, cursive or conflicting evidence is skipped. This is a conservative heuristic, not proof
               of camera roll; consistent intentionally sloped writing can still look like a tilted camera.
  6 after 4:   'strong' lighting on the board region only: the statistics are then the board's.
  7 after 6:   the final boxes come from the image the crops are cut from, so boxes and pixels agree.
  8 last, opt: [tested] blemish removal binarises; it is applied to the region the crops are cut from (detection stays
               on the colour image), and whether it helps is measured (variant V4), not assumed. On faint or
               small writing it breaks strokes (seen on quiz_12 and cnn_01).

VARIANTS gives the presets the notebook compares (V0 = no preprocessing = what the app does today).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict, replace

import cv2
import numpy as np

from . import board_geometry as geo
from . import text_lines as tl
from . import fix_lighting, fix_rotation, blemish_removal
from . import rotation_consensus
from . import line_grouping

MAX_ROLL = 15.0          # degrees: larger "residual rotations" of the text region are not believed
REGION_MARGIN = 0.6      # text region = bounding box of the writing grown by this fraction (min 150 px)
DIRECTIONAL_GROUPING = False  # opt-in development candidate; B3 guard did not pass


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


VARIANTS = {
    "V0 raw": Steps(crop_region=False),
    "V1 perspective": Steps(perspective=True),
    "V2 light+persp+strong": Steps(lighting_light=True, perspective=True, lighting_strong=True),
    "V2b persp+strong": Steps(perspective=True, lighting_strong=True),
    "V2u V2+unclip1.5": Steps(lighting_light=True, perspective=True, lighting_strong=True, unclip=1.5),
    "V3 V2+rotation": Steps(lighting_light=True, perspective=True, rotation=True, lighting_strong=True),
    "V4 V3+blemish": Steps(lighting_light=True, perspective=True, rotation=True, lighting_strong=True, blemish=True),
}

# Round 2 (22 Oct, after round 1): the detector setting unclip 1.5 lowered CER by 0.27, far more than any step,
# and every step had been measured with line-merging boxes. Round 2 re-measures the steps on top of unclip 1.5.
# Blemish removal (+0.12 CER, CI above 0) and rotation (+0.04) are not re-tested. Listed from simplest to most
# complex: the decision rule picks the simplest chain within 0.01 CER of the best (see preprocess_order_check.py).
VARIANTS_R2 = {
    "R0 detector u1.5": Steps(crop_region=False, unclip=1.5),
    "R0 detector u1.2": Steps(crop_region=False, unclip=1.2),
    "R0 detector u1.8": Steps(crop_region=False, unclip=1.8),
    "R1g persp>=9deg u1.5": Steps(perspective=True, unclip=1.5, min_board_angle=9.0),
    "R1 persp u1.5": Steps(perspective=True, unclip=1.5),
    "R2bg persp>=9+strong u1.5": Steps(perspective=True, lighting_strong=True, unclip=1.5, min_board_angle=9.0),
    "R2b persp+strong u1.5": Steps(perspective=True, lighting_strong=True, unclip=1.5),
    "R2 light+persp+strong u1.5": Steps(lighting_light=True, perspective=True, lighting_strong=True, unclip=1.5),
}
VARIANTS.update(VARIANTS_R2)

# Round 3 (29 Oct): the pipeline decides per photo (Part 1) and keeps a correction only if the reader is surer
# of the corrected photo (Part 2). Compared with today's chain (detector only, unclip 1.2); listed simplest first.
VARIANTS_R3 = {
    "R0 detector u1.2": VARIANTS_R2["R0 detector u1.2"],
    "A0 checks only u1.2": Steps(auto=True, safety_net=False, crop_region=False, unclip=1.2),
    "A1 checks+safety net u1.2": Steps(auto=True, safety_net=True, crop_region=False, unclip=1.2),
}
VARIANTS.update(VARIANTS_R3)

# Independent ablations. All share a writing-region crop policy; no confidence
# selection, learned SR, blemish removal, sharpening or extra transformations.
VARIANTS_R4 = {
    f"S{i} {mode} only" if i else "S0 baseline": Steps(
        crop_region=False, unclip=1.2, safety_net=False, isolated=mode)
    for i, mode in enumerate(("baseline", "perspective", "lighting", "rotation", "enlargement"))
}
VARIANTS.update(VARIANTS_R4)

# ---- Part 1: the per-photo checks (each measured on the photo itself; no labels, no OCR text)
PERSP_MIN_ANGLE = 9.0       # board turned at least this much (deg): the only per-photo rule that helped in round 1
PERSP_MIN_TEXT_SCALE = 0.9  # ... and the correction keeps the writing at least this size (far shots were shrunk)
LIGHT_ISSUES = ("dim", "low_contrast")   # Jian Yan's analyze() checks on the writing area that run lighting ...
LIGHT_MIN_ILLUM = 0.45      # ... or uneven light at least this strong (analyze()'s illum_ratio). Jian Yan's own
                            # "uneven" (> 0.13) is set for documents: it fired on 211 of our 246 photos; 0.45 = the
                            # most unevenly lit ~30 % (measured 29 Oct, no labels used)
RESIDUE_MIN = 0.07          # blemish: share of the board around the writing (outside the text boxes) that looks
                            # like ink; 0.07 = the ~25 % dirtiest photos (29 Oct, no labels used)
ROT_MIN_ANGLE = 2.0         # rotation: only if the writing is still tilted at least this much (the line crops are
                            # cut along their own boxes, so a small tilt costs nothing)
# ---- Part 2: the safety net (reads EVERY line of both versions; the original's reading is the final one anyway)
SAFETY_MIN_TEXT = 0.9       # the corrected version must keep at least this share of the characters read in the
                            # original (a correction that loses lines is refused however sure the reader is)
SAFETY_MARGIN = 0.02        # ... and the reader must be surer of it by at least this much (character-weighted mean
                            # confidence over all lines; reader noise)


def _sev(rep):
    # lighting severity before -> after (fix_lighting report)
    try:
        return [round(float(rep["before"]["severity"]), 3), round(float(rep["after"]["severity"]), 3)]
    except (KeyError, TypeError):
        return None


def _gated_rotation(img, boxes=None):
    # No global projection fallback: board rulings/room edges cannot vote.
    # "confidence" in this result is a row-agreement fraction, not OCR certainty.
    return rotation_consensus.correct(img, boxes, min_angle=ROT_MIN_ANGLE, max_angle=MAX_ROLL)


def residue_outside_text(region_bgr, boxes, line_h):
    """Part 1 check for blemish removal: the share of the board around the writing that looks like ink, with the
    text boxes (grown by half a line height) masked out. Writing does not count, so a dense but clean board scores
    low (the earlier 'ink share' counted the writing itself and could not tell dirt from notes)."""
    h, w = region_bgr.shape[:2]
    s = min(1.0, 1000 / max(h, w))
    g = cv2.cvtColor(cv2.resize(region_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), max(3.0, g.shape[0] / 15))
    ink = (g / np.maximum(bg, 1)) < 0.8
    text = np.zeros(g.shape, np.uint8)
    for b in boxes:
        cv2.fillPoly(text, [np.int32(np.asarray(b) * s)], 255)
    grow = max(1, int(0.5 * line_h * s))
    text = cv2.dilate(text, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    board = text == 0
    return float(ink[board].mean()) if board.any() else 0.0


def _ordered_lines(boxes):
    if DIRECTIONAL_GROUPING:
        lines, _ = line_grouping.group_into_lines(boxes, gap_factor=getattr(tl, 'GROUP_GAP', 1.2))
    else:
        lines, _ = tl.group_into_lines(boxes)
    return [lines[i] for i in tl.reading_order(lines)]


def _read_all(img, lines, reader):
    """Part 2: read every line (prepared exactly as for the final reading). Returns (characters read, confidence),
    confidence = mean over lines weighted by their length in characters. Counting characters, not lines, is not
    fooled by the detector splitting or joining lines between the two versions; reading all lines, not the largest
    few, means small lines count too."""
    if not lines:
        return 0, 0.0
    crops = [tl.prepare_for_reader(tl.quad_crop(img, p), tl.line_height(p)) for p in lines]
    out = reader.read(crops)
    n = [len(t.strip()) for t, _ in out]
    chars = int(sum(n))
    conf = float(sum(c * k for (_, c), k in zip(out, n)) / chars) if chars else 0.0
    return chars, conf


def preprocess_auto(img_bgr, detector, steps, reader, f35=None):
    """The pipeline decides per photo.
    Part 1: measure the photo and decide which steps it needs (perspective, rotation, lighting, blemish removal).
    Part 2: if any step was applied, read the 5 largest lines of the original and of the corrected version and keep
            the corrected one only if the reader is surer of it (by SAFETY_MARGIN).
    The original path is exactly today's chain (detector only), so a photo that needs nothing is unchanged."""
    t0 = time.time()
    log = dict(steps=steps.name())
    boxes, dinfo = detector.detect(img_bgr, board_mask_fn=geo.segment_board_floodfill, unclip=steps.unclip)
    base = dict(image=img_bgr, read_image=img_bgr, lines=_ordered_lines(boxes), log=log)
    log.update(detect=dinfo["path"], boxes=len(boxes), lines=len(base["lines"]), rectified=False)
    if not boxes:
        log.update(auto=dict(decision="no text found"), seconds=round(time.time() - t0, 1))
        return base

    # ---------------- Part 1: what does this photo need?
    sig, plan = {}, {}
    lh = float(np.median([tl.line_height(b) for b in boxes]))
    r = geo.auto_rectify(img_bgr, boxes, f35=f35, proposal_search=True)
    sig["board_angle"] = None if not np.isfinite(r["angle"]) else round(float(r["angle"]), 1)
    sig["perspective_status"] = r["status"]
    if r["rectified"] is not None:
        scale = np.median([tl.line_height(cv2.perspectiveTransform(np.float32(b)[None], r["H"])[0]) /
                           max(tl.line_height(b), 1) for b in boxes])
        sig["text_scale"] = round(float(scale), 2)
        sig["readability_gate"] = r.get("readability_gate")
        sig["excluded_boxes"] = r.get("excluded_indices", [])
        preserved = r.get("readability_gate", {}).get("approved", scale >= PERSP_MIN_TEXT_SCALE)
        plan["perspective"] = (sig["board_angle"] or 0) >= PERSP_MIN_ANGLE and preserved
    else:
        plan["perspective"] = False
    pts = np.vstack(boxes)
    x0, y0, x1, y1 = geo.crop_box(img_bgr.shape, pts, REGION_MARGIN)
    region = img_bgr[y0:y1, x0:x1]
    rep_ = fix_lighting.analyze(region)
    sig["lighting_issues"] = [i for i in rep_["issues"] if i in LIGHT_ISSUES]
    sig["illum_ratio"] = rep_["metrics"]["illum_ratio"]
    plan["lighting"] = bool(sig["lighting_issues"]) or sig["illum_ratio"] >= LIGHT_MIN_ILLUM
    sig["residue"] = round(residue_outside_text(region, [np.asarray(b) - [x0, y0] for b in boxes], lh), 4)
    sig["blemish_check"] = sig["residue"] >= RESIDUE_MIN          # logged either way (to calibrate it)
    plan["blemish"] = bool(sig["blemish_check"] and steps.auto_blemish)

    # ---------------- build the corrected version with the planned steps (in the tested order)
    x, text_pts = img_bgr, pts
    if plan["perspective"]:
        x, text_pts = r["rectified"], r["text_rect"]
    cx0, cy0, cx1, cy1 = geo.crop_box(x.shape, text_pts, REGION_MARGIN)
    cand = x[cy0:cy1, cx0:cx1]
    rotation_source = [boxes[i] for i in r.get("target_indices",range(len(boxes)))] if plan["perspective"] else boxes
    rotation_boxes = [cv2.perspectiveTransform(np.float32(b)[None], r["H"])[0]
                      if plan["perspective"] else np.float32(b) for b in rotation_source]
    rotation_boxes = [b - [cx0, cy0] for b in rotation_boxes]
    rot, rinfo = _gated_rotation(cand, rotation_boxes)
    sig["rotation"] = rinfo
    plan["rotation"] = bool(rinfo["used"]) and abs(rinfo["angle"]) >= ROT_MIN_ANGLE
    if not any(plan.values()):
        log.update(auto=dict(signals=sig, plan=plan, decision="nothing needed: original kept"),
                   seconds=round(time.time() - t0, 1))
        return base
    if plan["rotation"]:
        cand = rot
    if plan["lighting"]:
        cand, _ = fix_lighting.fix(cand, strength=rep_["recommend"]["strength"])
    cboxes, cinfo = detector.detect(cand, board_mask_fn=None, unclip=steps.unclip)
    clines = _ordered_lines(cboxes)
    cread = cand
    if plan["blemish"] and clines:
        cread, _ = blemish_removal.clean_text_region(cand, float(np.median([tl.line_height(p) for p in clines])))
    corrected = dict(image=cand, read_image=cread, lines=clines, log=log)

    # ---------------- Part 2: keep the corrected version only if the reader is surer of it
    if steps.safety_net:
        if reader is None:
            raise ValueError("the safety net needs the reader: preprocess_photo(..., reader=reader)")
        n_orig, c_orig = _read_all(img_bgr, base["lines"], reader)
        n_corr, c_corr = _read_all(cread, clines, reader)
        text_kept = n_corr >= SAFETY_MIN_TEXT * n_orig
        keep = text_kept and c_corr >= c_orig + SAFETY_MARGIN
        if keep:
            decision = f"corrected kept (confidence {c_orig:.2f} -> {c_corr:.2f}, characters {n_orig} -> {n_corr})"
        elif not text_kept:
            decision = f"original kept (corrected lost text: characters {n_orig} -> {n_corr})"
        else:
            decision = f"original kept (confidence {c_orig:.2f} vs corrected {c_corr:.2f})"
        sig.update(conf_original=round(c_orig, 3), conf_corrected=round(c_corr, 3),
                   chars_original=n_orig, chars_corrected=n_corr)
    else:
        keep, decision = bool(clines), "corrected used (no safety net)"
    log.update(auto=dict(signals=sig, plan=plan, decision=decision),
               seconds=round(time.time() - t0, 1))
    if keep:
        log.update(rectified=bool(plan["perspective"]), detect=cinfo["path"], boxes=len(cboxes), lines=len(clines))
        return corrected
    return base


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


def preprocess_photo(img_bgr, detector, steps: Steps, f35=None, reader=None):
    """Run the chain on one upright photo. Returns dict:
        image       : the image the lines were detected on (colour)
        read_image  : the image the crops are cut from (= image, or its blemish-cleaned version)
        lines       : text-line boxes (4x2, in `image` coords) in reading order, boxes on one line joined
        log         : what each step did (status strings, angles, seconds)
    """
    if steps.isolated:
        return preprocess_isolated(img_bgr, detector, steps, f35)
    if steps.auto:
        return preprocess_auto(img_bgr, detector, steps, reader, f35)
    t0 = time.time()
    log = dict(steps=steps.name())
    x = img_bgr
    if steps.lighting_light:
        x, rep = fix_lighting.fix(x, strength="light")
        log["lighting_light_severity"] = _sev(rep)

    rectified = False
    if steps.perspective or steps.crop_region:
        boxes, _ = detector.detect(x, board_mask_fn=geo.segment_board_floodfill, unclip=steps.unclip)
        log["quick_boxes"] = len(boxes)
        text_pts = np.vstack(boxes) if boxes else None
        if steps.perspective and boxes:
            r = geo.auto_rectify(x, boxes, f35=f35, proposal_search=True)
            log["perspective"] = r["status"]
            log["board_angle"] = None if not np.isfinite(r["angle"]) else round(float(r["angle"]), 1)
            if r["rectified"] is not None and np.isfinite(r["angle"]) and r["angle"] < steps.min_board_angle:
                log["perspective"] = f"board angle {r['angle']:.1f} < {steps.min_board_angle:g} deg: not applied"
            elif r["rectified"] is not None:
                x, text_pts, rectified = r["rectified"], r["text_rect"], True
        if steps.crop_region and text_pts is not None:
            x0, y0, x1, y1 = geo.crop_box(x.shape, text_pts, REGION_MARGIN)
            x = x[y0:y1, x0:x1]
            log["region"] = [int(x0), int(y0), int(x1), int(y1)]
    log["rectified"] = rectified

    if steps.rotation:
        # Redetect in the current coordinates, after any perspective warp/ROI.
        rotation_boxes, _ = detector.detect(x, board_mask_fn=None, unclip=steps.unclip)
        x, rinfo = _gated_rotation(x, rotation_boxes)
        log["rotation"] = rinfo
    if steps.lighting_strong:
        x, rep = fix_lighting.fix(x, strength="strong")
        log["lighting_strong_severity"] = _sev(rep)

    boxes, dinfo = detector.detect(x, board_mask_fn=None if steps.crop_region else geo.segment_board_floodfill,
                                   unclip=steps.unclip)
    log["detect"] = dinfo["path"]
    lines, parts = tl.group_into_lines(boxes)
    order = tl.reading_order(lines)
    lines = [lines[i] for i in order]
    log["boxes"], log["lines"] = len(boxes), len(lines)

    read_image = x
    if steps.blemish and lines:
        lh = float(np.median([tl.line_height(p) for p in lines]))
        read_image, binfo = blemish_removal.clean_text_region(x, lh)
        log["blemish"] = binfo
    log["seconds"] = round(time.time() - t0, 1)
    return dict(image=x, read_image=read_image, lines=lines, log=log)


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
