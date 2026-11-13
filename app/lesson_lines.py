"""Assemble reviewed OCR boxes into readable, evidence-linked lesson lines.

The review screen keeps every original box. This module only prepares a
separate, ordered view for the slide planner; it never guesses missing words.
"""
from __future__ import annotations

import re
from statistics import median


def _box(item: dict) -> tuple[float, float, float, float] | None:
    coords = item.get("box_xyxy")
    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
        return None
    try:
        x0, y0, x1, y1 = map(float, coords)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _same_row(first: dict, second: dict) -> bool:
    a, b = first["box"], second["box"]
    ah, bh = a[3] - a[1], b[3] - b[1]
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    centres = abs((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2)
    return overlap >= 0.45 * min(ah, bh) and centres <= 0.45 * max(ah, bh)


def _line(row: list[dict], photo_id: str) -> dict:
    ordered = sorted(row, key=lambda item: (item["box"][0], item["box"][1]))
    text = " ".join(item["region"]["text"].strip() for item in ordered)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    boxes = [item["box"] for item in ordered]
    return {"id": ordered[0]["region"]["id"], "text": text,
            "photo_id": photo_id,
            "source_region_ids": [item["region"]["id"] for item in ordered],
            "box": (min(b[0] for b in boxes), min(b[1] for b in boxes),
                    max(b[2] for b in boxes), max(b[3] for b in boxes))}


def _text_continues(text: str, next_text: str) -> bool:
    text, next_text = text.strip(), next_text.strip()
    if not text or not next_text or re.search(r"[.!?;]$|=", text):
        return False
    if re.match(r"^[\s•*\-\d]", next_text) or "=" in next_text:
        return False
    return (bool(re.search(r"\b(?:a|an|and|are|as|at|by|for|from|in|is|of|on|or|the|to|with)$|:$",
                           text, re.IGNORECASE)) or next_text[0].islower())


def _continues(first: dict, second: dict) -> bool:
    if not _text_continues(first["text"], second["text"]):
        return False
    a, b = first["box"], second["box"]
    height = median((a[3] - a[1], b[3] - b[1]))
    gap = b[1] - a[3]
    if gap > height or gap < -0.25 * height or abs(b[0] - a[0]) > 1.5 * height:
        return False
    return True


def assemble_ocr(ocr: dict, evidence: list[dict], photos: list[dict]) -> dict:
    """Return planner regions in photo and geometric reading order.

    Each assembled region keeps a real OCR region ID as its anchor. The source
    IDs remain available for the UI and audit, while chart/formula checks use
    the joined text that the planner actually received.
    """
    if not photos:
        return {"regions": [dict(region) for region in ocr["regions"]]}
    by_id = {item["id"]: item for item in evidence}
    result = []
    for photo in photos:
        photo_id = photo["id"]
        items, unboxed = [], []
        for region in ocr["regions"]:
            if region.get("photo_id") != photo_id:
                continue
            box = _box(by_id.get(region["id"], {}))
            if box is None:
                unboxed.append({"id": region["id"], "text": region["text"],
                                "photo_id": photo_id, "source_region_ids": [region["id"]]})
            else:
                items.append({"region": region, "box": box})
        items.sort(key=lambda item: ((item["box"][1] + item["box"][3]) / 2,
                                     item["box"][0]))
        rows: list[list[dict]] = []
        for item in items:
            matching = next((row for row in rows if _same_row(row[0], item)), None)
            if matching is None:
                rows.append([item])
            else:
                matching.append(item)
        lines = [_line(row, photo_id) for row in rows]
        lines.sort(key=lambda line: (line["box"][1], line["box"][0]))
        joined = []
        for line in lines:
            if joined and _continues(joined[-1], line) and len(joined[-1]["text"] + line["text"]) < 350:
                previous = joined[-1]
                previous["text"] += " " + line["text"]
                previous["source_region_ids"].extend(line["source_region_ids"])
                previous["box"] = (min(previous["box"][0], line["box"][0]),
                                   previous["box"][1], max(previous["box"][2], line["box"][2]),
                                   line["box"][3])
            else:
                joined.append(line)
        for line in unboxed:
            if joined and _text_continues(joined[-1]["text"], line["text"]) and len(joined[-1]["text"] + line["text"]) < 350:
                joined[-1]["text"] += " " + line["text"]
                joined[-1]["source_region_ids"].extend(line["source_region_ids"])
            else:
                joined.append(line)
        for line in joined:
            line.pop("box", None)
        result.extend(joined)
    return {"regions": result}
