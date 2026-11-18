"""Image helpers for the student photo-review flow; no OCR or model imports.

Rotations are clockwise. Percentage crops are measured on the *rotated* image.
All functions return a new image and leave the supplied image unchanged.
"""

from __future__ import annotations

import hashlib
import math
import struct
from io import BytesIO
from numbers import Integral
from typing import Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


def decode_photo(data: bytes) -> Image.Image:
    """Decode an uploaded photo, apply EXIF orientation, and return RGB pixels.

    JPEG, PNG, and other Pillow formats work without optional dependencies.
    HEIC/HEIF uploads are supported when ``pillow-heif`` is installed.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)) or not data:
        raise ValueError("Choose a non-empty image file.")
    try:
        import pillow_heif
    except ImportError:
        pass
    else:
        pillow_heif.register_heif_opener()
    try:
        with Image.open(BytesIO(data)) as source:
            return ImageOps.exif_transpose(source).convert("RGB").copy()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError(
            "This photo could not be opened. Try a JPEG, PNG, or supported HEIC file."
        ) from exc


def prepare_photo(
    image: Image.Image,
    quarter_turns: int = 0,
    crop_percent: Sequence[float] = (0, 0, 100, 100),
) -> Image.Image:
    """Rotate clockwise in 90° steps, then crop in rotated-image coordinates.

    Crop values are ``(left, top, right, bottom)`` percentages in [0, 100].
    The crop must have positive area and contain at least 16 × 16 pixels.
    Pixel boundaries round outward so fractional coordinates do not cut pixels.
    This never resizes, enhances, or otherwise changes the photo's pixels.
    """
    if not isinstance(quarter_turns, Integral):
        raise ValueError("Rotation must be a whole number of quarter turns.")
    try:
        left, top, right, bottom = (float(value) for value in crop_percent)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Crop needs four percentage values.") from exc
    values = (left, top, right, bottom)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Crop percentages must be finite numbers.")
    if not (0 <= left < right <= 100 and 0 <= top < bottom <= 100):
        raise ValueError("Crop edges must be within 0–100% and enclose a positive area.")

    rotated = image.convert("RGB")
    turns = int(quarter_turns) % 4
    if turns:
        rotated = rotated.transpose(
            {
                1: Image.Transpose.ROTATE_270,
                2: Image.Transpose.ROTATE_180,
                3: Image.Transpose.ROTATE_90,
            }[turns]
        )
    width, height = rotated.size
    bounds = (
        math.floor(width * left / 100),
        math.floor(height * top / 100),
        math.ceil(width * right / 100),
        math.ceil(height * bottom / 100),
    )
    if bounds[2] - bounds[0] < 16 or bounds[3] - bounds[1] < 16:
        raise ValueError("Keep a photo area of at least 16 × 16 pixels.")
    return rotated.crop(bounds)


def photo_token(image: Image.Image) -> str:
    """Identify the exact RGB pixels and dimensions, independent of metadata."""
    rgb = image.convert("RGB")
    digest = hashlib.sha256(struct.pack(">QQ", *rgb.size))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _points(box: Iterable[Sequence[float]]) -> list[tuple[float, float]]:
    try:
        points = [(float(x), float(y)) for x, y in box]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("A text region must contain (x, y) points.") from exc
    if len(points) < 3 or not all(
        math.isfinite(value) for point in points for value in point
    ):
        raise ValueError("A text region needs at least three finite points.")
    return points


def line_preview(
    image: Image.Image, box: Iterable[Sequence[float]], margin: float = 0.25
) -> Image.Image:
    """Crop a region with extra context, clamped safely to the photo edges.

    ``margin`` adds that fraction of the region width/height on *each* side.
    The returned crop retains source resolution; the UI may display it larger.
    """
    points = _points(box)
    try:
        margin = float(margin)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Preview margin must be a non-negative number.") from exc
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Preview margin must be a non-negative finite number.")
    xs, ys = zip(*points)
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        raise ValueError("The text region must have a positive width and height.")
    bounds = (
        max(0, math.floor(left - margin * width)),
        max(0, math.floor(top - margin * height)),
        min(image.width, math.ceil(right + margin * width)),
        min(image.height, math.ceil(bottom + margin * height)),
    )
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError("The text region does not overlap the photo.")
    return image.convert("RGB").crop(bounds)


def draw_regions(
    image: Image.Image,
    lines: Iterable[Mapping],
    selected_index: int | None = None,
) -> Image.Image:
    """Draw numbered regions; ``selected_index`` is a zero-based list position.

    Numbers are one-based reading order, matching the text-review list. Invalid
    polygons are skipped so one malformed detection cannot hide all results.
    """
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    stroke = max(2, round(min(result.size) / 350))
    font = ImageFont.load_default(size=max(14, min(26, round(min(result.size) / 35))))
    for position, line in enumerate(lines):
        try:
            points = _points(line["box"])
        except (KeyError, TypeError, ValueError):
            continue
        selected = position == selected_index
        colour = "#d97706" if selected else "#0d9488"
        draw.line(points + [points[0]], fill=colour, width=stroke + int(selected))
        label = str(position + 1)
        text_bounds = draw.textbbox((0, 0), label, font=font)
        label_width = text_bounds[2] - text_bounds[0] + 12
        label_height = text_bounds[3] - text_bounds[1] + 8
        x = max(0, min(int(min(point[0] for point in points)), result.width - label_width))
        y = max(0, min(int(min(point[1] for point in points)) - label_height, result.height - label_height))
        draw.rounded_rectangle(
            (x, y, x + label_width, y + label_height), radius=3, fill=colour
        )
        draw.text(
            (x + 6 - text_bounds[0], y + 4 - text_bounds[1]),
            label,
            font=font,
            fill="white",
        )
    return result
