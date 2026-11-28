"""Run with: python -m unittest discover -s tests -p 'test_student_flow.py'."""

import sys
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from student_flow import decode_photo, draw_regions, line_preview, photo_token, prepare_photo


class StudentFlowTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (80, 40), "white")
        self.image.paste("red", (0, 0, 40, 20))
        self.image.paste("blue", (40, 20, 80, 40))

    def test_clockwise_rotation_and_crop_coordinates(self):
        rotated = prepare_photo(self.image, quarter_turns=1)
        self.assertEqual(rotated.size, (40, 80))
        self.assertEqual(rotated.getpixel((30, 10)), (255, 0, 0))
        self.assertEqual(rotated.getpixel((10, 60)), (0, 0, 255))
        crop = prepare_photo(self.image, 1, (50, 0, 100, 50))
        self.assertEqual(crop.size, (20, 40))
        self.assertEqual(crop.getextrema(), ((255, 255), (0, 0), (0, 0)))
        self.assertEqual(photo_token(prepare_photo(self.image, 4)), photo_token(self.image))

    def test_invalid_crop_or_rotation_is_rejected(self):
        for bounds in [(-1, 0, 100, 100), (0, 0, 101, 100), (50, 0, 50, 100),
                       (80, 0, 20, 100), (0, 0, float("nan"), 100),
                       (0, 0, 10, 100), (0, 0, 100), None]:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                prepare_photo(self.image, crop_percent=bounds)
        with self.assertRaises(ValueError):
            prepare_photo(self.image, quarter_turns=0.5)

    def test_decode_applies_exif_orientation_and_returns_rgb(self):
        source = Image.new("RGB", (40, 20), "red")
        exif = source.getexif()
        exif[274] = 6  # 90 degrees clockwise for correct display.
        data = BytesIO()
        source.save(data, format="JPEG", exif=exif)
        decoded = decode_photo(data.getvalue())
        self.assertEqual(decoded.size, (20, 40))
        self.assertEqual(decoded.mode, "RGB")
        self.assertNotIn(274, decoded.getexif())
        with self.assertRaises(ValueError):
            decode_photo(b"not an image")

    def test_heic_decode_when_dependency_available(self):
        try:
            import pillow_heif
        except ImportError:
            self.skipTest("pillow-heif is optional")
        pillow_heif.register_heif_opener()
        data = BytesIO()
        self.image.save(data, format="HEIF")
        decoded = decode_photo(data.getvalue())
        self.assertEqual(decoded.size, self.image.size)
        self.assertEqual(decoded.mode, "RGB")

    def test_token_tracks_pixels_and_dimensions_only(self):
        copy = self.image.copy()
        copy.info["comment"] = "Different metadata"
        self.assertEqual(photo_token(copy), photo_token(self.image))
        copy.putpixel((0, 0), (0, 0, 0))
        self.assertNotEqual(photo_token(copy), photo_token(self.image))
        self.assertNotEqual(photo_token(Image.new("RGB", (40, 80), "white")),
                            photo_token(Image.new("RGB", (80, 40), "white")))

    def test_preview_expands_outward_and_clamps_without_padding(self):
        box = [(1.2, 2.2), (31.2, 2.2), (31.2, 22.2), (1.2, 22.2)]
        preview = line_preview(self.image, box, margin=0.25)
        # Outward rounding: (-7, -3, 39, 28), clamped to (0, 0, 39, 28).
        self.assertEqual(preview.size, (39, 28))
        self.assertEqual(preview.tobytes(), self.image.crop((0, 0, 39, 28)).tobytes())
        for bad_box in [[], [(float("inf"), 0), (1, 0), (1, 1)],
                        [(100, 100), (110, 100), (110, 110)]]:
            with self.subTest(box=bad_box), self.assertRaises(ValueError):
                line_preview(self.image, bad_box)
        with self.assertRaises(ValueError):
            line_preview(self.image, box, margin=-0.1)

    def test_drawing_keeps_original_and_highlights_selected_position(self):
        token = photo_token(self.image)
        lines = [{"index": 10, "box": [(5, 5), (35, 5), (35, 25), (5, 25)]}]
        drawing = draw_regions(self.image, lines)
        selected = draw_regions(self.image, lines, selected_index=0)
        self.assertEqual(photo_token(self.image), token)
        self.assertNotEqual(photo_token(drawing), token)
        self.assertNotEqual(photo_token(drawing), photo_token(selected))
        self.assertEqual(draw_regions(self.image, [{"box": []}]).tobytes(), self.image.tobytes())


if __name__ == "__main__":
    unittest.main()
