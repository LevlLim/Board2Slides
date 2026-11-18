"""Student journey checks with mocked OCR; no model inference.

From Assignment:
  .venv-paddle/bin/python -m unittest discover \
    -s whiteboard-ocr/ocr_prototype/tests -p 'test_student_app.py'

AppTest exercises Streamlit reruns/session state. It does not execute browser
JavaScript, so clipboard/download button behaviour still needs a browser check.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import student_app


class StudentAppTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="student-ui-test-")
        self.addCleanup(self.directory.cleanup)
        self.sample = Path(self.directory.name) / "sample.png"
        image = Image.new("RGB", (120, 80), "white")
        image.paste("red", (0, 0, 60, 40))
        image.save(self.sample)
        self.addCleanup(patch.stopall)
        patch.object(student_app, "SAMPLE_PHOTO", self.sample).start()
        self.reader = patch.object(student_app, "run_ocr", side_effect=self.fake_ocr).start()
        self.inputs = []
        self.app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=15)
        self.assert_no_errors()

    def fake_ocr(self, image):
        self.inputs.append(image.copy())
        return dict(text="Draft line one\nDraft line two", seconds=.01,
                    image_width=image.width, image_height=image.height,
                    lines=[dict(text="Draft line one", score=.95,
                                box=[[2, 2], [min(60, image.width - 2), 2],
                                     [min(60, image.width - 2), min(30, image.height - 2)],
                                     [2, min(30, image.height - 2)]])])

    def assert_no_errors(self):
        self.assertEqual(len(self.app.exception), 0, [str(e) for e in self.app.exception])

    def button(self, label):
        return next(button for button in self.app.button if button.label == label)

    def click(self, label):
        self.button(label).click().run(timeout=15)
        self.assert_no_errors()

    def add_sample(self):
        self.click("Try a sample photo")
        self.assertEqual(self.app.session_state["phase"], "check")

    def extract(self):
        self.add_sample()
        self.click("Extract notes")
        self.assertEqual(self.app.session_state["phase"], "review")

    def edit_notes(self, text="My corrected notes"):
        self.app.text_area[0].set_value(text).run(timeout=15)
        self.assert_no_errors()
        self.assertEqual(self.app.session_state["notes"], text)

    def test_add_sample_enters_check_without_loading_model(self):
        self.assertEqual(self.app.session_state["phase"], "add")
        self.assertEqual(len(self.app.selectbox), 0)  # No technical model selection.
        self.assertIn("Your whiteboard, ready to study.", [x.value for x in self.app.title])
        self.add_sample()
        self.assertEqual(self.app.session_state["photo"].size, (120, 80))
        self.assertFalse(self.button("Extract notes").disabled)
        self.reader.assert_not_called()

    def test_rotation_and_crop_are_the_pixels_sent_to_ocr(self):
        self.add_sample()
        self.click("↷ Rotate right")
        self.assertEqual(self.app.session_state["turns"], 1)
        self.app.toggle[0].set_value(True).run(timeout=15)
        self.app.slider(key="horizontal_area").set_value((25, 75)).run(timeout=15)
        self.app.slider(key="vertical_area").set_value((0, 50)).run(timeout=15)
        self.click("Extract notes")
        self.assertEqual(len(self.inputs), 1)
        # Clockwise rotation gives 80x120, then keep central half and top half.
        self.assertEqual(self.inputs[0].size, (40, 60))
        self.assertEqual(self.app.session_state["result_photo"].size, (40, 60))
        self.assertEqual(self.app.session_state["photo"].size, (120, 80))
        self.assertEqual(self.app.text_area[0].value, "Draft line one\nDraft line two")

    def test_invalid_tiny_crop_disables_extraction(self):
        self.add_sample()
        self.app.toggle[0].set_value(True).run(timeout=15)
        self.app.slider(key="horizontal_area").set_value((49, 50)).run(timeout=15)
        self.assert_no_errors()
        self.assertTrue(self.button("Extract notes").disabled)
        self.assertTrue(any("crop is too small" in w.value for w in self.app.warning))
        self.reader.assert_not_called()

    def test_region_selection_links_original_reading_without_overwriting_notes(self):
        self.reader.side_effect = lambda image: dict(text="First\nSecond", lines=[
            dict(text="First", box=[[2, 2], [65, 2], [65, 25], [2, 25]]),
            dict(text="Second", box=[[2, 40], [65, 40], [65, 65], [2, 65]])])
        self.extract()
        self.assertTrue(self.app.toggle(key="show_regions").value)
        self.edit_notes("My corrected notes")
        with patch.object(student_app, "region_photo", return_value={"index": 1, "event": 123}):
            self.app.run(timeout=15)
        self.assert_no_errors()
        self.assertEqual(self.app.selectbox(key="selected_line").value, 1)
        self.assertEqual(self.app.text_area[0].value, "My corrected notes")
        self.assertIn("Second", [element.value for element in self.app.text])
        self.app.selectbox(key="selected_line").set_value(0).run(timeout=15)
        self.assert_no_errors()
        self.assertEqual(self.app.text_area[0].value, "My corrected notes")
        self.assertIn("First", [element.value for element in self.app.text])
        self.assertEqual(self.reader.call_count, 1)

    def test_trim_and_crop_survive_review_navigation_and_toggle_off_on(self):
        self.add_sample()
        self.app.toggle(key="trim_edges").set_value(True).run(timeout=15)
        self.app.slider(key="horizontal_area").set_value((25, 75)).run(timeout=15)
        self.app.slider(key="vertical_area").set_value((25, 75)).run(timeout=15)
        self.click("Extract notes")
        first_pixels = self.inputs[0].tobytes()
        self.assertEqual(self.inputs[0].size, (60, 40))
        # Review no longer renders crop controls, so another rerun exercises
        # Streamlit's widget cleanup rather than only checking a recent value.
        self.app.run(timeout=15)
        self.click("Adjust photo and read again")
        self.app.run(timeout=15)
        self.assert_no_errors()
        self.assertTrue(self.app.toggle(key="trim_edges").value)
        self.assertEqual(self.app.slider(key="horizontal_area").value, (25, 75))
        self.assertEqual(self.app.slider(key="vertical_area").value, (25, 75))
        self.app.toggle(key="trim_edges").set_value(False).run(timeout=15)
        self.app.run(timeout=15)
        self.assertEqual(self.app.session_state["crop_x"], (25, 75))
        self.assertEqual(self.app.session_state["crop_y"], (25, 75))
        self.app.toggle(key="trim_edges").set_value(True).run(timeout=15)
        self.assertEqual(self.app.slider(key="horizontal_area").value, (25, 75))
        self.assertEqual(self.app.slider(key="vertical_area").value, (25, 75))
        self.click("Extract notes")
        self.assertEqual(self.inputs[1].size, (60, 40))
        self.assertEqual(self.inputs[1].tobytes(), first_pixels)
        self.assertEqual(self.app.session_state["result_photo"].size, (60, 40))

    def test_edits_survive_adjust_and_back_without_another_extraction(self):
        self.extract()
        self.edit_notes("Corrected: a line the OCR missed.")
        self.click("Adjust photo and read again")
        self.assertEqual(self.app.session_state["phase"], "check")
        self.assertEqual(self.app.session_state["draft_notes"], "Corrected: a line the OCR missed.")
        self.assertTrue(self.button("Extract notes").disabled)
        self.assertFalse(self.app.checkbox(key="replace_edits").value)
        self.click("↷ Rotate right")
        self.assertTrue(self.button("Extract notes").disabled)
        self.click("Back to my notes")
        self.assertEqual(self.app.text_area[0].value, "Corrected: a line the OCR missed.")
        # Old result retains the pixels it was generated from, not the new rotation.
        self.assertEqual(self.app.session_state["result_photo"].size, (120, 80))
        self.assertEqual(self.reader.call_count, 1)

    def test_replacing_edits_requires_explicit_checkbox(self):
        self.extract()
        self.edit_notes()
        self.click("Adjust photo and read again")
        self.assertTrue(self.button("Extract notes").disabled)
        self.app.checkbox(key="replace_edits").check().run(timeout=15)
        self.assertFalse(self.button("Extract notes").disabled)
        self.reader.side_effect = lambda image: dict(text="New extraction", lines=[], seconds=.01)
        self.click("Extract notes")
        self.assertEqual(self.app.session_state["phase"], "review")
        self.assertEqual(self.app.text_area[0].value, "New extraction")
        self.assertEqual(self.reader.call_count, 2)
        self.edit_notes("One more correction")
        self.click("Adjust photo and read again")
        self.assertFalse(self.app.checkbox(key="replace_edits").value)
        self.assertTrue(self.button("Extract notes").disabled)

    def test_start_new_opens_confirmation_without_discarding_notes(self):
        self.extract()
        self.edit_notes("Keep this draft")
        self.click("Add another photo")
        self.assertEqual(self.app.session_state["phase"], "review")
        self.assertEqual(self.app.session_state["notes"], "Keep this draft")
        self.assertIsNotNone(self.button("Start another note"))
        self.assertIsNotNone(self.button("Keep editing"))
        self.assertEqual(self.reader.call_count, 1)

    def dialog_harness(self):
        # AppTest replays full-script rather than dialog-fragment reruns. Keep
        # the dialog rendered in a small harness so its real button handlers
        # execute. The regular app test above checks entry into confirmation;
        # actual browser fragment handling must also be checked separately.
        self.app = AppTest.from_string('''
import streamlit as st
import student_app as ui
ui.initialise()
if "result" in st.session_state:
    ui.confirm_new_photo()
else:
    ui.add_photo_page()
''')
        self.app.session_state["phase"] = "review"
        self.app.session_state["photo"] = Image.open(self.sample).copy()
        self.app.session_state["photo_name"] = "sample.png"
        self.app.session_state["result"] = {"text": "Original"}
        self.app.session_state["draft_notes"] = "Keep this draft"
        self.app.session_state["notes"] = "Keep this draft"
        self.app.run(timeout=15)
        self.assert_no_errors()

    def test_confirmation_handler_resets_only_after_explicit_confirm(self):
        self.dialog_harness()
        self.click("Start another note")
        self.assertEqual(self.app.session_state["phase"], "add")
        for key in ("photo", "notes", "draft_notes", "result", "result_photo"):
            self.assertNotIn(key, self.app.session_state)
        self.assertEqual(self.app.session_state["upload_generation"], 1)

    def test_confirmation_cancel_handler_keeps_draft(self):
        self.dialog_harness()
        self.click("Keep editing")
        self.assertEqual(self.app.session_state["phase"], "review")
        self.assertEqual(self.app.session_state["draft_notes"], "Keep this draft")
        self.assertEqual(self.app.session_state["photo"].size, (120, 80))
        self.assertEqual(self.app.session_state["upload_generation"], 0)

    def test_ocr_failure_preserves_edited_draft_and_photo(self):
        self.extract()
        self.edit_notes("Corrected text must survive an OCR failure")
        self.click("Adjust photo and read again")
        self.app.checkbox(key="replace_edits").check().run(timeout=15)
        self.reader.side_effect = RuntimeError("simulated backend failure")
        self.click("Extract notes")
        self.assertEqual(self.app.session_state["phase"], "check")
        self.assertEqual(self.app.session_state["draft_notes"], "Corrected text must survive an OCR failure")
        self.assertEqual(self.app.session_state["photo"].size, (120, 80))
        self.assertTrue(any("couldn’t read" in e.value for e in self.app.error))
        self.click("Back to my notes")
        self.assertEqual(self.app.text_area[0].value, "Corrected text must survive an OCR failure")

    def test_first_ocr_failure_keeps_photo_for_retry(self):
        self.add_sample()
        self.reader.side_effect = RuntimeError("simulated backend failure")
        self.click("Extract notes")
        self.assertEqual(self.app.session_state["phase"], "check")
        self.assertEqual(self.app.session_state["photo"].size, (120, 80))
        self.assertNotIn("result", self.app.session_state)
        self.reader.side_effect = self.fake_ocr
        self.click("Extract notes")
        self.assertEqual(self.app.session_state["phase"], "review")

    def test_empty_ocr_allows_manual_notes_and_safe_return_to_photo(self):
        self.reader.side_effect = lambda image: dict(text="", lines=[], seconds=.01)
        self.extract()
        self.assertTrue(any("couldn’t find readable text" in w.value for w in self.app.warning))
        self.assertEqual(self.app.text_area[0].value, "")
        self.edit_notes("I can type the missing notes myself.")
        self.click("Adjust photo and read again")
        self.assertTrue(self.button("Extract notes").disabled)
        self.click("Back to my notes")
        self.assertEqual(self.app.text_area[0].value, "I can type the missing notes myself.")

    def test_export_fallback_encodes_notes_as_data_not_script(self):
        # The component exports the live browser textarea; this checks only its
        # server-side fallback encoding, not actual clipboard access in a browser.
        text = '</script><img src=x onerror="alert(1)"> & café\nNew line\u2028'
        with patch.object(student_app.st, "iframe") as component:
            student_app.notes_actions(text)
        source = component.call_args.args[0]
        encoded = re.search(r"const savedNotes = (.*?);\n", source).group(1)
        self.assertEqual(json.loads(encoded), text)
        self.assertNotIn('</script><img', source)
        self.assertIn('return editor.value', source)
        self.assertIn('new Blob([currentNotes()]', source)
        self.assertIn("message.textContent=", source)


if __name__ == "__main__":
    unittest.main()
