# OCR app and diagnosis

| Folder | What it is |
|---|---|
| `ocr_app_package_code/` | The OCR app package (`b2n_ocr/`, deploy, tools, fixtures, verification notes). Models left out: run the original `Assignment/board2notes_ocr_package/` (`run_local.command`). |
| `ocr_prototype/` | The earlier local OCR-only prototype |
| `detector_and_crop_diagnostics/crop_margin_diagnostic_20261109/` | Does a tighter crop margin change OCR error? (notebook 14) |
| `detector_and_crop_diagnostics/jh_full_photo_app_check_20261109/` | The app on JH's full photos (notebook 16) |
| `detector_and_crop_diagnostics/weakness_diagnosis_20261112/` | What the reader still gets wrong: deletions vs substitutions, CTC column room, the app's own crops. `additions_20261113/detector_size_sweep/`: the detector size test and rule C (big writing 36.0% to 12.0%, every other set unchanged; `text_lines_with_rule_C.py` is a copy of the app file with the rule, NOT applied to the app) |
| `fresh_board_test/` | The 11 Nov fresh-board test (5 boards, 152 lines): manifest, review sheets, scores. Crop images left out: `Assignment/board2notes_freshwriter_test_20261111/images/` |
| `whole_photo_test_20261125/` | The 25 Nov whole-photo test: the app's OCR path on 44 photos of the same 5 boards, one row per reader (the poster's whole-photo table) |
| `notebooks/` | 11 Jy evaluation, 12 full-photo evaluation, 14 crop margins, 15 full-photo detector comparison, 16 JH full-photo app evaluation |
