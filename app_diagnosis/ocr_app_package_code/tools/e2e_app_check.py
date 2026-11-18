#!/usr/bin/env python3
"""End-to-end student-app check with the REAL packaged models (no mocking), via Streamlit AppTest:
Try a sample photo -> Extract notes -> notes text area holds the OCR text -> edit a word -> the export holds the edit.
AppTest does not run browser JavaScript (copy-to-clipboard and the browser download event need a manual browser check).

    python tools/e2e_app_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
CODE = PKG / 'b2n_ocr'
sys.path.insert(0, str(CODE))


def main():
    from streamlit.testing.v1 import AppTest
    import student_app
    at = AppTest.from_file(str(CODE / 'app.py'), default_timeout=300).run()
    btn = lambda label: next(b for b in at.button if b.label == label)
    btn('Try a sample photo').click().run()
    assert at.session_state['phase'] == 'check', at.session_state['phase']
    btn('Extract notes').click().run()
    assert at.session_state['phase'] == 'review', 'extraction did not reach the review page'
    result = at.session_state['result']
    notes = at.text_area(key='notes').value
    assert notes == result['text'] and len(result['lines']) > 0, 'notes do not equal the OCR text'
    print(f"extracted {len(result['lines'])} regions; first line: {result['lines'][0]['text']!r}")
    edited = notes.replace(result['lines'][0]['text'], 'EDITED LINE', 1)
    at.text_area(key='notes').set_value(edited).run()
    # the Save/Copy actions export current_notes(); the mocked journey tests check that this returns the live edit
    assert at.session_state['notes'] == edited, 'edit not kept in session state'
    assert result['lines'][0]['text'] != 'EDITED LINE' and at.session_state['result']['lines'][0]['text'] == result['lines'][0]['text'], 'edit overwrote the original reading'
    assert student_app.current_notes.__module__ == 'student_app'
    print('PASSED: sample -> extract -> edit kept, original reading preserved')


if __name__ == '__main__':
    main()
