#!/usr/bin/env python3
"""Read one whiteboard photo with the packaged pipeline (the same path as the app's Extract notes button).

    python tools/ocr_image.py PHOTO [--text notes.txt] [--json result.json]

Exit codes: 0 read (possibly with no writing found), 2 the file could not be opened as an image, 1 other failure.
"""
from __future__ import annotations

import argparse, json, sys, tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / 'b2n_ocr'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('photo', type=Path); ap.add_argument('--text', type=Path); ap.add_argument('--json', type=Path)
    a = ap.parse_args()
    from student_flow import decode_photo
    try:
        image = decode_photo(a.photo.read_bytes())
    except FileNotFoundError:
        print(f'No such file: {a.photo}', file=sys.stderr); sys.exit(2)
    except ValueError as exc:
        print(f'Could not read {a.photo.name}: {exc}', file=sys.stderr); sys.exit(2)
    from ocr_backend import OCRService
    service = OCRService('S1')
    with tempfile.TemporaryDirectory(prefix='board2notes-ocr-') as d:
        path = Path(d) / 'whiteboard.png'; image.save(path, format='PNG')
        result = service.recognise(str(path))
    if not result['lines']:
        print('No writing was found. Check the photo is in focus and the board fills most of the frame, or type the notes by hand.', file=sys.stderr)
    else:
        print(result['text'])
        print(f"\n{len(result['lines'])} regions | {result['seconds']:.2f} s processing (+{result['detector_cold_load_seconds']:.2f} s detector load)", file=sys.stderr)
    if a.text:
        a.text.write_text(result['text'] + ('\n' if result['text'] else ''), encoding='utf-8')
    if a.json:
        a.json.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
