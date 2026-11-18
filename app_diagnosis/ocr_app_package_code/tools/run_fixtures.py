#!/usr/bin/env python3
"""Run one OCR code base (the packaged copy or the live app) on the fixed fixtures, exactly along the app's path:
decode_photo (EXIF) -> lossless PNG -> OCRService('S1').recognise. Writes one JSON file. Used by verify_package.py.

    python tools/run_fixtures.py --code-dir b2n_ocr --out outputs/fixtures_packaged.json
"""
from __future__ import annotations

import argparse, hashlib, io, json, os, platform, sys, tempfile, time, traceback
from importlib.metadata import version
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def derived_fixtures(src: Path, work: Path):
    """Deterministic format/edge-case fixtures made from jy_photo23.jpg (not stored in the package)."""
    from PIL import Image
    out = {}
    base = Image.open(src).convert('RGB')
    p = work / 'jy_photo23.png'; base.save(p, format='PNG'); out['format_png'] = p
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        p = work / 'jy_photo23.heic'; base.save(p, format='HEIF', quality=95); out['format_heic'] = p
    except Exception as exc:  # recorded, not hidden
        out['format_heic'] = f'SKIPPED: HEIC encoder unavailable ({exc})'
    rot = base.transpose(Image.Transpose.ROTATE_90)            # pixels stored sideways ...
    exif = Image.Exif(); exif[0x0112] = 6                       # ... and EXIF says "rotate 90 CW to display"
    p = work / 'jy_photo23_exif6.jpg'; rot.save(p, format='JPEG', quality=95, exif=exif.tobytes()); out['exif_orientation_6_jpeg'] = p
    p = work / 'not_an_image.jpg'; p.write_bytes(b'this is not an image\n' * 20); out['unreadable_bytes'] = p
    p = work / 'truncated.jpg'; p.write_bytes(src.read_bytes()[:2048]); out['truncated_jpeg'] = p
    p = work / 'blank_board.png'; Image.new('RGB', (1200, 900), (236, 238, 240)).save(p); out['blank_board'] = p
    p = work / 'tiny.png'; Image.new('RGB', (12, 12), 'white').save(p); out['tiny_12px'] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code-dir', required=True, type=Path, help='folder holding ocr_backend.py and student_flow.py')
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    code = a.code_dir.resolve(); sys.path.insert(0, str(code))
    t0 = time.perf_counter()
    import numpy as np
    from student_flow import decode_photo
    import ocr_backend as ob
    t = time.perf_counter(); svc = ob.OCRService('S1'); load_reader = time.perf_counter() - t
    manifest = json.loads((PKG / 'MANIFEST.json').read_text())
    photos = [(f['id'], PKG / 'fixtures/photos' / f['id']) for f in manifest['fixtures']]
    work = Path(tempfile.mkdtemp(prefix='b2n-fixtures-'))
    cases = photos + list(derived_fixtures(PKG / 'fixtures/photos/jy_photo23.jpg', work).items())
    results = {}
    for fid, path in cases:
        rec = dict(fixture=fid)
        if isinstance(path, str):
            rec['status'] = path; results[fid] = rec; continue
        rec['file_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        try:
            img = decode_photo(Path(path).read_bytes())                       # the app's upload decoder (raises ValueError)
            rec['decoded_size'] = list(img.size)
            with tempfile.TemporaryDirectory() as d:                           # the app's run_ocr(): lossless PNG of the exact pixels
                p = Path(d) / 'whiteboard.png'; img.save(p, format='PNG')
                t = time.perf_counter(); r = svc.recognise(str(p)); rec['wall_seconds'] = time.perf_counter() - t
            rec.update(status='ok', n_lines=len(r['lines']), text=r['text'],
                       lines=[dict(index=l['index'], text=l['text'], score=l['score'], box=l['box']) for l in r['lines']],
                       seconds=r['seconds'], detector_cold_load_seconds=r['detector_cold_load_seconds'],
                       detector=r['detector'], image_wh=[r['image_width'], r['image_height']])
            # export checks: text and JSON as the app/CLI write them
            txt = r['text'].encode('utf-8'); js = json.dumps(r, ensure_ascii=False, indent=1).encode('utf-8')
            rec['export'] = dict(text_bytes=len(txt), text_roundtrip=txt.decode('utf-8') == r['text'],
                                 json_bytes=len(js), json_roundtrip=json.loads(js)['text'] == r['text'])
        except ValueError as exc:
            rec.update(status='rejected', error_type=type(exc).__name__, message=str(exc))
        except Exception as exc:
            rec.update(status='error', error_type=type(exc).__name__, message=str(exc)[:500], trace=traceback.format_exc()[-1500:])
        results[fid] = rec
        print(f"{fid:26s} {rec['status']:9s} {rec.get('n_lines', '')}", flush=True)
    env = dict(python=sys.version.split()[0], platform=platform.platform(), machine=platform.machine())
    for pkg in ('paddlepaddle', 'paddleocr', 'paddlex', 'numpy', 'opencv-python', 'opencv-python-headless', 'opencv-contrib-python', 'pillow', 'pillow-heif'):
        try: env[pkg] = version(pkg)
        except Exception: env[pkg] = None
    import cv2
    env['cv2_in_use'] = cv2.__version__
    out = dict(code_dir=str(code), backend_sha256=hashlib.sha256((code / 'ocr_backend.py').read_bytes()).hexdigest(),
               reader_identity=svc.reader.identity(), reader_load_seconds=load_reader, total_seconds=time.perf_counter() - t0,
               env=env, results=results)
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
