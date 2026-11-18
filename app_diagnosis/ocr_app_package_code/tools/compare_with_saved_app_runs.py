#!/usr/bin/env python3
"""Compare a fixture run with the live app's earlier saved Mac outputs for the same four photos (no inference).
Saved runs: notebook 15 mobile-detector arm (Jy Photos 8, 23) and notebook 16 app run (JH Photos 2, 5), both
made on the Mac with the unchanged app pipeline (mobile_det -> group -> order -> quad crop -> S1).

    python tools/compare_with_saved_app_runs.py outputs/fixtures_packaged.json
"""
import json, sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]; A = PKG.parent
SAVED = {'jy_photo23.jpg': 'whiteboard-ocr/outputs/detector_comparison/PP-OCRv5_mobile_det/Photo_23.json',
         'jy_photo8.jpg': 'whiteboard-ocr/outputs/detector_comparison/PP-OCRv5_mobile_det/Photo_8.json',
         'jh_photo2.jpg': 'whiteboard-ocr/outputs/jh_full_photos_app/Photo_2.json',
         'jh_photo5.jpg': 'whiteboard-ocr/outputs/jh_full_photos_app/Photo_5.json'}
run = json.loads(Path(sys.argv[1]).read_text())
ok = True
for fid, rel in SAVED.items():
    ref = json.loads((A / rel).read_text())['grouped']; got = run['results'][fid]['lines']
    n = len(ref) == len(got)
    box = max((abs(a - b) for x, y in zip(ref, got) for pa, pb in zip(x['box'], y['box']) for a, b in zip(pa, pb)), default=0) if n else None
    same = sum(x['text'] == y['text'] for x, y in zip(ref, got))
    good = n and box is not None and box <= 0.01 and same == len(ref); ok &= good
    print(f"{fid:16s} saved {len(ref):2d} lines | this run {len(got):2d} | max box diff {box} | identical text {same}/{len(ref)} | {'IDENTICAL' if good else 'DIFFERENT'}")
    for x, y in zip(ref, got):
        if x['text'] != y['text']: print(f"    saved {x['text']!r} | now {y['text']!r}")
print('env of this run:', {k: run['env'].get(k) for k in ('platform', 'paddlepaddle', 'cv2_in_use')})
sys.exit(0 if ok else 1)
