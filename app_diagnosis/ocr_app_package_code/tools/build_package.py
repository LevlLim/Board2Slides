#!/usr/bin/env python3
"""Build board2notes_ocr_package/ from the live app, its two model folders and fixed fixtures.

Run from the Assignment folder (any Python >= 3.9 with PyYAML):
    python3 board2notes_ocr_package/tools/build_package.py --detector-dir ~/.paddlex/official_models/PP-OCRv5_mobile_det

The live app (whiteboard-ocr/ocr_prototype, whiteboard-ocr/preprocessing/text_lines.py) is only READ.
Vendored copies receive the small, listed patches below; everything else is byte-identical.
"""
from __future__ import annotations

import argparse, hashlib, json, re, shutil, time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
A = PKG.parent
APP = A / 'whiteboard-ocr/ocr_prototype'
TL = A / 'whiteboard-ocr/preprocessing/text_lines.py'
S1 = A / 'results_strategies_s0s3/results/models/PP-OCRv5_server_rec_board2notes_S1'
CODE = PKG / 'b2n_ocr'
MODELS = PKG / 'models'
DET_FILES = ('inference.json', 'inference.pdiparams', 'inference.yml')
REC_FILES = ('inference.json', 'inference.pdiparams', 'inference.yml', 'verified.json')
EXPECTED_DET_PARAMS = 'afa1820cb16c1fd0dad589d0f8b389139061c1ef6d68019685fd07be997dda5b'   # used by every recorded app run
EXPECTED_REC_PARAMS = '067b34a571b6f1521534596e74788c6cad80de306c497cd84e439724e8e5b07d'   # S1 seed 1 export


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def patch(text, old, new, name):
    if text.count(old) != 1:
        raise SystemExit(f'patch {name}: expected exactly one match, found {text.count(old)}; the live app changed - review the patch list')
    return text.replace(old, new)


PATCHES = {
 'ocr_backend.py': [
  ('P1 package-relative paths',
   "ASSIGNMENT = Path(__file__).resolve().parents[2]\nPROJECT = ASSIGNMENT / 'whiteboard-ocr'\n",
   "PROJECT = Path(__file__).resolve().parent            # packaged: vendored modules live here\n"
   "PACKAGE = Path(os.environ.get('B2N_PACKAGE_ROOT', PROJECT.parent))\n"),
  ('P2 only the packaged S1 reader',
   "MODEL_GROUPS = {\n    'results_versions_2x2': ['V1b', 'V2short', 'V3short', 'V2', 'V3', 'C1', 'P40', 'P128'],\n"
   "    'results_strategies_s0s3': ['S0', 'S0s2', 'S1', 'S1s2', 'S2', 'S2s2', 'S3', 'S3s2'],\n    'results_guide_g0': ['G0', 'G0s2'],\n}\n"
   "MODELS = {name: ASSIGNMENT / group / 'results/models' / f'PP-OCRv5_server_rec_board2notes_{name}'\n          for group, names in MODEL_GROUPS.items() for name in names}\n",
   "MODELS = {'S1': PACKAGE / 'models' / 'PP-OCRv5_server_rec_board2notes_S1'}   # packaged: S1 seed 1 only\n"
   "DETECTOR_NAME = 'PP-OCRv5_mobile_det'\nDETECTOR_DIR = PACKAGE / 'models' / DETECTOR_NAME\n"
   "# Plain Paddle CPU engine everywhere (macOS arm64 builds have no oneDNN); stops Linux x86 silently switching to oneDNN.\n"
   "DETECTOR_RUNTIME = dict(device='cpu', enable_mkldnn=False)\n"
   "if os.environ.get('B2N_DET_CPU_THREADS'):   # default: PaddleOCR's own (10), as tested on the Mac; set 1 on Linux aarch64 + Paddle 3.2.2 (multi-thread crash)\n"
   "    DETECTOR_RUNTIME['cpu_threads'] = int(os.environ['B2N_DET_CPU_THREADS'])\n"),
  ('P3 torch import only where torch is installed',
   "        import torch  # noqa: F401\n",
   "        try:\n            import torch  # noqa: F401  (only where torch is co-installed, e.g. the original Mac env)\n"
   "        except ImportError:\n            pass\n"),
  ('P4 packaged detector weights, no download',
   "self.detector = tl.Detector('PP-OCRv5_mobile_det', det_args={'thresh': .20, 'box_thresh': .40, 'unclip_ratio': 1.2})",
   "self.detector = tl.Detector(DETECTOR_NAME, det_args={'thresh': .20, 'box_thresh': .40, 'unclip_ratio': 1.2},\n"
   "                                            model_dir=DETECTOR_DIR, runtime=DETECTOR_RUNTIME)"),
 ],
 'preprocessing/text_lines.py': [
  ('P5 Detector accepts a local model folder and runtime options',
   '    def __init__(self, model_name="PP-OCRv5_mobile_det", det_args=None):\n        import os\n'
   '        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")\n        from paddleocr import TextDetection\n'
   '        self.model_name = model_name\n        self.model = TextDetection(model_name=model_name)\n',
   '    def __init__(self, model_name="PP-OCRv5_mobile_det", det_args=None, model_dir=None, runtime=None):\n        import os\n'
   '        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")\n        from paddleocr import TextDetection\n'
   '        self.model_name = model_name\n        extra = dict(runtime or {})\n'
   '        if model_dir is not None:\n            if not (Path(model_dir) / "inference.pdiparams").is_file():\n'
   '                raise FileNotFoundError(f"Packaged detector weights missing: {model_dir}")\n            extra["model_dir"] = str(model_dir)\n'
   '        self.model = TextDetection(model_name=model_name, **extra)\n'),
  ('P5b Path import for P5', 'import cv2\nimport numpy as np\n', 'from pathlib import Path\n\nimport cv2\nimport numpy as np\n'),
 ],
 'student_app.py': [
  ('P6 packaged sample photo',
   'SAMPLE_PHOTO = Path(__file__).resolve().parents[2] / "Jy Dataset" / "Photo 23.jpg"',
   'SAMPLE_PHOTO = Path(__file__).resolve().parents[1] / "fixtures" / "photos" / "jy_photo23.jpg"   # packaged copy of Jy Photo 23'),
 ],
}
VERBATIM = ['student_flow.py', 'visual_review.py', 'app.py', 'region_viewer/index.html', '.streamlit/config.toml',
            'tests/test_student_flow.py', 'tests/test_student_app.py']

FIXTURES = [  # (fixture id, source relative to Assignment, purpose)
    ('jy_photo23.jpg', 'Jy Dataset/Photo 23.jpg', 'app sample photo; 9 regions, single column'),
    ('jy_photo8.jpg', 'Jy Dataset/Photo 8.jpg', 'dense multi-column photo (~38 regions)'),
    ('jh_photo2.jpg', 'JH_datasetNov5Sentence/Photo2.jpg', '2304x4096 phone photo; raw split joined by grouping; eraser region with empty reading'),
    ('jh_photo5.jpg', 'JH_datasetNov5Sentence/Photo5.jpg', 'full-resolution phone photo, 11 regions incl. faint extra text'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector-dir', required=True, type=Path)
    a = ap.parse_args()
    det_src = a.detector_dir.expanduser()
    # ---- code
    for rel, plist in PATCHES.items():
        src = (TL if rel.endswith('text_lines.py') else APP / rel)
        text = src.read_text()
        for name, old, new in plist:
            text = patch(text, old, new, name)
        dst = CODE / rel; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_text(text)
    (CODE / 'preprocessing/__init__.py').write_text('"""Packaged subset of Board2Notes preprocessing: text_lines only."""\n')
    for rel in VERBATIM:
        dst = CODE / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(APP / rel, dst)
    # ---- models (inference files only)
    for name, src, files, expect in (('PP-OCRv5_mobile_det', det_src, DET_FILES, EXPECTED_DET_PARAMS),
                                     ('PP-OCRv5_server_rec_board2notes_S1', S1, REC_FILES, EXPECTED_REC_PARAMS)):
        if sha(src / 'inference.pdiparams') != expect:
            raise SystemExit(f'{name}: weights differ from the recorded app runs; not packaging')
        (MODELS / name).mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copyfile(src / f, MODELS / name / f)
    # ---- fixtures
    fx = PKG / 'fixtures/photos'; fx.mkdir(parents=True, exist_ok=True)
    for fid, src, _ in FIXTURES:
        shutil.copyfile(A / src, fx / fid)
    # ---- manifest
    import yaml
    rec_yml = yaml.safe_load((MODELS / 'PP-OCRv5_server_rec_board2notes_S1/inference.yml').read_text())
    det_yml = yaml.safe_load((MODELS / 'PP-OCRv5_mobile_det/inference.yml').read_text())
    chars = rec_yml['PostProcess']['character_dict']
    # integrity set checked by tools/verify_package.py: model files, vendored code and fixture photos
    files = sorted(p for d in ('models', 'b2n_ocr', 'fixtures/photos') for p in (PKG / d).rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts)
    manifest = dict(
        package='board2notes-ocr', version='2026-11-09', created=time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        configuration='PP-OCRv5_mobile_det -> existing line grouping and quad cropping -> S1 seed-1 recogniser -> editable OCR text',
        status='provisional development configuration; development evidence only: notebook 15 page CER 27.5% on 16 provisionally annotated Jy photos (mostly reading order), notebook 16 0.87% mean line CER on 59 matched JH lines, earlier crop-level evaluations in PROJECT_PROGRESS.md; no fresh-writer confirmation yet',
        detector=dict(name='PP-OCRv5_mobile_det', folder='models/PP-OCRv5_mobile_det',
                      app_args=dict(thresh=0.20, box_thresh=0.40, unclip_ratio=1.2), detect_call_unclip=1.2,
                      passes='quick pass with long side limited to 1600 px (limit_type max); if median line height < 20 px as seen by the detector, re-detect text regions at full resolution (<= 4000 px)',
                      runtime=dict(device='cpu', enable_mkldnn=False, engine='paddle static (PaddleX default)', cpu_threads='PaddleOCR default (10); env B2N_DET_CPU_THREADS overrides (use 1 on Linux aarch64 Paddle 3.2.2)'),
                      model_defaults_overridden=dict(thresh=det_yml['PostProcess']['thresh'], box_thresh=det_yml['PostProcess']['box_thresh'],
                                                     unclip_ratio=det_yml['PostProcess']['unclip_ratio'])),
        grouping=dict(function='text_lines.group_into_lines', gap_factor=1.2, min_v_overlap=0.5,
                      order='ocr_backend.spatial_order (provisional column/row heuristic; provisionally evaluated in notebook 15, improvement deferred)'),
        cropping=dict(function='text_lines.quad_crop', pad=0.0, interpolation='INTER_CUBIC, BORDER_REPLICATE'),
        recogniser=dict(name='S1 seed 1 (PP-OCRv5_server_rec fine-tuned, results_strategies_s0s3)', folder='models/PP-OCRv5_server_rec_board2notes_S1',
                        input_shape=next(op['RecResizeImg']['image_shape'] for op in rec_yml['PreProcess']['transform_ops'] if 'RecResizeImg' in op),
                        resize='ocr_backend.resize_for_reader (= PaddleOCR resize_norm_img_chinese incl. rounding): height 64, width = ceil(64*w/h), zero-padded to >= 1024, (x/255-0.5)/0.5',
                        decode='greedy CTC (argmax, collapse repeats, drop blank); space class appended when the output has len(dict)+2 classes',
                        runtime='Paddle inference CPU, ir_optim off, custom passes [] (verified export policy), oneDNN off, 4 threads, one crop at a time',
                        dictionary=dict(characters=len(chars), sha256_utf8_newline_joined=hashlib.sha256('\n'.join(chars).encode()).hexdigest(),
                                        source='embedded in inference.yml PostProcess.character_dict')),
        preprocessing='EXIF orientation applied once (Pillow exif_transpose), RGB->BGR; optional user rotate/trim in the UI; NO enhancement, perspective correction, deskew, tightening, SR, VLM or LLM',
        excluded='training checkpoints, optimizer state, datasets (only 4 fixture photos), other readers, credentials, API keys, private configuration, notebooks',
        source_files={str(p.relative_to(A)): sha(p) for p in [TL] + [APP / r for r in PATCHES if r != 'preprocessing/text_lines.py'] + [APP / r for r in VERBATIM]},
        patches={rel: [n for n, _, _ in pl] for rel, pl in PATCHES.items()},
        fixtures=[dict(id=f, source=s, purpose=p) for f, s, p in FIXTURES],
        files={str(p.relative_to(PKG)): dict(sha256=sha(p), bytes=p.stat().st_size) for p in files})
    (PKG / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print('package built:', PKG, '|', len(files), 'files |', round(sum(p.stat().st_size for p in files) / 2**20, 1), 'MB')


if __name__ == '__main__':
    main()
