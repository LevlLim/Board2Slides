"""Local OCR inference using our exported readers and the existing mobile detector.

No LLM, training, learned SR or automatic board-perspective correction is used.
Recognition resizing/decoding follows the verified RunPod scorer.
"""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import math
import os
import sys
import threading
import time

os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
os.environ.setdefault('FLAGS_use_mkldnn', '0')
import cv2
import numpy as np
from PIL import Image, ImageOps
import yaml

ASSIGNMENT = Path(__file__).resolve().parents[2]
PROJECT = ASSIGNMENT / 'whiteboard-ocr'
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

MODEL_GROUPS = {
    'results_versions_2x2': ['V1b', 'V2short', 'V3short', 'V2', 'V3', 'C1', 'P40', 'P128'],
    'results_strategies_s0s3': ['S0', 'S0s2', 'S1', 'S1s2', 'S2', 'S2s2', 'S3', 'S3s2'],
    'results_guide_g0': ['G0', 'G0s2'],
}
MODELS = {name: ASSIGNMENT / group / 'results/models' / f'PP-OCRv5_server_rec_board2notes_{name}'
          for group, names in MODEL_GROUPS.items() for name in names}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def load_image(path):
    """Decode original pixels with EXIF orientation; return BGR for Paddle/OpenCV."""
    if Path(path).suffix.lower() in ('.heic', '.heif'):
        import pillow_heif
        pillow_heif.register_heif_opener()
    with Image.open(path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def resize_for_reader(img, shape):
    """Equivalent to pinned PaddleOCR resize_norm_img_chinese (including rounding)."""
    channels, height, min_width = shape
    h, w = img.shape[:2]
    ratio = w / h
    width = int(height * max(min_width / height, ratio))
    resized_width = min(width, int(math.ceil(height * ratio)))
    x = cv2.resize(img, (resized_width, height)).astype('float32')
    x = x.transpose(2, 0, 1) / 255.0
    x = (x - .5) / .5
    out = np.zeros((channels, height, width), dtype=np.float32)
    out[:, :, :resized_width] = x
    return out


class RecognitionModel:
    def __init__(self, model_id='G0s2', threads=4):
        if model_id not in MODELS:
            raise ValueError(f'Unknown model {model_id!r}; choose {list(MODELS)}')
        self.model_id = model_id
        self.model_dir = MODELS[model_id]
        self.threads = threads
        self._lock = threading.Lock()
        graph = self.model_dir / 'inference.json'
        params = self.model_dir / 'inference.pdiparams'
        config_path = self.model_dir / 'inference.yml'
        for p in (graph, params, config_path):
            if not p.is_file():
                raise FileNotFoundError(f'Missing exported model file: {p}')
        config = yaml.safe_load(config_path.read_text())
        self.characters = config['PostProcess']['character_dict']
        self.shape = next(op['RecResizeImg']['image_shape'] for op in config['PreProcess']['transform_ops']
                          if 'RecResizeImg' in op)
        # Existing Mac environments have a libiomp conflict when Paddle is loaded before Torch.
        import torch  # noqa: F401
        import paddle.inference as pi
        cfg = pi.Config(str(graph), str(params))
        cfg.disable_gpu()
        cfg.set_cpu_math_library_num_threads(threads)
        if hasattr(cfg, 'disable_mkldnn'):
            cfg.disable_mkldnn()
        cfg.switch_ir_optim(False)
        if not hasattr(cfg, 'enable_custom_passes'):
            raise RuntimeError('This Paddle build cannot use our verified custom-pass policy.')
        cfg.enable_custom_passes([], True)
        cfg.disable_glog_info()
        self.predictor = pi.create_predictor(cfg)
        self.input = self.predictor.get_input_handle(self.predictor.get_input_names()[0])
        self.output = self.predictor.get_output_handle(self.predictor.get_output_names()[0])
        self.table = None
        self._identity = {
            'model_id': model_id, 'graph_sha256': sha256(graph), 'params_sha256': sha256(params),
            'config_sha256': sha256(config_path), 'image_shape': self.shape,
            'runtime': 'cpu_pir_no_default_passes_official_resize_v2', 'threads': threads,
            'paddle': version('paddlepaddle'), 'opencv': cv2.__version__,
            'backend_sha256': sha256(Path(__file__)),
        }

    def identity(self):
        return dict(self._identity)

    def read(self, images):
        rows = []
        with self._lock:
            for img in images:
                start = time.perf_counter()
                if img is None or img.size == 0:
                    raise ValueError('An empty crop was supplied to the reader.')
                x = resize_for_reader(img, self.shape)
                self.input.reshape([1, *x.shape])
                self.input.copy_from_cpu(x[None])
                self.predictor.run()
                prob = self.output.copy_to_cpu()[0]
                if not np.isfinite(prob).all():
                    raise RuntimeError('Reader returned nonfinite probabilities.')
                if self.table is None:
                    self.table = [''] + list(self.characters)
                    if prob.shape[-1] == len(self.characters) + 2:
                        self.table.append(' ')
                    if prob.shape[-1] != len(self.table):
                        raise RuntimeError('Exported character dictionary does not match output classes.')
                indices = prob.argmax(-1)
                keep = [i for i, value in enumerate(indices) if value != 0 and (i == 0 or value != indices[i - 1])]
                text = ''.join(self.table[int(indices[i])] for i in keep)
                confidence = float(np.mean([prob[i, indices[i]] for i in keep])) if keep else 0.0
                rows.append(dict(text=text, score=confidence, seconds=time.perf_counter() - start))
        return rows


def spatial_order(boxes):
    """Use vertical whitespace to separate simple columns, then order rows within each.

    This geometry-only heuristic is provisional; complex layouts need review.
    """
    from preprocessing import text_lines as tl
    if not boxes:
        return []
    ranges = sorted((float(p[:, 0].min()), float(p[:, 0].max())) for p in boxes)
    merged = []
    for a, b in ranges:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    threshold = max(20.0, 1.5 * float(np.median([tl.line_height(b) for b in boxes])))
    cuts = [(b + c) / 2 for (_, b), (c, _) in zip(merged, merged[1:]) if c - b > threshold]
    columns = [[] for _ in range(len(cuts) + 1)]
    for i, p in enumerate(boxes):
        columns[sum(float(p[:, 0].mean()) > cut for cut in cuts)].append(i)
    return [col[j] for col in columns for j in tl.reading_order([boxes[i] for i in col]) if col]


class OCRService:
    def __init__(self, model_id='G0s2', threads=4):
        self.reader = RecognitionModel(model_id, threads=threads)
        self.detector = None
        self._lock = threading.Lock()

    def recognise(self, image_path):
        from preprocessing import text_lines as tl
        detector_load = 0.0
        with self._lock:
            if self.detector is None:
                begin = time.perf_counter()
                self.detector = tl.Detector('PP-OCRv5_mobile_det', det_args={'thresh': .20, 'box_thresh': .40, 'unclip_ratio': 1.2})
                detector_load = time.perf_counter() - begin
            start = time.perf_counter()
            image = load_image(image_path)
            boxes, det_info = self.detector.detect(image, unclip=1.2)
            boxes, _ = tl.group_into_lines(boxes)
            boxes = [boxes[i] for i in spatial_order(boxes)]
            crops = [tl.quad_crop(image, p) for p in boxes]
            readings = self.reader.read(crops)
            lines = [dict(index=i + 1, text=r['text'], score=r['score'],
                          box=np.asarray(p).astype(float).tolist(), recognition_seconds=r['seconds'])
                     for i, (p, r) in enumerate(zip(boxes, readings))]
            return dict(text='\n'.join(r['text'] for r in lines), lines=lines,
                        seconds=time.perf_counter() - start, detector_cold_load_seconds=detector_load,
                        image_width=int(image.shape[1]), image_height=int(image.shape[0]),
                        model=self.reader.identity(), detector=dict(name='PP-OCRv5_mobile_det', unclip_ratio=1.2,
                        thresh=.20, box_thresh=.40, info=det_info),
                        preprocessing='EXIF orientation, existing detection/retry, line grouping, quad crop, required reader resize',
                        reading_order='simple column whitespace then top-to-bottom; provisional')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run our OCR-only prototype on one image.')
    parser.add_argument('image', type=Path)
    parser.add_argument('--model', default='G0s2', choices=list(MODELS))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = OCRService(args.model).recognise(args.image)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(result['text'])
    print(f"\n{len(result['lines'])} regions; {result['seconds']:.2f}s; {args.model}")
