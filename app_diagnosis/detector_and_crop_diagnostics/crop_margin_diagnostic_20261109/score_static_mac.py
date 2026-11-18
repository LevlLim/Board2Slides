"""Score exported Board2Notes readers on the Mac (CPU), the same way b2n_versions.py's static reader does.
usage: python3 score_static_mac.py LABELS IMAGE_ROOT OUT_JSON MODEL_DIR [MODEL_DIR ...]
LABELS: lines 'relative/path<TAB>text' relative to IMAGE_ROOT. Height/min width come from verified.json export_h
(64 -> 64x1024, 48 -> 48x320 for C1)."""
import json, sys, os
from pathlib import Path
A = Path(os.environ.get("A", Path.home() / "mnt/Computer Vision & Deep Learning/Assignment"))
import math
import cv2, numpy as np, yaml
import paddle.inference as pi


def resize_norm_img_chinese(img, image_shape):   # verbatim from PaddleOCR @ 91279fee (the RunPod checkout)
    imgC, imgH, imgW = image_shape
    max_wh_ratio = imgW * 1.0 / imgH
    h, w = img.shape[0], img.shape[1]
    ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, ratio)
    imgW = int(imgH * max_wh_ratio)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype("float32")
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im, min(1.0, float(resized_w / imgW))
DICT = A / "PaddleOCR-fine-tune/ppocr/utils/dict/ppocrv5_dict.txt"
labels, root, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
items = [l.rstrip("\n").split("\t", 1) for l in open(labels, encoding="utf-8") if l.strip()]
res = json.load(open(out)) if out.exists() else {}
for md in map(Path, sys.argv[4:]):
    name = md.name.replace("PP-OCRv5_server_rec_board2notes_", "")
    v = json.load(open(md / "verified.json"))
    H = int(v["identity"]["export_h"]); minW = 320 if name.startswith("C1") else 1024
    y = yaml.safe_load(open(md / "inference.yml", encoding="utf-8"))
    chars = (y.get("PostProcess") or {}).get("character_dict") or open(DICT, encoding="utf-8").read().splitlines()
    graph = md / "inference.json" if (md / "inference.json").exists() else md / "inference.pdmodel"
    cfg = pi.Config(str(graph), str(md / "inference.pdiparams")); cfg.disable_gpu()
    cfg.switch_ir_optim(False); cfg.enable_custom_passes([], True)
    pred = pi.create_predictor(cfg); inp = pred.get_input_handle(pred.get_input_names()[0])
    table, preds = None, {}
    for p, _ in items:
        img = cv2.imread(str(root / p))
        if img is None: raise SystemExit(f"unreadable {p}")
        pad, _ = resize_norm_img_chinese(img, [3, H, minW]); W = pad.shape[2]
        inp.reshape([1, 3, H, W]); inp.copy_from_cpu(pad[None]); pred.run()
        prob = pred.get_output_handle(pred.get_output_names()[0]).copy_to_cpu()[0]
        if table is None:
            table = [""] + list(chars) + ([" "] if prob.shape[-1] == len(chars) + 2 else [])
            assert prob.shape[-1] == len(table)
        idx = prob.argmax(-1)
        preds[p] = "".join(table[i] for k, i in enumerate(idx) if i != 0 and (k == 0 or i != idx[k - 1]))
    res[name] = preds
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=0)
    print(name, H, minW, len(preds), flush=True)
