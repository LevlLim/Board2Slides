import sys, os, json, math, yaml, paddle, numpy as np
sys.path.insert(0, '/home/claude/rdtest/repo')
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
U = "/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment"
EXPORTS = {'pretrained': '/root/.paddlex/official_models/PP-OCRv5_server_rec/inference',
           'S0': f'{U}/results_strategies_s0s3/results/models/PP-OCRv5_server_rec_board2notes_S0/inference',
           'S0s2': f'{U}/results_strategies_s0s3/results/models/PP-OCRv5_server_rec_board2notes_S0s2/inference',
           'G0': f'{U}/results_guide_g0/results/models/PP-OCRv5_server_rec_board2notes_G0/inference',
           'G0s2': f'{U}/results_guide_g0/results/models/PP-OCRv5_server_rec_board2notes_G0s2/inference'}
MAP = json.load(open('/home/claude/sweep/export_name_map.json'))
def group(name):
    import re
    m = re.match(r"backbone\.stages\.(\d)\.", name)
    if m: return f"stage {int(m.group(1)) + 1}"
    if name.startswith("backbone.stem."): return "stem"
    if name.startswith("head.ctc_encoder."): return "CTC neck (SVTR)"
    if name.startswith("head.ctc_head."): return "CTC head"
    return None
GROUPS = ["stem", "stage 1", "stage 2", "stage 3", "stage 4", "CTC neck (SVTR)", "CTC head"]
def named(model_id):
    e = paddle.jit.load(EXPORTS[model_id]).state_dict()
    assert len(e) == len(MAP), (model_id, len(e))
    return {MAP[k]: v.numpy() for k, v in e.items()}
def build(model_id):
    cfg = yaml.safe_load(open('/home/claude/rdtest/b2n/experiments/strategies_s0s3/runs/S0/config.yml'))
    post = build_post_process(cfg['PostProcess'], cfg['Global']); n = len(post.character)
    cfg['Architecture']['Head']['out_channels_list'] = {'CTCLabelDecode': n, 'SARLabelDecode': n + 2, 'NRTRLabelDecode': n + 3}
    paddle.seed(0); m = build_model(cfg['Architecture'])
    w = named(model_id); sd = m.state_dict()
    for k, v in w.items():
        assert tuple(sd[k].shape) == v.shape, k
    m.set_state_dict({k: paddle.to_tensor(v) for k, v in w.items()}); m.eval()
    return m, post
