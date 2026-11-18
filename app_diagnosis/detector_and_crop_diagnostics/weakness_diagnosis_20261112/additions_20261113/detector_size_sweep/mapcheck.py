import sys, os, yaml, paddle, numpy as np
sys.path.insert(0, os.getcwd())
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
cfg = yaml.safe_load(open('/home/claude/rdtest/b2n/experiments/strategies_s0s3/runs/S0/config.yml'))
n = len(build_post_process(cfg['PostProcess'], cfg['Global']).character)
cfg['Architecture']['Head']['out_channels_list'] = {'CTCLabelDecode': n, 'SARLabelDecode': n + 2, 'NRTRLabelDecode': n + 3}
m = build_model(cfg['Architecture'])
sd = m.state_dict()
keep = [(k, tuple(v.shape)) for k, v in sd.items() if not k.startswith(('head.gtc_head', 'head.before_gtc', 'backbone.last_conv', 'backbone.fc'))]
p = "/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment/results_strategies_s0s3/results/models/PP-OCRv5_server_rec_board2notes_S0/inference"
e = paddle.jit.load(p).state_dict()
es = sorted([(k, tuple(v.shape)) for k, v in e.items()], key=lambda kv: int(kv[0].split('_')[1]))
print([s for _, s in es[:12]]); print([(k.split('.')[-1], s) for k, s in keep[:12]])
