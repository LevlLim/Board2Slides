import sys, json, math, os
from pathlib import Path
A = Path(os.environ["A"])
import cv2, numpy as np, yaml
import paddle.inference as pi
exec(open(Path.home()/'b2n/score_static_mac.py').read().split('DICT = ')[0].split('import paddle.inference as pi')[1])
D = A/'claude_margin_diagnostic_20261109'
R = json.load(open(D/'RESULTS_MODELS.json')); C = {c['name']: c for c in R['crops']}
md = A/'results_strategies_s0s3/results/models/PP-OCRv5_server_rec_board2notes_S1'
y = yaml.safe_load(open(md/'inference.yml')); table = [''] + list(y['PostProcess']['character_dict']) + [' ']
cfg = pi.Config(str(md/'inference.json'), str(md/'inference.pdiparams')); cfg.disable_gpu(); cfg.switch_ir_optim(False); cfg.enable_custom_passes([], True); cfg.disable_glog_info()
pred = pi.create_predictor(cfg); inp = pred.get_input_handle(pred.get_input_names()[0])
out = {}
for n, c in C.items():
    img = cv2.imread(str(D/'crops_tight'/(Path(n).stem + '.png'))); pad, _ = resize_norm_img_chinese(img, [3, 64, 1024])
    inp.reshape([1, 3, 64, pad.shape[2]]); inp.copy_from_cpu(pad[None]); pred.run()
    prob = pred.get_output_handle(pred.get_output_names()[0]).copy_to_cpu()[0]; idx = prob.argmax(-1)
    T = prob.shape[0]; text_pos = math.ceil(64*img.shape[1]/img.shape[0]) * T / pad.shape[2]
    keep = [k for k, i in enumerate(idx) if i != 0 and (k == 0 or i != idx[k-1])]
    inside = ''.join(table[idx[k]] for k in keep if k < math.ceil(text_pos)); outside = ''.join(table[idx[k]] for k in keep if k >= math.ceil(text_pos))
    out[n] = dict(T=int(T), text_positions=round(text_pos, 1), chars_inside=len(inside), chars_in_padding=len(outside), padding_text=outside,
                  per_position=''.join(table[i] if i else '_' for i in idx))
json.dump(out, open(D/'persistent_review/s1_position_trace.json', 'w'), indent=1, ensure_ascii=False)
P = set(R['persistent_tight'])
import statistics
a = [v['chars_in_padding'] for k, v in out.items() if k in P]; b = [v['chars_in_padding'] for k, v in out.items() if k not in P]
print('persistent: crops with text emitted in padding', sum(x > 0 for x in a), '/', len(a), '| others', sum(x > 0 for x in b), '/', len(b))
for k in R['persistent_tight']:
    print(k, out[k]['text_positions'], out[k]['chars_inside'], out[k]['chars_in_padding'], repr(out[k]['padding_text']))
