import sys, os, json, paddle, numpy as np
fake = paddle.load('/home/claude/sweep/unique.pdparams')
e = paddle.jit.load('/home/claude/sweep/exp_fake/inference').state_dict()
# match every export param to the structured name with identical values
index = {}
for k, v in fake.items():
    a = v.numpy(); index.setdefault((a.shape, round(float(a.ravel()[:8].sum()), 6), round(float(a.sum()), 4)), []).append(k)
mp = {}; amb = 0
for k, v in e.items():
    a = v.numpy(); c = index.get((a.shape, round(float(a.ravel()[:8].sum()), 6), round(float(a.sum()), 4)), [])
    c = [x for x in c if np.array_equal(fake[x].numpy(), a)]
    if len(c) != 1: amb += 1; continue
    mp[k] = c[0]
print('mapped', len(mp), 'of', len(e), 'ambiguous/unmatched', amb)
ref = paddle.jit.load("/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment/results_strategies_s0s3/results/models/PP-OCRv5_server_rec_board2notes_S0/inference").state_dict()
bad = [k for k in mp if tuple(ref[k].shape) != tuple(e[k].shape)]
print('shape check against the real S0 export, mismatches:', len(bad), '| same param count', len(ref) == len(e))
json.dump(mp, open('/home/claude/sweep/export_name_map.json', 'w'), indent=0)
