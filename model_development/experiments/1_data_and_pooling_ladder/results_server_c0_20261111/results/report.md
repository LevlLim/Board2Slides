# Server ladder: pretrained -> C0 -> C1 -> P40 -> P128 -> V3

Experiment `server_c0_20261111` trains only C0; the other rows are the versions_2x2 predictions (identical data lists checked; pretrained reads whiteboard validation 100% identically, so the same scorer). One seed per model; whiteboard validation picked each checkpoint and is the primary measure.

| model | what | whiteboard val | CNN test | long sentences (held out) | IAM val |
|---|---|---|---|---|---|
| pretrained | PP-OCRv5_server_rec as downloaded | 15.48 | 45.79 | 30.93 | 12.20 |
| C0 | V2 data (our crops, no augmented copies), PaddleOCR's original recipe (48x320, 40 positions) | 8.88 | 29.00 | 7.84 | 30.57 |
| C1 | V3 data, PaddleOCR's original recipe (48x320, 40 positions) | 9.30 | 28.72 | 8.81 | 31.40 |
| P40 | pooling-only pair: original training pooling (40 positions), lines that fit 40 | 54.54 | 93.88 | 45.26 | 60.98 |
| P128 | pooling-only pair: our training pooling (width/8 positions), SAME lines as P40 | 6.44 | 26.60 | 4.55 | 21.68 |
| V3 | D: our crops, short + long lines + our augmented copies | 7.18 | 24.27 | 2.65 | 25.71 |
| V2 | C: our crops, short + long lines, no augmented copies (replaces IAM) | 7.31 | 23.72 | 2.38 | 23.26 |

Mean line CER (%). Each step, A - B in points with a 95% interval (whole photos / sentences resampled):

| step | what changes | whiteboard val | CNN test | long sentences (held out) | IAM val |
|---|---|---|---|---|---|
| C0 vs pretrained | fine-tuning on our crops (original recipe) | -6.60 [-9.71, -3.37] (better) | -16.79 [-20.13, -13.40] (better) | -23.09 [-33.52, -10.64] (better) | +18.38 [+17.57, +19.16] (worse) |
| C1 vs C0 | + our augmented copies (original recipe) | +0.42 [-0.49, +1.46] | -0.28 [-1.16, +0.62] | +0.97 [+0.04, +2.02] (worse) | +0.83 [+0.36, +1.29] (worse) |
| P40 vs C1 | input 48x320 -> 64x1024, still 40 positions (same list) | +45.24 [+40.08, +48.37] (worse) | +65.16 [+60.21, +69.86] (worse) | +36.46 [+28.14, +46.42] (worse) | +29.58 [+28.43, +30.69] (worse) |
| P128 vs P40 | training pooling 40 -> width/8 (same list) | -48.10 [-51.00, -43.65] (better) | -67.27 [-71.80, -62.91] (better) | -40.72 [-48.68, -33.96] (better) | -39.30 [-40.45, -38.06] (better) |
| V3 vs P128 | + lines needing more than 40 positions | +0.74 [-0.22, +1.99] | -2.33 [-3.74, -0.86] (better) | -1.90 [-3.56, -0.29] (better) | +4.02 [+3.62, +4.42] (worse) |
| V2 vs C0 | side: adapted vs original recipe, no copies (several changes) | -1.57 [-3.14, -0.48] (better) | -5.28 [-6.77, -3.62] (better) | -5.46 [-8.82, -2.16] (better) | -7.31 [-7.97, -6.69] (better) |
| V3 vs C1 | side: adapted vs original recipe, with copies (several changes) | -2.12 [-3.72, -1.07] (better) | -4.45 [-6.37, -2.54] (better) | -6.16 [-10.23, -2.30] (better) | -5.69 [-6.20, -5.21] (better) |

C0 on the new-writer test: 42.38% (the old runs were not scored on it on the GPU, so no ladder for this set).

C0 training: 1113 lines, 86 epochs, 6020 updates, 9.6 min; best: norm_edit_dis: 0.8817494989496045, is_float16: False, acc: 0.4926470225994836, f

Lists: {'C0_lines': 1113, 'C1_lines': 4269, 'copies_added_in_C1': 3156, 'copies_whose_crop_is_not_in_C0': 0}.
Limits: one seed each (seed-to-seed noise on whiteboard validation was about 1 point); C0 was trained today, the others on 6 Nov with the same training code and data (repo changes since then touch only the mobile backbone).
