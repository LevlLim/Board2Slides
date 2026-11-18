# Board2Notes reader versions: results

Experiment `mobile_official_20261110`; about 6000 updates x 16 lines per training run (exact counts in 'Training runs'; runs on the same training list have identical counts); one seed. **Mean line CER**: per crop, case-insensitive, spaces normalised (lower is better). **Exact**: crop read perfectly, case-sensitive. **Repeated insertions**: aligned inserted characters inside repeated substrings of 1-16 characters, per crop. This is a diagnostic, not proof of a causal mechanism. Raw insertion counts and the older adjacent-double diagnostic are saved in predictions.csv.

Every training run gets PaddleOCR's built-in random augmentation (RecAug, in memory).

The held-out long sentences are new SENTENCES in the SAME handwriting on the SAME board; they do not test a new writer or a new board.

| version | what |
|---|---|
| V1 | historical Baseline_model (Oct 13), read as deployed (48 px, width follows the crop) |
| pretrained | PP-OCRv5_server_rec as downloaded, no fine-tuning (48 px) |
| MC0 | mobile control C0: guidance on, CTC/NRTR 1.0/1.0, uniform LR (seed 1) |
| MG | C0 + guidance off (seed 1) |
| ML | C0 + six-group backbone LR [0.1, 0.1, 0.1, 0.25, 0.5, 1.0] (seed 1) |
| ME | C0 + SE added to blocks6.3 (seed 1) |
| MC0s2 | mobile control C0: guidance on, CTC/NRTR 1.0/1.0, uniform LR (seed 2) |
| MGs2 | C0 + guidance off (seed 2) |
| MLs2 | C0 + six-group backbone LR [0.1, 0.1, 0.1, 0.25, 0.5, 1.0] (seed 2) |
| MEs2 | C0 + SE added to blocks6.3 (seed 2) |
| MV1b | mobile IAM handwriting lines only, adapted recipe |
| MV2short | mobile A: our crops, short lines only (40 chars or fewer), no augmented copies |
| MV3short | mobile B: our crops, short lines only + our augmented copies |
| MV2 | mobile C: our crops, short + long lines, no augmented copies (replaces IAM) |
| MV3 | mobile D: our crops, short + long lines + our augmented copies |
| MC1 | mobile V3 data, PaddleOCR's original recipe (48x320, 40 positions) |
| MP40 | mobile pooling-only pair: original training pooling (40 positions), lines that fit 40 |
| MP128 | mobile pooling-only pair: our training pooling (width/8 positions), SAME lines as P40 |
| MT | teammate's Colab recipe (64x960, heights 32/48/64, 40 positions, LR 5e-5) on MC1/MP40's lines |
| MC0avg | per-crop mean of MC0 and MC0s2 |
| MGavg | per-crop mean of MG and MGs2 |
| MLavg | per-crop mean of ML and MLs2 |
| MEavg | per-crop mean of ME and MEs2 |

## All sets

| version | set | crops | mean line CER | corpus CER | exact | repeated insertions |
|---|---|---|---|---|---|---|
| start | whiteboard val | 136 | 0.2058 | 0.2335 | 0.331 | 0.000 |
| T0export | whiteboard val | 136 | 0.1616 | 0.1754 | 0.191 | 0.360 |
| MC0 | whiteboard val | 136 | 0.1033 | 0.1146 | 0.368 | 0.007 |
| MG | whiteboard val | 136 | 0.0939 | 0.1095 | 0.434 | 0.015 |
| ML | whiteboard val | 136 | 0.0984 | 0.1146 | 0.426 | 0.015 |
| ME | whiteboard val | 136 | 0.1130 | 0.1228 | 0.353 | 0.029 |
| MC0s2 | whiteboard val | 136 | 0.0948 | 0.1166 | 0.404 | 0.022 |
| MGs2 | whiteboard val | 136 | 0.0933 | 0.1142 | 0.471 | 0.000 |
| MLs2 | whiteboard val | 136 | 0.1039 | 0.1209 | 0.390 | 0.007 |
| MEs2 | whiteboard val | 136 | 0.1004 | 0.1134 | 0.463 | 0.007 |
| MV1b | whiteboard val | 136 | 0.1907 | 0.2268 | 0.301 | 0.000 |
| MV2short | whiteboard val | 136 | 0.0890 | 0.1044 | 0.463 | 0.015 |
| MV3short | whiteboard val | 136 | 0.0900 | 0.1071 | 0.485 | 0.000 |
| MV2 | whiteboard val | 136 | 0.0971 | 0.1130 | 0.412 | 0.000 |
| MV3 | whiteboard val | 136 | 0.1008 | 0.1158 | 0.426 | 0.022 |
| MC1 | whiteboard val | 136 | 0.1116 | 0.1299 | 0.397 | 0.029 |
| MP40 | whiteboard val | 136 | 0.3959 | 0.4286 | 0.029 | 0.713 |
| MP128 | whiteboard val | 136 | 0.0919 | 0.1099 | 0.493 | 0.007 |
| MT | whiteboard val | 136 | 0.4419 | 0.4403 | 0.000 | 0.493 |
| MC0avg | whiteboard val | 136 | 0.0991 | 0.1156 | 0.386 | 0.015 |
| MGavg | whiteboard val | 136 | 0.0936 | 0.1119 | 0.452 | 0.007 |
| MLavg | whiteboard val | 136 | 0.1012 | 0.1177 | 0.408 | 0.011 |
| MEavg | whiteboard val | 136 | 0.1067 | 0.1181 | 0.408 | 0.018 |
| start | CNN test | 174 | 0.5630 | 0.5710 | 0.011 | 0.000 |
| T0export | CNN test | 174 | 0.3260 | 0.3353 | 0.000 | 0.299 |
| MC0 | CNN test | 174 | 0.3002 | 0.2996 | 0.034 | 0.040 |
| MG | CNN test | 174 | 0.3104 | 0.3093 | 0.040 | 0.017 |
| ML | CNN test | 174 | 0.3185 | 0.3154 | 0.046 | 0.029 |
| ME | CNN test | 174 | 0.3192 | 0.3179 | 0.023 | 0.006 |
| MC0s2 | CNN test | 174 | 0.3117 | 0.3087 | 0.029 | 0.023 |
| MGs2 | CNN test | 174 | 0.2835 | 0.2852 | 0.040 | 0.029 |
| MLs2 | CNN test | 174 | 0.3243 | 0.3248 | 0.034 | 0.040 |
| MEs2 | CNN test | 174 | 0.3004 | 0.2982 | 0.034 | 0.098 |
| MV1b | CNN test | 174 | 0.5128 | 0.5135 | 0.000 | 0.017 |
| MV2short | CNN test | 174 | 0.3250 | 0.3195 | 0.034 | 0.034 |
| MV3short | CNN test | 174 | 0.2973 | 0.2944 | 0.046 | 0.040 |
| MV2 | CNN test | 174 | 0.2866 | 0.2861 | 0.040 | 0.052 |
| MV3 | CNN test | 174 | 0.3041 | 0.3027 | 0.046 | 0.040 |
| MC1 | CNN test | 174 | 0.3336 | 0.3353 | 0.011 | 0.034 |
| MP40 | CNN test | 174 | 0.6583 | 0.6515 | 0.000 | 0.391 |
| MP128 | CNN test | 174 | 0.3087 | 0.3093 | 0.057 | 0.069 |
| MT | CNN test | 174 | 0.7049 | 0.6968 | 0.000 | 0.529 |
| MC0avg | CNN test | 174 | 0.3060 | 0.3042 | 0.032 | 0.032 |
| MGavg | CNN test | 174 | 0.2969 | 0.2973 | 0.040 | 0.023 |
| MLavg | CNN test | 174 | 0.3214 | 0.3201 | 0.040 | 0.034 |
| MEavg | CNN test | 174 | 0.3098 | 0.3080 | 0.029 | 0.052 |
| start | long sentences (held out) | 57 | 0.0540 | 0.0592 | 0.175 | 0.000 |
| T0export | long sentences (held out) | 57 | 0.1223 | 0.1300 | 0.018 | 0.316 |
| MC0 | long sentences (held out) | 57 | 0.0362 | 0.0422 | 0.526 | 0.000 |
| MG | long sentences (held out) | 57 | 0.0262 | 0.0318 | 0.421 | 0.000 |
| ML | long sentences (held out) | 57 | 0.0363 | 0.0438 | 0.614 | 0.000 |
| ME | long sentences (held out) | 57 | 0.0329 | 0.0387 | 0.579 | 0.000 |
| MC0s2 | long sentences (held out) | 57 | 0.0381 | 0.0467 | 0.614 | 0.000 |
| MGs2 | long sentences (held out) | 57 | 0.0356 | 0.0414 | 0.386 | 0.000 |
| MLs2 | long sentences (held out) | 57 | 0.0342 | 0.0403 | 0.579 | 0.000 |
| MEs2 | long sentences (held out) | 57 | 0.0316 | 0.0372 | 0.526 | 0.000 |
| MV1b | long sentences (held out) | 57 | 0.0498 | 0.0557 | 0.263 | 0.000 |
| MV2short | long sentences (held out) | 57 | 0.0396 | 0.0480 | 0.386 | 0.000 |
| MV3short | long sentences (held out) | 57 | 0.0409 | 0.0488 | 0.333 | 0.000 |
| MV2 | long sentences (held out) | 57 | 0.0331 | 0.0393 | 0.439 | 0.000 |
| MV3 | long sentences (held out) | 57 | 0.0370 | 0.0427 | 0.561 | 0.000 |
| MC1 | long sentences (held out) | 57 | 0.0849 | 0.1035 | 0.351 | 0.018 |
| MP40 | long sentences (held out) | 57 | 0.4249 | 0.4472 | 0.000 | 1.088 |
| MP128 | long sentences (held out) | 57 | 0.0430 | 0.0523 | 0.386 | 0.000 |
| MT | long sentences (held out) | 57 | 0.3343 | 0.3219 | 0.000 | 0.772 |
| MC0avg | long sentences (held out) | 57 | 0.0372 | 0.0445 | 0.570 | 0.000 |
| MGavg | long sentences (held out) | 57 | 0.0309 | 0.0366 | 0.404 | 0.000 |
| MLavg | long sentences (held out) | 57 | 0.0353 | 0.0421 | 0.596 | 0.000 |
| MEavg | long sentences (held out) | 57 | 0.0322 | 0.0380 | 0.553 | 0.000 |
| start | IAM val | 976 | 0.1871 | 0.1796 | 0.019 | 0.020 |
| T0export | IAM val | 976 | 0.3780 | 0.3704 | 0.000 | 0.575 |
| MC0 | IAM val | 976 | 0.2445 | 0.2405 | 0.005 | 0.092 |
| MG | IAM val | 976 | 0.2716 | 0.2671 | 0.002 | 0.150 |
| ML | IAM val | 976 | 0.2666 | 0.2618 | 0.001 | 0.118 |
| ME | IAM val | 976 | 0.2467 | 0.2421 | 0.003 | 0.117 |
| MC0s2 | IAM val | 976 | 0.2615 | 0.2565 | 0.002 | 0.103 |
| MGs2 | IAM val | 976 | 0.2558 | 0.2502 | 0.003 | 0.052 |
| MLs2 | IAM val | 976 | 0.2605 | 0.2540 | 0.005 | 0.043 |
| MEs2 | IAM val | 976 | 0.2554 | 0.2515 | 0.003 | 0.118 |
| MV1b | IAM val | 976 | 0.0676 | 0.0659 | 0.166 | 0.042 |
| MV2short | IAM val | 976 | 0.2351 | 0.2310 | 0.005 | 0.070 |
| MV3short | IAM val | 976 | 0.2617 | 0.2567 | 0.003 | 0.071 |
| MV2 | IAM val | 976 | 0.2399 | 0.2367 | 0.003 | 0.068 |
| MV3 | IAM val | 976 | 0.2508 | 0.2471 | 0.004 | 0.131 |
| MC1 | IAM val | 976 | 0.2498 | 0.2469 | 0.003 | 0.090 |
| MP40 | IAM val | 976 | 0.5239 | 0.5196 | 0.000 | 0.943 |
| MP128 | IAM val | 976 | 0.2463 | 0.2421 | 0.005 | 0.064 |
| MT | IAM val | 976 | 0.5548 | 0.5379 | 0.000 | 0.704 |
| MC0avg | IAM val | 976 | 0.2530 | 0.2485 | 0.004 | 0.098 |
| MGavg | IAM val | 976 | 0.2637 | 0.2587 | 0.003 | 0.101 |
| MLavg | IAM val | 976 | 0.2635 | 0.2579 | 0.003 | 0.080 |
| MEavg | IAM val | 976 | 0.2510 | 0.2468 | 0.003 | 0.117 |
| start | new writers | 160 | 0.3232 | 0.3145 | 0.031 | 0.056 |
| T0export | new writers | 160 | 0.3965 | 0.4273 | 0.013 | 0.181 |
| MC0 | new writers | 160 | 0.3112 | 0.3361 | 0.075 | 0.075 |
| MG | new writers | 160 | 0.3061 | 0.3060 | 0.056 | 0.025 |
| ML | new writers | 160 | 0.3126 | 0.3225 | 0.056 | 0.031 |
| ME | new writers | 160 | 0.3238 | 0.3241 | 0.050 | 0.050 |
| MC0s2 | new writers | 160 | 0.3353 | 0.3450 | 0.025 | 0.087 |
| MGs2 | new writers | 160 | 0.3123 | 0.3107 | 0.075 | 0.050 |
| MLs2 | new writers | 160 | 0.3228 | 0.3419 | 0.044 | 0.062 |
| MEs2 | new writers | 160 | 0.2985 | 0.2981 | 0.037 | 0.069 |
| MV1b | new writers | 160 | 0.3645 | 0.3752 | 0.037 | 0.044 |
| MV2short | new writers | 160 | 0.3292 | 0.3312 | 0.062 | 0.075 |
| MV3short | new writers | 160 | 0.3403 | 0.3527 | 0.069 | 0.062 |
| MV2 | new writers | 160 | 0.2955 | 0.3135 | 0.113 | 0.056 |
| MV3 | new writers | 160 | 0.3246 | 0.3389 | 0.056 | 0.050 |
| MC1 | new writers | 160 | 0.4781 | 0.4832 | 0.025 | 0.050 |
| MP40 | new writers | 160 | 0.6733 | 0.6644 | 0.000 | 0.487 |
| MP128 | new writers | 160 | 0.3264 | 0.3337 | 0.062 | 0.056 |
| MT | new writers | 160 | 0.7123 | 0.6367 | 0.000 | 0.375 |
| MC0avg | new writers | 160 | 0.3232 | 0.3406 | 0.050 | 0.081 |
| MGavg | new writers | 160 | 0.3092 | 0.3083 | 0.066 | 0.037 |
| MLavg | new writers | 160 | 0.3177 | 0.3322 | 0.050 | 0.047 |
| MEavg | new writers | 160 | 0.3112 | 0.3111 | 0.044 | 0.059 |
| start | held-out long lines only (over 40 chars) | 41 | 0.0615 | 0.0620 | 0.098 | 0.000 |
| T0export | held-out long lines only (over 40 chars) | 41 | 0.1344 | 0.1347 | 0.000 | 0.146 |
| MC0 | held-out long lines only (over 40 chars) | 41 | 0.0461 | 0.0461 | 0.463 | 0.000 |
| MG | held-out long lines only (over 40 chars) | 41 | 0.0356 | 0.0356 | 0.317 | 0.000 |
| ML | held-out long lines only (over 40 chars) | 41 | 0.0488 | 0.0488 | 0.512 | 0.000 |
| ME | held-out long lines only (over 40 chars) | 41 | 0.0423 | 0.0425 | 0.512 | 0.000 |
| MC0s2 | held-out long lines only (over 40 chars) | 41 | 0.0522 | 0.0524 | 0.488 | 0.000 |
| MGs2 | held-out long lines only (over 40 chars) | 41 | 0.0453 | 0.0452 | 0.317 | 0.000 |
| MLs2 | held-out long lines only (over 40 chars) | 41 | 0.0442 | 0.0443 | 0.512 | 0.000 |
| MEs2 | held-out long lines only (over 40 chars) | 41 | 0.0405 | 0.0407 | 0.439 | 0.000 |
| MV1b | held-out long lines only (over 40 chars) | 41 | 0.0592 | 0.0596 | 0.171 | 0.000 |
| MV2short | held-out long lines only (over 40 chars) | 41 | 0.0533 | 0.0536 | 0.195 | 0.000 |
| MV3short | held-out long lines only (over 40 chars) | 41 | 0.0535 | 0.0539 | 0.146 | 0.000 |
| MV2 | held-out long lines only (over 40 chars) | 41 | 0.0435 | 0.0434 | 0.293 | 0.000 |
| MV3 | held-out long lines only (over 40 chars) | 41 | 0.0464 | 0.0464 | 0.537 | 0.000 |
| MC1 | held-out long lines only (over 40 chars) | 41 | 0.1154 | 0.1159 | 0.171 | 0.000 |
| MP40 | held-out long lines only (over 40 chars) | 41 | 0.4627 | 0.4629 | 0.000 | 0.707 |
| MP128 | held-out long lines only (over 40 chars) | 41 | 0.0581 | 0.0584 | 0.195 | 0.000 |
| MT | held-out long lines only (over 40 chars) | 41 | 0.3130 | 0.3135 | 0.000 | 0.561 |
| MC0avg | held-out long lines only (over 40 chars) | 41 | 0.0491 | 0.0493 | 0.476 | 0.000 |
| MGavg | held-out long lines only (over 40 chars) | 41 | 0.0404 | 0.0404 | 0.317 | 0.000 |
| MLavg | held-out long lines only (over 40 chars) | 41 | 0.0465 | 0.0466 | 0.512 | 0.000 |
| MEavg | held-out long lines only (over 40 chars) | 41 | 0.0414 | 0.0416 | 0.476 | 0.000 |

## Comparisons on whiteboard val (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| MGavg vs MC0avg | guidance off vs C0 (mean of 2 seeds) | -0.0055 | -0.0116 .. +0.0008 | 42 / 27 | 26 | no clear difference |
| MLavg vs MC0avg | six-group backbone LR vs C0 (mean of 2 seeds) | +0.0021 | -0.0030 .. +0.0091 | 31 / 33 | 26 | no clear difference |
| MEavg vs MC0avg | extra SE in blocks6.3 vs C0 (mean of 2 seeds) | +0.0076 | +0.0005 .. +0.0166 | 37 / 39 | 26 | worse |
| MG vs MC0 | guidance off, seed 1 | -0.0094 | -0.0185 .. +0.0035 | 37 / 19 | 26 | no clear difference |
| MGs2 vs MC0s2 | guidance off, seed 2 | -0.0016 | -0.0126 .. +0.0046 | 30 / 24 | 26 | no clear difference |
| ML vs MC0 | six-group LR, seed 1 | -0.0049 | -0.0135 .. +0.0070 | 23 / 19 | 26 | no clear difference |
| MLs2 vs MC0s2 | six-group LR, seed 2 | +0.0091 | -0.0004 .. +0.0177 | 23 / 30 | 26 | no clear difference |
| ME vs MC0 | extra SE, seed 1 | +0.0097 | -0.0009 .. +0.0230 | 29 / 35 | 26 | no clear difference |
| MEs2 vs MC0s2 | extra SE, seed 2 | +0.0055 | -0.0047 .. +0.0156 | 29 / 32 | 26 | no clear difference |
| MC0s2 vs MC0 | seed noise: control recipe, different seed | -0.0085 | -0.0167 .. +0.0036 | 26 / 26 | 26 | no clear difference |
| MV1b vs start | IAM-only adapted fine-tune vs no fine-tune | -0.0151 | -0.0359 .. +0.0060 | 42 / 34 | 26 | no clear difference |
| MV2 vs MV1b | our whiteboard data vs IAM (replacement, not addition) | -0.0936 | -0.1401 .. -0.0445 | 74 / 9 | 26 | better |
| MV2 vs MV2short | C vs A: adding long sentences, without our augmented copies | +0.0081 | -0.0020 .. +0.0161 | 19 / 31 | 26 | no clear difference |
| MV3 vs MV3short | D vs B: adding long sentences, with our augmented copies | +0.0109 | -0.0044 .. +0.0207 | 21 / 30 | 26 | no clear difference |
| MV3short vs MV2short | B vs A: our augmented copies, short lines only | +0.0009 | -0.0088 .. +0.0124 | 22 / 23 | 26 | no clear difference |
| MV3 vs MV2 | D vs C: our augmented copies, short + long lines | +0.0038 | -0.0055 .. +0.0106 | 16 / 25 | 26 | no clear difference |
| MV3 vs MC1 | adapted recipe vs original recipe (several changes at once) | -0.0107 | -0.0338 .. +0.0027 | 34 / 19 | 26 | no clear difference |
| MP128 vs MP40 | training pooling ONLY (mobile; same lines, input, weights, budget) | -0.3041 | -0.3285 .. -0.2701 | 129 / 2 | 26 | better |
| MP128 vs MT | our adapted recipe vs the teammate's recipe (same lines; several changes at once) | -0.3500 | -0.3852 .. -0.3084 | 132 / 1 | 26 | better |
| MT vs MC1 | teammate's recipe vs PaddleOCR's original recipe (same lines) | +0.3303 | +0.2871 .. +0.3652 | 4 / 131 | 26 | worse |
| MC0avg vs start | control vs its starting weights (extra training on our data) | -0.1067 | -0.1569 .. -0.0554 | 77 / 21 | 26 | better |
| MC0avg vs T0export | control vs the teammate's unchanged export | -0.0625 | -0.0825 .. -0.0454 | 81 / 28 | 26 | better |

## Comparisons on CNN test (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| MGavg vs MC0avg | guidance off vs C0 (mean of 2 seeds) | -0.0090 | -0.0197 .. +0.0013 | 78 / 62 | 38 | no clear difference |
| MLavg vs MC0avg | six-group backbone LR vs C0 (mean of 2 seeds) | +0.0154 | +0.0073 .. +0.0249 | 53 / 85 | 38 | worse |
| MEavg vs MC0avg | extra SE in blocks6.3 vs C0 (mean of 2 seeds) | +0.0038 | -0.0039 .. +0.0117 | 63 / 78 | 38 | no clear difference |
| MG vs MC0 | guidance off, seed 1 | +0.0101 | -0.0015 .. +0.0214 | 54 / 79 | 38 | no clear difference |
| MGs2 vs MC0s2 | guidance off, seed 2 | -0.0282 | -0.0416 .. -0.0152 | 88 / 44 | 38 | better |
| ML vs MC0 | six-group LR, seed 1 | +0.0183 | +0.0060 .. +0.0309 | 49 / 83 | 38 | worse |
| MLs2 vs MC0s2 | six-group LR, seed 2 | +0.0126 | +0.0020 .. +0.0241 | 52 / 70 | 38 | worse |
| ME vs MC0 | extra SE, seed 1 | +0.0189 | +0.0085 .. +0.0295 | 44 / 80 | 38 | worse |
| MEs2 vs MC0s2 | extra SE, seed 2 | -0.0113 | -0.0214 .. -0.0008 | 72 / 57 | 38 | better |
| MC0s2 vs MC0 | seed noise: control recipe, different seed | +0.0114 | +0.0033 .. +0.0205 | 58 / 74 | 38 | worse |
| MV1b vs start | IAM-only adapted fine-tune vs no fine-tune | -0.0501 | -0.0796 .. -0.0223 | 105 / 49 | 38 | better |
| MV2 vs MV1b | our whiteboard data vs IAM (replacement, not addition) | -0.2262 | -0.2516 .. -0.1995 | 166 / 2 | 38 | better |
| MV2 vs MV2short | C vs A: adding long sentences, without our augmented copies | -0.0384 | -0.0483 .. -0.0286 | 89 / 35 | 38 | better |
| MV3 vs MV3short | D vs B: adding long sentences, with our augmented copies | +0.0068 | -0.0040 .. +0.0177 | 60 / 68 | 38 | no clear difference |
| MV3short vs MV2short | B vs A: our augmented copies, short lines only | -0.0277 | -0.0393 .. -0.0166 | 80 / 38 | 38 | better |
| MV3 vs MV2 | D vs C: our augmented copies, short + long lines | +0.0175 | +0.0073 .. +0.0266 | 36 / 70 | 38 | worse |
| MV3 vs MC1 | adapted recipe vs original recipe (several changes at once) | -0.0295 | -0.0431 .. -0.0164 | 92 / 50 | 38 | better |
| MP128 vs MP40 | training pooling ONLY (mobile; same lines, input, weights, budget) | -0.3496 | -0.3936 .. -0.3025 | 167 / 2 | 38 | better |
| MP128 vs MT | our adapted recipe vs the teammate's recipe (same lines; several changes at once) | -0.3962 | -0.4335 .. -0.3564 | 173 / 0 | 38 | better |
| MT vs MC1 | teammate's recipe vs PaddleOCR's original recipe (same lines) | +0.3713 | +0.3362 .. +0.4054 | 0 / 173 | 38 | worse |
| MC0avg vs start | control vs its starting weights (extra training on our data) | -0.2570 | -0.3032 .. -0.2101 | 160 / 9 | 38 | better |
| MC0avg vs T0export | control vs the teammate's unchanged export | -0.0200 | -0.0647 .. +0.0257 | 86 / 74 | 38 | no clear difference |

## Comparisons on long sentences (held out) (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| MGavg vs MC0avg | guidance off vs C0 (mean of 2 seeds) | -0.0062 | -0.0120 .. -0.0011 | 21 / 7 | 7 | better |
| MLavg vs MC0avg | six-group backbone LR vs C0 (mean of 2 seeds) | -0.0019 | -0.0049 .. +0.0005 | 12 / 8 | 7 | no clear difference |
| MEavg vs MC0avg | extra SE in blocks6.3 vs C0 (mean of 2 seeds) | -0.0049 | -0.0116 .. +0.0002 | 12 / 9 | 7 | no clear difference |
| MG vs MC0 | guidance off, seed 1 | -0.0100 | -0.0148 .. -0.0053 | 16 / 2 | 7 | better |
| MGs2 vs MC0s2 | guidance off, seed 2 | -0.0025 | -0.0131 .. +0.0095 | 13 / 10 | 7 | no clear difference |
| ML vs MC0 | six-group LR, seed 1 | +0.0001 | -0.0051 .. +0.0037 | 7 / 7 | 7 | no clear difference |
| MLs2 vs MC0s2 | six-group LR, seed 2 | -0.0039 | -0.0109 .. +0.0030 | 11 / 7 | 7 | no clear difference |
| ME vs MC0 | extra SE, seed 1 | -0.0033 | -0.0118 .. +0.0022 | 7 / 5 | 7 | no clear difference |
| MEs2 vs MC0s2 | extra SE, seed 2 | -0.0066 | -0.0135 .. +0.0007 | 11 / 10 | 7 | no clear difference |
| MC0s2 vs MC0 | seed noise: control recipe, different seed | +0.0020 | -0.0065 .. +0.0098 | 9 / 11 | 7 | no clear difference |
| MV1b vs start | IAM-only adapted fine-tune vs no fine-tune | -0.0042 | -0.0190 .. +0.0070 | 25 / 14 | 7 | no clear difference |
| MV2 vs MV1b | our whiteboard data vs IAM (replacement, not addition) | -0.0166 | -0.0260 .. -0.0008 | 35 / 6 | 7 | better |
| MV2 vs MV2short | C vs A: adding long sentences, without our augmented copies | -0.0065 | -0.0186 .. +0.0092 | 25 / 4 | 7 | no clear difference |
| MV3 vs MV3short | D vs B: adding long sentences, with our augmented copies | -0.0039 | -0.0132 .. +0.0066 | 22 / 6 | 7 | no clear difference |
| MV3short vs MV2short | B vs A: our augmented copies, short lines only | +0.0013 | -0.0047 .. +0.0076 | 14 / 12 | 7 | no clear difference |
| MV3 vs MV2 | D vs C: our augmented copies, short + long lines | +0.0039 | -0.0002 .. +0.0085 | 6 / 9 | 7 | no clear difference |
| MV3 vs MC1 | adapted recipe vs original recipe (several changes at once) | -0.0479 | -0.0763 .. -0.0171 | 34 / 4 | 7 | better |
| MP128 vs MP40 | training pooling ONLY (mobile; same lines, input, weights, budget) | -0.3818 | -0.4299 .. -0.3314 | 57 / 0 | 7 | better |
| MP128 vs MT | our adapted recipe vs the teammate's recipe (same lines; several changes at once) | -0.2913 | -0.3467 .. -0.2405 | 57 / 0 | 7 | better |
| MT vs MC1 | teammate's recipe vs PaddleOCR's original recipe (same lines) | +0.2494 | +0.1872 .. +0.3194 | 3 / 54 | 7 | worse |
| MC0avg vs start | control vs its starting weights (extra training on our data) | -0.0168 | -0.0327 .. -0.0002 | 38 / 4 | 7 | better |
| MC0avg vs T0export | control vs the teammate's unchanged export | -0.0852 | -0.1016 .. -0.0720 | 54 / 2 | 7 | better |

## Comparisons on held-out long lines only (over 40 chars) (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| MGavg vs MC0avg | guidance off vs C0 (mean of 2 seeds) | -0.0087 | -0.0151 .. -0.0024 | 19 / 5 | 5 | better |
| MLavg vs MC0avg | six-group backbone LR vs C0 (mean of 2 seeds) | -0.0026 | -0.0068 .. +0.0003 | 10 / 6 | 5 | no clear difference |
| MEavg vs MC0avg | extra SE in blocks6.3 vs C0 (mean of 2 seeds) | -0.0077 | -0.0163 .. -0.0022 | 12 / 7 | 5 | better |
| MG vs MC0 | guidance off, seed 1 | -0.0105 | -0.0141 .. -0.0074 | 12 / 2 | 5 | better |
| MGs2 vs MC0s2 | guidance off, seed 2 | -0.0069 | -0.0182 .. +0.0079 | 13 / 6 | 5 | no clear difference |
| ML vs MC0 | six-group LR, seed 1 | +0.0027 | +0.0004 .. +0.0049 | 4 / 7 | 5 | worse |
| MLs2 vs MC0s2 | six-group LR, seed 2 | -0.0080 | -0.0150 .. -0.0022 | 11 / 4 | 5 | better |
| ME vs MC0 | extra SE, seed 1 | -0.0038 | -0.0158 .. +0.0032 | 5 / 4 | 5 | no clear difference |
| MEs2 vs MC0s2 | extra SE, seed 2 | -0.0116 | -0.0175 .. -0.0071 | 11 / 7 | 5 | better |
| MC0s2 vs MC0 | seed noise: control recipe, different seed | +0.0061 | -0.0010 .. +0.0147 | 5 / 11 | 5 | no clear difference |
| MV1b vs start | IAM-only adapted fine-tune vs no fine-tune | -0.0022 | -0.0086 .. +0.0036 | 20 / 11 | 5 | no clear difference |
| MV2 vs MV1b | our whiteboard data vs IAM (replacement, not addition) | -0.0157 | -0.0286 .. +0.0077 | 30 / 6 | 5 | no clear difference |
| MV2 vs MV2short | C vs A: adding long sentences, without our augmented copies | -0.0098 | -0.0221 .. +0.0128 | 24 / 2 | 5 | no clear difference |
| MV3 vs MV3short | D vs B: adding long sentences, with our augmented copies | -0.0071 | -0.0181 .. +0.0085 | 21 / 3 | 5 | no clear difference |
| MV3short vs MV2short | B vs A: our augmented copies, short lines only | +0.0002 | -0.0074 .. +0.0080 | 14 / 10 | 5 | no clear difference |
| MV3 vs MV2 | D vs C: our augmented copies, short + long lines | +0.0029 | -0.0026 .. +0.0079 | 6 / 6 | 5 | no clear difference |
| MV3 vs MC1 | adapted recipe vs original recipe (several changes at once) | -0.0690 | -0.0899 .. -0.0470 | 33 / 0 | 5 | better |
| MP128 vs MP40 | training pooling ONLY (mobile; same lines, input, weights, budget) | -0.4046 | -0.4529 .. -0.3466 | 41 / 0 | 5 | better |
| MP128 vs MT | our adapted recipe vs the teammate's recipe (same lines; several changes at once) | -0.2549 | -0.2969 .. -0.2164 | 41 / 0 | 5 | better |
| MT vs MC1 | teammate's recipe vs PaddleOCR's original recipe (same lines) | +0.1976 | +0.1588 .. +0.2305 | 3 / 38 | 5 | worse |
| MC0avg vs start | control vs its starting weights (extra training on our data) | -0.0123 | -0.0240 .. +0.0081 | 31 / 4 | 5 | no clear difference |
| MC0avg vs T0export | control vs the teammate's unchanged export | -0.0853 | -0.1060 .. -0.0690 | 39 / 2 | 5 | better |

## Comparisons on new writers (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| MGavg vs MC0avg | guidance off vs C0 (mean of 2 seeds) | -0.0141 | -0.0293 .. +0.0053 | 90 / 48 | 40 | no clear difference |
| MLavg vs MC0avg | six-group backbone LR vs C0 (mean of 2 seeds) | -0.0056 | -0.0163 .. +0.0067 | 76 / 52 | 40 | no clear difference |
| MEavg vs MC0avg | extra SE in blocks6.3 vs C0 (mean of 2 seeds) | -0.0121 | -0.0272 .. +0.0072 | 80 / 56 | 40 | no clear difference |
| MG vs MC0 | guidance off, seed 1 | -0.0051 | -0.0254 .. +0.0201 | 72 / 52 | 40 | no clear difference |
| MGs2 vs MC0s2 | guidance off, seed 2 | -0.0230 | -0.0421 .. +0.0011 | 87 / 35 | 40 | no clear difference |
| ML vs MC0 | six-group LR, seed 1 | +0.0014 | -0.0110 .. +0.0206 | 63 / 49 | 40 | no clear difference |
| MLs2 vs MC0s2 | six-group LR, seed 2 | -0.0125 | -0.0345 .. +0.0069 | 64 / 56 | 40 | no clear difference |
| ME vs MC0 | extra SE, seed 1 | +0.0126 | -0.0089 .. +0.0392 | 61 / 67 | 40 | no clear difference |
| MEs2 vs MC0s2 | extra SE, seed 2 | -0.0368 | -0.0573 .. -0.0135 | 91 / 37 | 40 | better |
| MC0s2 vs MC0 | seed noise: control recipe, different seed | +0.0241 | +0.0048 .. +0.0494 | 47 / 82 | 40 | worse |
| MV1b vs start | IAM-only adapted fine-tune vs no fine-tune | +0.0413 | +0.0133 .. +0.0631 | 39 / 100 | 40 | worse |
| MV2 vs MV1b | our whiteboard data vs IAM (replacement, not addition) | -0.0690 | -0.0942 .. -0.0464 | 103 / 35 | 40 | better |
| MV2 vs MV2short | C vs A: adding long sentences, without our augmented copies | -0.0336 | -0.0582 .. -0.0145 | 84 / 44 | 40 | better |
| MV3 vs MV3short | D vs B: adding long sentences, with our augmented copies | -0.0157 | -0.0317 .. +0.0003 | 65 / 58 | 40 | no clear difference |
| MV3short vs MV2short | B vs A: our augmented copies, short lines only | +0.0111 | -0.0085 .. +0.0271 | 44 / 75 | 40 | no clear difference |
| MV3 vs MV2 | D vs C: our augmented copies, short + long lines | +0.0290 | +0.0130 .. +0.0454 | 38 / 93 | 40 | worse |
| MV3 vs MC1 | adapted recipe vs original recipe (several changes at once) | -0.1535 | -0.1747 .. -0.1332 | 130 / 11 | 40 | better |
| MP128 vs MP40 | training pooling ONLY (mobile; same lines, input, weights, budget) | -0.3470 | -0.4105 .. -0.2845 | 140 / 15 | 40 | better |
| MP128 vs MT | our adapted recipe vs the teammate's recipe (same lines; several changes at once) | -0.3859 | -0.4559 .. -0.3244 | 141 / 15 | 40 | better |
| MT vs MC1 | teammate's recipe vs PaddleOCR's original recipe (same lines) | +0.2342 | +0.1705 .. +0.3116 | 27 / 123 | 40 | worse |
| MC0avg vs start | control vs its starting weights (extra training on our data) | +0.0001 | -0.0349 .. +0.0224 | 58 / 88 | 40 | no clear difference |
| MC0avg vs T0export | control vs the teammate's unchanged export | -0.0733 | -0.1078 .. -0.0331 | 102 / 53 | 40 | better |

## Comparisons on IAM val (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| MGavg vs MC0avg | guidance off vs C0 (mean of 2 seeds) | +0.0107 | +0.0078 .. +0.0134 | 344 / 519 | 976 | worse |
| MLavg vs MC0avg | six-group backbone LR vs C0 (mean of 2 seeds) | +0.0106 | +0.0081 .. +0.0131 | 315 / 513 | 976 | worse |
| MEavg vs MC0avg | extra SE in blocks6.3 vs C0 (mean of 2 seeds) | -0.0019 | -0.0044 .. +0.0005 | 433 / 404 | 976 | no clear difference |
| MG vs MC0 | guidance off, seed 1 | +0.0272 | +0.0231 .. +0.0309 | 236 / 568 | 976 | worse |
| MGs2 vs MC0s2 | guidance off, seed 2 | -0.0057 | -0.0094 .. -0.0019 | 438 / 365 | 976 | better |
| ML vs MC0 | six-group LR, seed 1 | +0.0221 | +0.0187 .. +0.0256 | 246 / 537 | 976 | worse |
| MLs2 vs MC0s2 | six-group LR, seed 2 | -0.0010 | -0.0046 .. +0.0028 | 404 / 380 | 976 | no clear difference |
| ME vs MC0 | extra SE, seed 1 | +0.0022 | -0.0015 .. +0.0058 | 390 / 399 | 976 | no clear difference |
| MEs2 vs MC0s2 | extra SE, seed 2 | -0.0061 | -0.0095 .. -0.0027 | 424 / 355 | 976 | better |
| MC0s2 vs MC0 | seed noise: control recipe, different seed | +0.0170 | +0.0132 .. +0.0208 | 307 / 477 | 976 | worse |
| MV1b vs start | IAM-only adapted fine-tune vs no fine-tune | -0.1195 | -0.1256 .. -0.1135 | 898 / 30 | 976 | better |
| MV2 vs MV1b | our whiteboard data vs IAM (replacement, not addition) | +0.1723 | +0.1661 .. +0.1784 | 7 / 952 | 976 | worse |
| MV2 vs MV2short | C vs A: adding long sentences, without our augmented copies | +0.0048 | +0.0010 .. +0.0087 | 352 / 437 | 976 | worse |
| MV3 vs MV3short | D vs B: adding long sentences, with our augmented copies | -0.0109 | -0.0152 .. -0.0066 | 442 / 357 | 976 | better |
| MV3short vs MV2short | B vs A: our augmented copies, short lines only | +0.0267 | +0.0219 .. +0.0312 | 269 / 550 | 976 | worse |
| MV3 vs MV2 | D vs C: our augmented copies, short + long lines | +0.0109 | +0.0074 .. +0.0147 | 336 / 444 | 976 | worse |
| MV3 vs MC1 | adapted recipe vs original recipe (several changes at once) | +0.0010 | -0.0036 .. +0.0052 | 419 / 401 | 976 | no clear difference |
| MP128 vs MP40 | training pooling ONLY (mobile; same lines, input, weights, budget) | -0.2777 | -0.2866 .. -0.2685 | 965 / 2 | 976 | better |
| MP128 vs MT | our adapted recipe vs the teammate's recipe (same lines; several changes at once) | -0.3085 | -0.3181 .. -0.2986 | 969 / 3 | 976 | better |
| MT vs MC1 | teammate's recipe vs PaddleOCR's original recipe (same lines) | +0.3049 | +0.2945 .. +0.3152 | 8 / 964 | 976 | worse |
| MC0avg vs start | control vs its starting weights (extra training on our data) | +0.0659 | +0.0613 .. +0.0702 | 117 / 811 | 976 | worse |
| MC0avg vs T0export | control vs the teammate's unchanged export | -0.1250 | -0.1305 .. -0.1195 | 911 / 54 | 976 | better |

## New writers, by writer (mean line CER)

Teammates' handwriting that no S0-S3 run trained on (the V-runs of versions_2x2 saw 64 teammates' crops of unknown writer). Two writers only: treat as a first look at generalisation, not a population estimate.

| model | writer_DS (70) | writer_JH (90) |
|---|---|---|
| start | 0.2970 | 0.3436 |
| T0export | 0.4106 | 0.3855 |
| MC0 | 0.3344 | 0.2932 |
| MG | 0.3010 | 0.3101 |
| ML | 0.3220 | 0.3053 |
| ME | 0.3138 | 0.3316 |
| MC0s2 | 0.3373 | 0.3337 |
| MGs2 | 0.3022 | 0.3201 |
| MLs2 | 0.3240 | 0.3218 |
| MEs2 | 0.2949 | 0.3013 |
| MV1b | 0.3713 | 0.3592 |
| MV2short | 0.3367 | 0.3233 |
| MV3short | 0.3631 | 0.3225 |
| MV2 | 0.3081 | 0.2857 |
| MV3 | 0.3407 | 0.3120 |
| MC1 | 0.4852 | 0.4726 |
| MP40 | 0.7114 | 0.6438 |
| MP128 | 0.3356 | 0.3192 |
| MT | 0.6797 | 0.7376 |
| MC0avg | 0.3358 | 0.3135 |
| MGavg | 0.3016 | 0.3151 |
| MLavg | 0.3230 | 0.3136 |
| MEavg | 0.3043 | 0.3165 |

## By label length (does a version fail on long lines?)

| set | length | start | T0export | MC0 | MG | ML | ME | MC0s2 | MGs2 | MLs2 | MEs2 | MV1b | MV2short | MV3short | MV2 | MV3 | MC1 | MP40 | MP128 | MT | MC0avg | MGavg | MLavg | MEavg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| whiteboard val | 1-20 chars (86) | 0.175 | 0.146 | 0.098 | 0.085 | 0.090 | 0.109 | 0.078 | 0.079 | 0.095 | 0.097 | 0.155 | 0.081 | 0.081 | 0.090 | 0.095 | 0.098 | 0.363 | 0.078 | 0.457 | 0.088 | 0.082 | 0.092 | 0.103 |
| whiteboard val | 21-40 chars (42) | 0.273 | 0.195 | 0.114 | 0.114 | 0.120 | 0.127 | 0.126 | 0.123 | 0.123 | 0.108 | 0.262 | 0.106 | 0.106 | 0.112 | 0.113 | 0.144 | 0.429 | 0.122 | 0.426 | 0.120 | 0.119 | 0.121 | 0.117 |
| whiteboard val | over 40 chars (8) | 0.191 | 0.154 | 0.098 | 0.085 | 0.079 | 0.086 | 0.109 | 0.093 | 0.102 | 0.093 | 0.204 | 0.083 | 0.105 | 0.097 | 0.102 | 0.093 | 0.574 | 0.085 | 0.361 | 0.104 | 0.089 | 0.090 | 0.090 |
| CNN test | 1-20 chars (57) | 0.509 | 0.247 | 0.297 | 0.304 | 0.329 | 0.312 | 0.320 | 0.257 | 0.309 | 0.300 | 0.505 | 0.358 | 0.312 | 0.278 | 0.301 | 0.310 | 0.704 | 0.295 | 0.768 | 0.309 | 0.280 | 0.319 | 0.306 |
| CNN test | 21-40 chars (117) | 0.589 | 0.365 | 0.302 | 0.313 | 0.314 | 0.323 | 0.307 | 0.297 | 0.332 | 0.300 | 0.516 | 0.309 | 0.290 | 0.291 | 0.306 | 0.345 | 0.636 | 0.315 | 0.674 | 0.305 | 0.305 | 0.323 | 0.311 |
| long sentences (held out) | 21-40 chars (16) | 0.035 | 0.091 | 0.011 | 0.002 | 0.004 | 0.009 | 0.002 | 0.011 | 0.009 | 0.009 | 0.025 | 0.004 | 0.009 | 0.006 | 0.013 | 0.007 | 0.328 | 0.004 | 0.389 | 0.006 | 0.006 | 0.006 | 0.009 |
| long sentences (held out) | over 40 chars (41) | 0.061 | 0.134 | 0.046 | 0.036 | 0.049 | 0.042 | 0.052 | 0.045 | 0.044 | 0.041 | 0.059 | 0.053 | 0.054 | 0.044 | 0.046 | 0.115 | 0.463 | 0.058 | 0.313 | 0.049 | 0.040 | 0.047 | 0.041 |
| IAM val | 1-20 chars (36) | 0.438 | 0.530 | 0.396 | 0.414 | 0.426 | 0.402 | 0.406 | 0.421 | 0.456 | 0.391 | 0.117 | 0.390 | 0.402 | 0.351 | 0.378 | 0.380 | 0.649 | 0.379 | 0.739 | 0.401 | 0.417 | 0.441 | 0.397 |
| IAM val | 21-40 chars (308) | 0.178 | 0.390 | 0.233 | 0.265 | 0.257 | 0.237 | 0.257 | 0.250 | 0.257 | 0.243 | 0.069 | 0.226 | 0.254 | 0.232 | 0.240 | 0.237 | 0.511 | 0.239 | 0.614 | 0.245 | 0.258 | 0.257 | 0.240 |
| IAM val | over 40 chars (632) | 0.177 | 0.364 | 0.241 | 0.267 | 0.262 | 0.243 | 0.255 | 0.249 | 0.251 | 0.254 | 0.064 | 0.231 | 0.257 | 0.237 | 0.249 | 0.249 | 0.523 | 0.242 | 0.515 | 0.248 | 0.258 | 0.257 | 0.248 |
| new writers | 1-20 chars (35) | 0.457 | 0.313 | 0.312 | 0.400 | 0.370 | 0.402 | 0.387 | 0.413 | 0.347 | 0.384 | 0.409 | 0.413 | 0.385 | 0.331 | 0.364 | 0.529 | 0.745 | 0.394 | 1.050 | 0.349 | 0.406 | 0.358 | 0.393 |
| new writers | 21-40 chars (56) | 0.206 | 0.379 | 0.210 | 0.198 | 0.208 | 0.238 | 0.244 | 0.208 | 0.222 | 0.198 | 0.280 | 0.229 | 0.250 | 0.176 | 0.226 | 0.408 | 0.617 | 0.223 | 0.615 | 0.227 | 0.203 | 0.215 | 0.218 |
| new writers | over 40 chars (69) | 0.350 | 0.453 | 0.393 | 0.346 | 0.368 | 0.354 | 0.383 | 0.346 | 0.392 | 0.337 | 0.410 | 0.368 | 0.391 | 0.375 | 0.385 | 0.509 | 0.683 | 0.376 | 0.620 | 0.388 | 0.346 | 0.380 | 0.345 |
| held-out long lines only (over 40 chars) | over 40 chars (41) | 0.061 | 0.134 | 0.046 | 0.036 | 0.049 | 0.042 | 0.052 | 0.045 | 0.044 | 0.041 | 0.059 | 0.053 | 0.054 | 0.044 | 0.046 | 0.115 | 0.463 | 0.058 | 0.313 | 0.049 | 0.040 | 0.047 | 0.041 |

## Training lines per run (only hard limits are applied)

| run | lines | left out (reason) | kept although the text covers fewer positions than it needs |
|---|---|---|---|
| MC0 | 1359 | {'label longer than the limit': 3} | 28 |
| MC0s2 | 1359 | {'label longer than the limit': 3} | 28 |
| MC1 | 4269 | {'label longer than the limit': 12, 'needs more than 40 positions': 1231} | 509 |
| ME | 1359 | {'label longer than the limit': 3} | 28 |
| MEs2 | 1359 | {'label longer than the limit': 3} | 28 |
| MG | 1359 | {'label longer than the limit': 3} | 28 |
| MGs2 | 1359 | {'label longer than the limit': 3} | 28 |
| ML | 1359 | {'label longer than the limit': 3} | 28 |
| MLs2 | 1359 | {'label longer than the limit': 3} | 28 |
| MP128 | 4269 | {'label longer than the limit': 12, 'needs more than 40 positions': 1231} | 55 |
| MP40 | 4269 | {'label longer than the limit': 12, 'needs more than 40 positions': 1231} | 55 |
| MT | 4269 | {'label longer than the limit': 12, 'needs more than 40 positions': 1231} | 55 |
| MV1b | 6482 | none | 3 |
| MV2 | 1423 | {'label longer than the limit': 3} | 28 |
| MV2short | 1118 | {'longer than 40 characters (short-only run)': 308} | 9 |
| MV3 | 5500 | {'label longer than the limit': 12} | 142 |
| MV3short | 4289 | {'longer than 40 characters (short-only run)': 1223} | 55 |

## Training runs

| run | lines | epochs | updates | minutes | best validation |
|---|---|---|---|---|---|
| MC0 | 1359 | 71 | 6035 | 35.7 | norm_edit_dis: 0.8929679842556788, is_float16: False, acc: 0.374999972 |
| MG | 1359 | 71 | 6035 | 35.4 | norm_edit_dis: 0.9005289786829141, is_float16: False, acc: 0.455882319 |
| ML | 1359 | 71 | 6035 | 35.5 | norm_edit_dis: 0.8973390673262962, is_float16: False, acc: 0.426470556 |
| ME | 1359 | 71 | 6035 | 35.3 | norm_edit_dis: 0.8925431190464694, is_float16: False, acc: 0.448529378 |
| MC0s2 | 1359 | 71 | 6035 | 35.4 | norm_edit_dis: 0.8960785998453137, is_float16: False, acc: 0.426470556 |
| MGs2 | 1359 | 71 | 6035 | 35.4 | norm_edit_dis: 0.8990804442294629, is_float16: False, acc: 0.463235260 |
| MLs2 | 1359 | 71 | 6035 | 35.5 | norm_edit_dis: 0.8929001398819808, is_float16: False, acc: 0.411764675 |
| MEs2 | 1359 | 71 | 6035 | 35.3 | norm_edit_dis: 0.9064282557514578, is_float16: False, acc: 0.499999963 |
| MV1b | 6482 | 15 | 6090 | 34.0 | norm_edit_dis: 0.8008243876789682, is_float16: False, acc: 0.301470566 |
| MV2short | 1118 | 86 | 6020 | 35.8 | norm_edit_dis: 0.9052740865492659, is_float16: False, acc: 0.492647022 |
| MV3short | 4289 | 22 | 5918 | 33.3 | norm_edit_dis: 0.9035751737225672, is_float16: False, acc: 0.499999963 |
| MV2 | 1423 | 67 | 5963 | 35.5 | norm_edit_dis: 0.8978779113316656, is_float16: False, acc: 0.426470556 |
| MV3 | 5500 | 17 | 5848 | 33.3 | norm_edit_dis: 0.8945559646871055, is_float16: False, acc: 0.441176438 |
| MC1 | 4269 | 22 | 5874 | 10.1 | norm_edit_dis: 0.8522645415389805, is_float16: False, acc: 0.397058794 |
| MP40 | 4269 | 22 | 5874 | 25.6 | norm_edit_dis: 0.6698152586534423, is_float16: False, acc: 0.029411762 |
| MP128 | 4269 | 22 | 5874 | 33.1 | norm_edit_dis: 0.9044711177840811, is_float16: False, acc: 0.514705844 |
| MT | 4269 | 16 | 6048 | 13.3 | norm_edit_dis: 0.6138282274994076, is_float16: False, acc: 0.0, fps: 2 |

Limits: one seed per run; the intervals reflect evaluation-cluster sampling, not training-seed variability. Validation shares source photos with training (different texts); the CNN test board has few distinct texts (see predictions_all.csv); the held-out long sentences are 7 sentences from one board and writer; the new-writer set is 160 crops from 2 writers (intervals resample whole source photos).

## Mobile verdicts (rules fixed before training)

### Primary criterion: whiteboard validation, mean of 2 seeds

A setting **helps** only if the seed-averaged 95% interval is below 0 AND both seeds improve on their own control (MC0 / MC0s2). Not met = 'improvement not established' (not proof of no effect); interval above 0 and both seeds worse = 'hurts'.

| setting | seed 1 vs MC0 | seed 2 vs MC0s2 | mean of seeds | 95% interval | decision |
|---|---|---|---|---|---|
| MG (guidance off) | -0.0094 | -0.0016 | -0.0055 | -0.0116 .. +0.0008 | **improvement not established** |
| ML (six-group backbone LR) | -0.0049 | +0.0091 | +0.0021 | -0.0030 .. +0.0091 | **improvement not established** |
| ME (extra SE in blocks6.3) | +0.0097 | +0.0055 | +0.0076 | +0.0005 .. +0.0166 | **hurts** |

For scale: the control recipe with a different seed (MC0s2 vs MC0) differs by -0.0085.

### 2x2 interaction: do our augmented copies help MORE when long sentences are included? (mobile)

Per crop: (MV3 - MV2) - (MV3short - MV2short) in CER. Negative = the copies help more with long sentences.

| set | interaction | 95% interval | verdict |
|---|---|---|---|
| whiteboard val | +0.0028 | -0.0126 .. +0.0121 | no clear interaction |
| CNN test | +0.0452 | +0.0302 .. +0.0599 | copies help less with long data |
| long sentences (held out) | +0.0026 | -0.0051 .. +0.0099 | no clear interaction |
| IAM val | -0.0157 | -0.0212 .. -0.0101 | copies help more with long data |
| new writers | +0.0179 | -0.0073 .. +0.0461 | no clear interaction |
| held-out long lines only (over 40 chars) | +0.0027 | -0.0088 .. +0.0123 | no clear interaction |

Caution: short + long runs also have more original training lines, so 'adding long sentences' is partly 'more data'.

### Repetition diagnostic (MP40 vs MP128)

Aligned insertions inside repeated substrings, per crop. Fewer flags with width/8 pooling are consistent with the train/read position-mismatch hypothesis but do not prove it.

| set | repeated insertions MP40 - MP128 | 95% interval | diagnostic |
|---|---|---|---|
| whiteboard val | +0.706 | +0.493 .. +0.888 | MP40 has more repetition flags |
| CNN test | +0.322 | +0.205 .. +0.455 | MP40 has more repetition flags |
| long sentences (held out) | +1.088 | +0.537 .. +1.821 | MP40 has more repetition flags |
| IAM val | +0.879 | +0.804 .. +0.952 | MP40 has more repetition flags |
| new writers | +0.431 | +0.212 .. +0.673 | MP40 has more repetition flags |
| held-out long lines only (over 40 chars) | +0.707 | +0.349 .. +1.023 | MP40 has more repetition flags |

### Reading caveats

- **Writer provenance:** the data batch (MV2short, MV3short, MV2, MV3, MC1, MP40, MP128) and MT train on the 64 teammates' crops of uncertain writer identity, exactly as the server data batch did. Their new-writer scores may be optimistic; the strategy batch (MC0/MG/ML/ME) excludes those crops.
- **T0export** (the teammate's export) trained on our whiteboard-validation and CNN-test crops, all of Jy and 101 of the 160 new-writer crops: its scores on those sets are 'seen in training', not held out.
- **MT** is a recipe comparison (several settings at once). Its budget matches the others in UPDATES (multi-height batches of 16/10/8 images), so it sees fewer images per update than the fixed-batch runs.
- Everything is mobile: 64x960 input (server runs used 64x1024), official mobile weights.
