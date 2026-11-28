# Board2Notes reader versions: results

Experiment `versions_2x2`; 6000 updates x 16 lines per training run; one seed. **Mean line CER**: per crop, case-insensitive, spaces normalised (lower is better). **Exact**: crop read perfectly, case-sensitive. **Repeated insertions**: aligned inserted characters inside repeated substrings of 1-16 characters, per crop. This is a diagnostic, not proof of a causal mechanism. Raw insertion counts and the older adjacent-double diagnostic are saved in predictions.csv.

Every training run, including those 'without our augmented copies', also gets PaddleOCR's built-in random augmentation (RecAug, applied in memory). The 2x2 therefore tests OUR augmented copies on top of it.

The held-out long sentences are new SENTENCES in the SAME handwriting on the SAME board; they do not test a new writer or a new board.

| version | what |
|---|---|
| V1 | historical Baseline_model (Oct 13), read as deployed (48 px, width follows the crop) |
| pretrained | PP-OCRv5_server_rec as downloaded, no fine-tuning (48 px) |
| V1b | IAM handwriting lines only, adapted recipe |
| V2short | A: our crops, short lines only (40 chars or fewer), no augmented copies |
| V3short | B: our crops, short lines only + our augmented copies |
| V2 | C: our crops, short + long lines, no augmented copies (replaces IAM) |
| V3 | D: our crops, short + long lines + our augmented copies |
| C1 | V3 data, PaddleOCR's original recipe (48x320, 40 positions) |
| P40 | pooling-only pair: original training pooling (40 positions), lines that fit 40 |
| P128 | pooling-only pair: our training pooling (width/8 positions), SAME lines as P40 |

## All sets

| version | set | crops | mean line CER | corpus CER | exact | repeated insertions |
|---|---|---|---|---|---|---|
| V1 | whiteboard val | 136 | 0.9059 | 0.7951 | 0.000 | 0.382 |
| pretrained | whiteboard val | 136 | 0.1548 | 0.1778 | 0.338 | 0.007 |
| V1b | whiteboard val | 136 | 0.1677 | 0.2119 | 0.353 | 0.000 |
| V2short | whiteboard val | 136 | 0.0725 | 0.0911 | 0.500 | 0.007 |
| V3short | whiteboard val | 136 | 0.0713 | 0.0899 | 0.507 | 0.000 |
| V2 | whiteboard val | 136 | 0.0731 | 0.0899 | 0.471 | 0.000 |
| V3 | whiteboard val | 136 | 0.0718 | 0.0891 | 0.507 | 0.000 |
| C1 | whiteboard val | 136 | 0.0930 | 0.1142 | 0.456 | 0.022 |
| P40 | whiteboard val | 136 | 0.5454 | 0.6593 | 0.037 | 1.610 |
| P128 | whiteboard val | 136 | 0.0644 | 0.0840 | 0.574 | 0.000 |
| V1 | CNN test | 174 | 0.8132 | 0.8096 | 0.000 | 0.006 |
| pretrained | CNN test | 174 | 0.4579 | 0.4655 | 0.006 | 0.017 |
| V1b | CNN test | 174 | 0.4976 | 0.4997 | 0.000 | 0.000 |
| V2short | CNN test | 174 | 0.2857 | 0.2858 | 0.040 | 0.023 |
| V3short | CNN test | 174 | 0.2583 | 0.2576 | 0.046 | 0.046 |
| V2 | CNN test | 174 | 0.2372 | 0.2388 | 0.069 | 0.029 |
| V3 | CNN test | 174 | 0.2427 | 0.2427 | 0.075 | 0.029 |
| C1 | CNN test | 174 | 0.2872 | 0.2872 | 0.034 | 0.034 |
| P40 | CNN test | 174 | 0.9388 | 0.9461 | 0.000 | 2.920 |
| P128 | CNN test | 174 | 0.2660 | 0.2662 | 0.040 | 0.006 |
| V1 | long sentences (held out) | 57 | 0.8494 | 0.8726 | 0.000 | 0.000 |
| pretrained | long sentences (held out) | 57 | 0.3093 | 0.3803 | 0.246 | 0.000 |
| V1b | long sentences (held out) | 57 | 0.0579 | 0.0671 | 0.175 | 0.000 |
| V2short | long sentences (held out) | 57 | 0.0581 | 0.0722 | 0.351 | 0.000 |
| V3short | long sentences (held out) | 57 | 0.0449 | 0.0549 | 0.316 | 0.000 |
| V2 | long sentences (held out) | 57 | 0.0238 | 0.0295 | 0.632 | 0.000 |
| V3 | long sentences (held out) | 57 | 0.0265 | 0.0324 | 0.579 | 0.000 |
| C1 | long sentences (held out) | 57 | 0.0881 | 0.1085 | 0.368 | 0.018 |
| P40 | long sentences (held out) | 57 | 0.4526 | 0.4350 | 0.000 | 0.719 |
| P128 | long sentences (held out) | 57 | 0.0455 | 0.0565 | 0.421 | 0.000 |
| V1 | IAM val | 976 | 0.7946 | 0.7909 | 0.000 | 0.025 |
| pretrained | IAM val | 976 | 0.1220 | 0.1208 | 0.044 | 0.028 |
| V1b | IAM val | 976 | 0.0779 | 0.0760 | 0.117 | 0.036 |
| V2short | IAM val | 976 | 0.2706 | 0.2661 | 0.002 | 0.108 |
| V3short | IAM val | 976 | 0.2108 | 0.2080 | 0.007 | 0.058 |
| V2 | IAM val | 976 | 0.2326 | 0.2287 | 0.004 | 0.083 |
| V3 | IAM val | 976 | 0.2571 | 0.2536 | 0.002 | 0.128 |
| C1 | IAM val | 976 | 0.3140 | 0.3102 | 0.000 | 0.201 |
| P40 | IAM val | 976 | 0.6098 | 0.6044 | 0.000 | 1.453 |
| P128 | IAM val | 976 | 0.2168 | 0.2131 | 0.010 | 0.069 |
| V1 | held-out long lines only (over 40 chars) | 41 | 0.8871 | 0.8880 | 0.000 | 0.000 |
| pretrained | held-out long lines only (over 40 chars) | 41 | 0.4273 | 0.4281 | 0.000 | 0.000 |
| V1b | held-out long lines only (over 40 chars) | 41 | 0.0731 | 0.0734 | 0.098 | 0.000 |
| V2short | held-out long lines only (over 40 chars) | 41 | 0.0808 | 0.0814 | 0.098 | 0.000 |
| V3short | held-out long lines only (over 40 chars) | 41 | 0.0616 | 0.0617 | 0.073 | 0.000 |
| V2 | held-out long lines only (over 40 chars) | 41 | 0.0331 | 0.0332 | 0.488 | 0.000 |
| V3 | held-out long lines only (over 40 chars) | 41 | 0.0360 | 0.0362 | 0.439 | 0.000 |
| C1 | held-out long lines only (over 40 chars) | 41 | 0.1215 | 0.1222 | 0.146 | 0.000 |
| P40 | held-out long lines only (over 40 chars) | 41 | 0.4234 | 0.4222 | 0.000 | 0.317 |
| P128 | held-out long lines only (over 40 chars) | 41 | 0.0632 | 0.0638 | 0.195 | 0.000 |

## Comparisons on CNN test (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| V1b vs pretrained | IAM-only adapted fine-tune vs no fine-tune | +0.0397 | +0.0214 .. +0.0589 | 46 / 96 | 38 | worse |
| V2 vs V1b | our whiteboard data vs IAM (replacement, not addition) | -0.2604 | -0.2909 .. -0.2281 | 168 / 1 | 38 | better |
| V2 vs V2short | C vs A: adding long sentences, without our augmented copies | -0.0485 | -0.0588 .. -0.0384 | 99 / 31 | 38 | better |
| V3 vs V3short | D vs B: adding long sentences, with our augmented copies | -0.0156 | -0.0284 .. -0.0018 | 77 / 54 | 38 | better |
| V3short vs V2short | B vs A: our augmented copies, short lines only | -0.0274 | -0.0398 .. -0.0153 | 84 / 38 | 38 | better |
| V3 vs V2 | D vs C: our augmented copies, short + long lines | +0.0055 | -0.0059 .. +0.0170 | 61 / 66 | 38 | no clear difference |
| V3 vs C1 | adapted recipe vs original recipe (several changes at once) | -0.0445 | -0.0637 .. -0.0254 | 99 / 42 | 38 | better |
| P128 vs P40 | training pooling ONLY (same lines, input, weights, budget) | -0.6727 | -0.7180 .. -0.6291 | 173 / 0 | 38 | better |
| V3 vs pretrained | best adapted version vs no fine-tune | -0.2152 | -0.2568 .. -0.1734 | 156 / 9 | 38 | better |
| V3 vs V1 | best adapted version vs historical baseline | -0.5705 | -0.6095 .. -0.5297 | 174 / 0 | 38 | better |

## Comparisons on long sentences (held out) (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| V1b vs pretrained | IAM-only adapted fine-tune vs no fine-tune | -0.2514 | -0.3794 .. -0.1051 | 43 / 7 | 7 | better |
| V2 vs V1b | our whiteboard data vs IAM (replacement, not addition) | -0.0341 | -0.0431 .. -0.0250 | 42 / 0 | 7 | better |
| V2 vs V2short | C vs A: adding long sentences, without our augmented copies | -0.0343 | -0.0591 .. -0.0111 | 33 / 2 | 7 | better |
| V3 vs V3short | D vs B: adding long sentences, with our augmented copies | -0.0184 | -0.0347 .. -0.0021 | 31 / 3 | 7 | better |
| V3short vs V2short | B vs A: our augmented copies, short lines only | -0.0132 | -0.0320 .. +0.0051 | 17 / 15 | 7 | no clear difference |
| V3 vs V2 | D vs C: our augmented copies, short + long lines | +0.0027 | -0.0023 .. +0.0089 | 6 / 11 | 7 | no clear difference |
| V3 vs C1 | adapted recipe vs original recipe (several changes at once) | -0.0616 | -0.1023 .. -0.0230 | 33 / 2 | 7 | better |
| P128 vs P40 | training pooling ONLY (same lines, input, weights, budget) | -0.4072 | -0.4868 .. -0.3396 | 57 / 0 | 7 | better |
| V3 vs pretrained | best adapted version vs no fine-tune | -0.2828 | -0.4215 .. -0.1292 | 43 / 0 | 7 | better |
| V3 vs V1 | best adapted version vs historical baseline | -0.8229 | -0.8653 .. -0.7794 | 57 / 0 | 7 | better |

## Comparisons on held-out long lines only (over 40 chars) (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| V1b vs pretrained | IAM-only adapted fine-tune vs no fine-tune | -0.3543 | -0.4313 .. -0.2772 | 41 / 0 | 5 | better |
| V2 vs V1b | our whiteboard data vs IAM (replacement, not addition) | -0.0400 | -0.0480 .. -0.0333 | 34 / 0 | 5 | better |
| V2 vs V2short | C vs A: adding long sentences, without our augmented copies | -0.0477 | -0.0705 .. -0.0205 | 33 / 2 | 5 | better |
| V3 vs V3short | D vs B: adding long sentences, with our augmented copies | -0.0256 | -0.0419 .. -0.0037 | 31 / 3 | 5 | better |
| V3short vs V2short | B vs A: our augmented copies, short lines only | -0.0192 | -0.0424 .. +0.0052 | 17 / 14 | 5 | no clear difference |
| V3 vs V2 | D vs C: our augmented copies, short + long lines | +0.0029 | -0.0039 .. +0.0122 | 6 / 10 | 5 | no clear difference |
| V3 vs C1 | adapted recipe vs original recipe (several changes at once) | -0.0855 | -0.1292 .. -0.0477 | 32 / 1 | 5 | better |
| P128 vs P40 | training pooling ONLY (same lines, input, weights, budget) | -0.3601 | -0.4165 .. -0.3128 | 41 / 0 | 5 | better |
| V3 vs pretrained | best adapted version vs no fine-tune | -0.3913 | -0.4765 .. -0.2995 | 41 / 0 | 5 | better |
| V3 vs V1 | best adapted version vs historical baseline | -0.8511 | -0.8856 .. -0.8148 | 41 / 0 | 5 | better |

## 2x2: do our augmented copies help MORE when long sentences are included? (interaction)

Per crop: (D - C) - (B - A) in CER, i.e. the gain from our augmented copies with long sentences minus the gain without them. Negative = the copies help more when long sentences are present.

| set | interaction | 95% interval | verdict |
|---|---|---|---|
| CNN test | +0.0329 | +0.0181 .. +0.0486 | copies help less with long data |
| long sentences (held out) | +0.0159 | -0.0009 .. +0.0346 | no clear interaction |
| held-out long lines only (over 40 chars) | +0.0221 | -0.0001 .. +0.0457 | no clear interaction |

Caution: short + long runs also have ~28% more original training lines, so 'adding long sentences' is partly 'more data'. Gains concentrated on the long-line test point to long sentences specifically.

## Repetition diagnostic (P40 vs P128 only)

The hypothesis is that the training/inference position mismatch contributes to repetition. Count aligned insertions inside repeated substrings, including repeated syllables. Fewer flags are consistent with this hypothesis but do not prove it. Inspect the saved predictions and CER together; alignment and the substring rule are imperfect.

| set | repeated insertions P40 - P128 | 95% interval | diagnostic |
|---|---|---|---|
| whiteboard val | +1.610 | +1.304 .. +1.935 | P40 has more repetition flags |
| CNN test | +2.914 | +2.571 .. +3.260 | P40 has more repetition flags |
| long sentences (held out) | +0.719 | +0.250 .. +1.259 | P40 has more repetition flags |
| IAM val | +1.384 | +1.281 .. +1.486 | P40 has more repetition flags |
| held-out long lines only (over 40 chars) | +0.317 | +0.125 .. +0.581 | P40 has more repetition flags |

## By label length (does a version fail on long lines?)

| set | length | V1 | pretrained | V1b | V2short | V3short | V2 | V3 | C1 | P40 | P128 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| whiteboard val | 1-20 chars (86) | 0.998 | 0.133 | 0.123 | 0.056 | 0.055 | 0.061 | 0.062 | 0.076 | 0.430 | 0.046 |
| whiteboard val | 21-40 chars (42) | 0.736 | 0.199 | 0.247 | 0.102 | 0.100 | 0.098 | 0.092 | 0.127 | 0.761 | 0.098 |
| whiteboard val | over 40 chars (8) | 0.803 | 0.152 | 0.232 | 0.094 | 0.091 | 0.069 | 0.078 | 0.100 | 0.654 | 0.085 |
| CNN test | 1-20 chars (57) | 0.851 | 0.407 | 0.478 | 0.271 | 0.251 | 0.211 | 0.229 | 0.273 | 0.883 | 0.259 |
| CNN test | 21-40 chars (117) | 0.795 | 0.483 | 0.507 | 0.293 | 0.262 | 0.250 | 0.249 | 0.294 | 0.966 | 0.269 |
| long sentences (held out) | 21-40 chars (16) | 0.753 | 0.007 | 0.019 | 0.000 | 0.002 | 0.000 | 0.002 | 0.003 | 0.528 | 0.000 |
| long sentences (held out) | over 40 chars (41) | 0.887 | 0.427 | 0.073 | 0.081 | 0.062 | 0.033 | 0.036 | 0.121 | 0.423 | 0.063 |
| IAM val | 1-20 chars (36) | 0.883 | 0.222 | 0.114 | 0.382 | 0.333 | 0.348 | 0.344 | 0.382 | 0.703 | 0.345 |
| IAM val | 21-40 chars (308) | 0.797 | 0.108 | 0.084 | 0.263 | 0.200 | 0.229 | 0.252 | 0.316 | 0.607 | 0.209 |
| IAM val | over 40 chars (632) | 0.788 | 0.123 | 0.073 | 0.268 | 0.209 | 0.228 | 0.255 | 0.309 | 0.606 | 0.213 |
| held-out long lines only (over 40 chars) | over 40 chars (41) | 0.887 | 0.427 | 0.073 | 0.081 | 0.062 | 0.033 | 0.036 | 0.121 | 0.423 | 0.063 |

## Training lines per run (only hard limits are applied)

| run | lines | left out (reason) | kept although the text covers fewer positions than it needs |
|---|---|---|---|
| C1 | 4269 | {'label longer than the limit': 12, 'needs more than 40 positions': 1231} | 509 |
| P128 | 4269 | {'label longer than the limit': 12, 'needs more than 40 positions': 1231} | 55 |
| P40 | 4269 | {'label longer than the limit': 12, 'needs more than 40 positions': 1231} | 55 |
| V1b | 6482 | none | 3 |
| V2 | 1423 | {'label longer than the limit': 3} | 28 |
| V2short | 1118 | {'longer than 40 characters (short-only run)': 308} | 9 |
| V3 | 5500 | {'label longer than the limit': 12} | 142 |
| V3short | 4289 | {'longer than 40 characters (short-only run)': 1223} | 55 |

## Training runs

| run | lines | epochs | updates | minutes | best validation |
|---|---|---|---|---|---|
| V1b | 6482 | 15 | 6090 | 22.0 | norm_edit_dis: 0.8315513495063338, is_float16: False, acc: 0.374999972 |
| V2short | 1118 | 86 | 6020 | 23.2 | norm_edit_dis: 0.9259150135151607, is_float16: False, acc: 0.499999963 |
| V3short | 4289 | 22 | 5918 | 21.6 | norm_edit_dis: 0.9228811838854927, is_float16: False, acc: 0.522058785 |
| V2 | 1423 | 67 | 5963 | 23.9 | norm_edit_dis: 0.9234434854601046, is_float16: False, acc: 0.485294081 |
| V3 | 5500 | 17 | 5848 | 21.9 | norm_edit_dis: 0.9233095767084583, is_float16: False, acc: 0.536764666 |
| C1 | 4269 | 22 | 5874 | 6.5 | norm_edit_dis: 0.8798138042416982, is_float16: False, acc: 0.448529378 |
| P40 | 4269 | 22 | 5874 | 12.9 | norm_edit_dis: 0.6032639386452736, is_float16: False, acc: 0.036764703 |
| P128 | 4269 | 22 | 5874 | 21.3 | norm_edit_dis: 0.9334905903474966, is_float16: False, acc: 0.580882310 |

Limits: one seed per run; the intervals reflect evaluation-cluster sampling, not training-seed variability. Validation shares source photos with training (different texts); the CNN test board has few distinct texts (see predictions_all.csv); the held-out long sentences are 7 sentences from one board and writer.
