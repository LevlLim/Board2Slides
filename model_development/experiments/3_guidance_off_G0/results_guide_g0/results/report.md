# Board2Notes reader versions: results

Experiment `guide_g0`; about 6000 updates x 16 lines per training run (exact counts in 'Training runs'; runs on the same training list have identical counts); one seed. **Mean line CER**: per crop, case-insensitive, spaces normalised (lower is better). **Exact**: crop read perfectly, case-sensitive. **Repeated insertions**: aligned inserted characters inside repeated substrings of 1-16 characters, per crop. This is a diagnostic, not proof of a causal mechanism. Raw insertion counts and the older adjacent-double diagnostic are saved in predictions.csv.

Every training run gets PaddleOCR's built-in random augmentation (RecAug, in memory).

The held-out long sentences are new SENTENCES in the SAME handwriting on the SAME board; they do not test a new writer or a new board.

| version | what |
|---|---|
| V1 | historical Baseline_model (Oct 13), read as deployed (48 px, width follows the crop) |
| pretrained | PP-OCRv5_server_rec as downloaded, no fine-tuning (48 px) |
| G0 | S0 with guidance off: CTC loss also trains the backbone (seed 1) |
| G0s2 | S0 with guidance off: CTC loss also trains the backbone (seed 2) |
| G0avg | per-crop mean of G0 and G0s2 |

## All sets

| version | set | crops | mean line CER | corpus CER | exact | repeated insertions |
|---|---|---|---|---|---|---|
| V1 | whiteboard val | 136 | 0.9059 | 0.7951 | 0.000 | 0.382 |
| pretrained | whiteboard val | 136 | 0.1548 | 0.1778 | 0.338 | 0.007 |
| G0 | whiteboard val | 136 | 0.0739 | 0.0926 | 0.478 | 0.000 |
| G0s2 | whiteboard val | 136 | 0.0695 | 0.0863 | 0.515 | 0.007 |
| G0avg | whiteboard val | 136 | 0.0717 | 0.0895 | 0.496 | 0.004 |
| V1 | CNN test | 174 | 0.8132 | 0.8096 | 0.000 | 0.006 |
| pretrained | CNN test | 174 | 0.4579 | 0.4655 | 0.006 | 0.017 |
| G0 | CNN test | 174 | 0.2594 | 0.2576 | 0.063 | 0.011 |
| G0s2 | CNN test | 174 | 0.2545 | 0.2568 | 0.057 | 0.011 |
| G0avg | CNN test | 174 | 0.2570 | 0.2572 | 0.060 | 0.011 |
| V1 | long sentences (held out) | 57 | 0.8494 | 0.8726 | 0.000 | 0.000 |
| pretrained | long sentences (held out) | 57 | 0.3093 | 0.3803 | 0.246 | 0.000 |
| G0 | long sentences (held out) | 57 | 0.0240 | 0.0297 | 0.614 | 0.000 |
| G0s2 | long sentences (held out) | 57 | 0.0243 | 0.0295 | 0.614 | 0.000 |
| G0avg | long sentences (held out) | 57 | 0.0241 | 0.0296 | 0.614 | 0.000 |
| V1 | IAM val | 976 | 0.7946 | 0.7909 | 0.000 | 0.025 |
| pretrained | IAM val | 976 | 0.1220 | 0.1208 | 0.044 | 0.028 |
| G0 | IAM val | 976 | 0.2936 | 0.2877 | 0.001 | 0.085 |
| G0s2 | IAM val | 976 | 0.2493 | 0.2440 | 0.006 | 0.078 |
| G0avg | IAM val | 976 | 0.2715 | 0.2658 | 0.004 | 0.081 |
| V1 | new writers | 160 | 0.8980 | 0.9117 | 0.000 | 0.062 |
| pretrained | new writers | 160 | 0.4308 | 0.4884 | 0.075 | 0.062 |
| G0 | new writers | 160 | 0.2477 | 0.2673 | 0.138 | 0.056 |
| G0s2 | new writers | 160 | 0.2078 | 0.2396 | 0.156 | 0.069 |
| G0avg | new writers | 160 | 0.2277 | 0.2534 | 0.147 | 0.062 |
| V1 | held-out long lines only (over 40 chars) | 41 | 0.8871 | 0.8880 | 0.000 | 0.000 |
| pretrained | held-out long lines only (over 40 chars) | 41 | 0.4273 | 0.4281 | 0.000 | 0.000 |
| G0 | held-out long lines only (over 40 chars) | 41 | 0.0334 | 0.0335 | 0.463 | 0.000 |
| G0s2 | held-out long lines only (over 40 chars) | 41 | 0.0328 | 0.0329 | 0.488 | 0.000 |
| G0avg | held-out long lines only (over 40 chars) | 41 | 0.0331 | 0.0332 | 0.476 | 0.000 |

## Comparisons on whiteboard val (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| G0s2 vs G0 | seed noise: guidance-off recipe, different seed | -0.0044 | -0.0113 .. +0.0020 | 23 / 11 | 26 | no clear difference |
| G0avg vs pretrained | guidance off vs no fine-tune | -0.0831 | -0.1145 .. -0.0505 | 73 / 6 | 26 | better |

## Comparisons on CNN test (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| G0s2 vs G0 | seed noise: guidance-off recipe, different seed | -0.0049 | -0.0138 .. +0.0032 | 57 / 57 | 38 | no clear difference |
| G0avg vs pretrained | guidance off vs no fine-tune | -0.2009 | -0.2356 .. -0.1676 | 156 / 11 | 38 | better |

## Comparisons on long sentences (held out) (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| G0s2 vs G0 | seed noise: guidance-off recipe, different seed | +0.0003 | -0.0030 .. +0.0032 | 5 / 6 | 7 | no clear difference |
| G0avg vs pretrained | guidance off vs no fine-tune | -0.2852 | -0.4233 .. -0.1296 | 43 / 0 | 7 | better |

## Comparisons on held-out long lines only (over 40 chars) (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| G0s2 vs G0 | seed noise: guidance-off recipe, different seed | -0.0006 | -0.0046 .. +0.0031 | 5 / 5 | 5 | no clear difference |
| G0avg vs pretrained | guidance off vs no fine-tune | -0.3943 | -0.4773 .. -0.3082 | 41 / 0 | 5 | better |

## Comparisons on new writers (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| G0s2 vs G0 | seed noise: guidance-off recipe, different seed | -0.0398 | -0.0588 .. -0.0265 | 75 / 36 | 40 | better |
| G0avg vs pretrained | guidance off vs no fine-tune | -0.2030 | -0.2374 .. -0.1632 | 137 / 12 | 40 | better |

## Comparisons on IAM val (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| G0s2 vs G0 | seed noise: guidance-off recipe, different seed | -0.0443 | -0.0498 .. -0.0388 | 596 / 229 | 976 | better |
| G0avg vs pretrained | guidance off vs no fine-tune | +0.1495 | +0.1424 .. +0.1563 | 28 / 926 | 976 | worse |

## New writers, by writer (mean line CER)

Teammates' handwriting that no S0-S3 run trained on (the V-runs of versions_2x2 saw 64 teammates' crops of unknown writer). Two writers only: treat as a first look at generalisation, not a population estimate.

| model | writer_DS (70) | writer_JH (90) |
|---|---|---|
| V1 | 0.9175 | 0.8829 |
| pretrained | 0.5116 | 0.3680 |
| G0 | 0.2759 | 0.2257 |
| G0s2 | 0.2451 | 0.1789 |
| G0avg | 0.2605 | 0.2023 |

## By label length (does a version fail on long lines?)

| set | length | V1 | pretrained | G0 | G0s2 | G0avg |
|---|---|---|---|---|---|---|
| whiteboard val | 1-20 chars (86) | 0.998 | 0.133 | 0.060 | 0.059 | 0.060 |
| whiteboard val | 21-40 chars (42) | 0.736 | 0.199 | 0.101 | 0.090 | 0.095 |
| whiteboard val | over 40 chars (8) | 0.803 | 0.152 | 0.084 | 0.072 | 0.078 |
| CNN test | 1-20 chars (57) | 0.851 | 0.407 | 0.259 | 0.224 | 0.241 |
| CNN test | 21-40 chars (117) | 0.795 | 0.483 | 0.260 | 0.270 | 0.265 |
| long sentences (held out) | 21-40 chars (16) | 0.753 | 0.007 | 0.000 | 0.003 | 0.001 |
| long sentences (held out) | over 40 chars (41) | 0.887 | 0.427 | 0.033 | 0.033 | 0.033 |
| IAM val | 1-20 chars (36) | 0.883 | 0.222 | 0.400 | 0.383 | 0.391 |
| IAM val | 21-40 chars (308) | 0.797 | 0.108 | 0.292 | 0.246 | 0.269 |
| IAM val | over 40 chars (632) | 0.788 | 0.123 | 0.288 | 0.244 | 0.266 |
| new writers | 1-20 chars (35) | 0.888 | 0.311 | 0.254 | 0.168 | 0.211 |
| new writers | 21-40 chars (56) | 0.850 | 0.358 | 0.169 | 0.135 | 0.152 |
| new writers | over 40 chars (69) | 0.942 | 0.551 | 0.308 | 0.287 | 0.298 |
| held-out long lines only (over 40 chars) | over 40 chars (41) | 0.887 | 0.427 | 0.033 | 0.033 | 0.033 |

## Training lines per run (only hard limits are applied)

| run | lines | left out (reason) | kept although the text covers fewer positions than it needs |
|---|---|---|---|
| G0 | 1359 | {'label longer than the limit': 3} | 28 |
| G0s2 | 1359 | {'label longer than the limit': 3} | 28 |

## Training runs

| run | lines | epochs | updates | minutes | best validation |
|---|---|---|---|---|---|
| G0 | 1359 | 71 | 6035 | 23.4 | norm_edit_dis: 0.9251489879592634, is_float16: False, acc: 0.492647022 |
| G0s2 | 1359 | 71 | 6035 | 23.2 | norm_edit_dis: 0.9287975233983479, is_float16: False, acc: 0.514705844 |

Limits: two seeds per setting (S0s2 vs S0 shows the size of seed-to-seed change); the intervals reflect evaluation-cluster sampling, not training-seed variability. Validation shares source photos with training (different texts); the CNN test board has few distinct texts (see predictions_all.csv); the held-out long sentences are 7 sentences from one board and writer; the new-writer set is 160 crops from 2 writers (intervals resample whole source photos).
