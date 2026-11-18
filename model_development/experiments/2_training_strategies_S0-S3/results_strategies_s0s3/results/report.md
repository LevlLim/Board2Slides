# Board2Notes reader versions: results

Experiment `strategies_s0s3`; about 6000 updates x 16 lines per training run (exact counts in 'Training runs'; runs on the same training list have identical counts); two seeds for S0-S3 (avg = per-crop mean of both). **Mean line CER**: per crop, case-insensitive, spaces normalised (lower is better). **Exact**: crop read perfectly, case-sensitive. **Repeated insertions**: aligned inserted characters inside repeated substrings of 1-16 characters, per crop. This is a diagnostic, not proof of a causal mechanism. Raw insertion counts and the older adjacent-double diagnostic are saved in predictions.csv.

Every training run gets PaddleOCR's built-in random augmentation (RecAug, in memory).

The held-out long sentences are new SENTENCES in the SAME handwriting on the SAME board; they do not test a new writer or a new board.

| version | what |
|---|---|
| V1 | historical Baseline_model (Oct 13), read as deployed (48 px, width follows the crop) |
| pretrained | PP-OCRv5_server_rec as downloaded, no fine-tuning (48 px) |
| S0 | clean baseline: V2 data minus the 64 teammates' crops (seed 1) |
| S0s2 | clean baseline: V2 data minus the 64 teammates' crops (seed 2) |
| S0avg | per-crop mean of S0 and S0s2 |
| S1 | S0 + stage-wise backbone learning rates (seed 1) |
| S1s2 | S0 + stage-wise backbone learning rates (seed 2) |
| S1avg | per-crop mean of S1 and S1s2 |
| S2 | S0 + focal CTC loss (seed 1) |
| S2s2 | S0 + focal CTC loss (seed 2) |
| S2avg | per-crop mean of S2 and S2s2 |
| S3 | S0 + stage-wise learning rates + focal CTC (seed 1) |
| S3s2 | S0 + stage-wise learning rates + focal CTC (seed 2) |
| S3avg | per-crop mean of S3 and S3s2 |

## All sets

| version | set | crops | mean line CER | corpus CER | exact | repeated insertions |
|---|---|---|---|---|---|---|
| V1 | whiteboard val | 136 | 0.9059 | 0.7951 | 0.000 | 0.382 |
| pretrained | whiteboard val | 136 | 0.1548 | 0.1778 | 0.338 | 0.007 |
| S0 | whiteboard val | 136 | 0.0751 | 0.0946 | 0.515 | 0.000 |
| S0s2 | whiteboard val | 136 | 0.0752 | 0.0911 | 0.544 | 0.022 |
| S0avg | whiteboard val | 136 | 0.0752 | 0.0928 | 0.529 | 0.011 |
| S1 | whiteboard val | 136 | 0.0685 | 0.0883 | 0.544 | 0.007 |
| S1s2 | whiteboard val | 136 | 0.0734 | 0.0895 | 0.537 | 0.007 |
| S1avg | whiteboard val | 136 | 0.0709 | 0.0889 | 0.540 | 0.007 |
| S2 | whiteboard val | 136 | 0.0711 | 0.0879 | 0.485 | 0.007 |
| S2s2 | whiteboard val | 136 | 0.0737 | 0.0903 | 0.485 | 0.007 |
| S2avg | whiteboard val | 136 | 0.0724 | 0.0891 | 0.485 | 0.007 |
| S3 | whiteboard val | 136 | 0.0726 | 0.0899 | 0.471 | 0.000 |
| S3s2 | whiteboard val | 136 | 0.0712 | 0.0879 | 0.485 | 0.007 |
| S3avg | whiteboard val | 136 | 0.0719 | 0.0889 | 0.478 | 0.004 |
| V1 | CNN test | 174 | 0.8132 | 0.8096 | 0.000 | 0.006 |
| pretrained | CNN test | 174 | 0.4579 | 0.4655 | 0.006 | 0.017 |
| S0 | CNN test | 174 | 0.2681 | 0.2700 | 0.057 | 0.034 |
| S0s2 | CNN test | 174 | 0.2418 | 0.2413 | 0.086 | 0.029 |
| S0avg | CNN test | 174 | 0.2550 | 0.2557 | 0.072 | 0.032 |
| S1 | CNN test | 174 | 0.2434 | 0.2438 | 0.057 | 0.029 |
| S1s2 | CNN test | 174 | 0.2482 | 0.2507 | 0.069 | 0.034 |
| S1avg | CNN test | 174 | 0.2458 | 0.2472 | 0.063 | 0.032 |
| S2 | CNN test | 174 | 0.2292 | 0.2330 | 0.063 | 0.023 |
| S2s2 | CNN test | 174 | 0.2682 | 0.2709 | 0.029 | 0.023 |
| S2avg | CNN test | 174 | 0.2487 | 0.2519 | 0.046 | 0.023 |
| S3 | CNN test | 174 | 0.2440 | 0.2454 | 0.057 | 0.023 |
| S3s2 | CNN test | 174 | 0.2566 | 0.2565 | 0.069 | 0.023 |
| S3avg | CNN test | 174 | 0.2503 | 0.2510 | 0.063 | 0.023 |
| V1 | long sentences (held out) | 57 | 0.8494 | 0.8726 | 0.000 | 0.000 |
| pretrained | long sentences (held out) | 57 | 0.3093 | 0.3803 | 0.246 | 0.000 |
| S0 | long sentences (held out) | 57 | 0.0233 | 0.0289 | 0.632 | 0.018 |
| S0s2 | long sentences (held out) | 57 | 0.0240 | 0.0292 | 0.684 | 0.035 |
| S0avg | long sentences (held out) | 57 | 0.0236 | 0.0291 | 0.658 | 0.026 |
| S1 | long sentences (held out) | 57 | 0.0252 | 0.0311 | 0.649 | 0.070 |
| S1s2 | long sentences (held out) | 57 | 0.0256 | 0.0316 | 0.632 | 0.000 |
| S1avg | long sentences (held out) | 57 | 0.0254 | 0.0313 | 0.640 | 0.035 |
| S2 | long sentences (held out) | 57 | 0.0215 | 0.0265 | 0.614 | 0.018 |
| S2s2 | long sentences (held out) | 57 | 0.0246 | 0.0305 | 0.614 | 0.018 |
| S2avg | long sentences (held out) | 57 | 0.0230 | 0.0285 | 0.614 | 0.018 |
| S3 | long sentences (held out) | 57 | 0.0203 | 0.0252 | 0.614 | 0.088 |
| S3s2 | long sentences (held out) | 57 | 0.0248 | 0.0308 | 0.702 | 0.000 |
| S3avg | long sentences (held out) | 57 | 0.0226 | 0.0280 | 0.658 | 0.044 |
| V1 | IAM val | 976 | 0.7946 | 0.7909 | 0.000 | 0.025 |
| pretrained | IAM val | 976 | 0.1220 | 0.1208 | 0.044 | 0.028 |
| S0 | IAM val | 976 | 0.2417 | 0.2376 | 0.006 | 0.064 |
| S0s2 | IAM val | 976 | 0.2360 | 0.2311 | 0.005 | 0.057 |
| S0avg | IAM val | 976 | 0.2389 | 0.2344 | 0.006 | 0.060 |
| S1 | IAM val | 976 | 0.2248 | 0.2205 | 0.009 | 0.058 |
| S1s2 | IAM val | 976 | 0.2496 | 0.2455 | 0.002 | 0.097 |
| S1avg | IAM val | 976 | 0.2372 | 0.2330 | 0.006 | 0.078 |
| S2 | IAM val | 976 | 0.2315 | 0.2273 | 0.007 | 0.101 |
| S2s2 | IAM val | 976 | 0.2992 | 0.2940 | 0.003 | 0.163 |
| S2avg | IAM val | 976 | 0.2653 | 0.2606 | 0.005 | 0.132 |
| S3 | IAM val | 976 | 0.2323 | 0.2278 | 0.005 | 0.099 |
| S3s2 | IAM val | 976 | 0.2456 | 0.2398 | 0.004 | 0.067 |
| S3avg | IAM val | 976 | 0.2389 | 0.2338 | 0.005 | 0.083 |
| V1 | new writers | 160 | 0.8980 | 0.9117 | 0.000 | 0.062 |
| pretrained | new writers | 160 | 0.4308 | 0.4884 | 0.075 | 0.062 |
| S0 | new writers | 160 | 0.2739 | 0.2826 | 0.087 | 0.075 |
| S0s2 | new writers | 160 | 0.2619 | 0.2812 | 0.081 | 0.044 |
| S0avg | new writers | 160 | 0.2679 | 0.2819 | 0.084 | 0.059 |
| S1 | new writers | 160 | 0.2413 | 0.2682 | 0.131 | 0.075 |
| S1s2 | new writers | 160 | 0.2573 | 0.2809 | 0.138 | 0.056 |
| S1avg | new writers | 160 | 0.2493 | 0.2745 | 0.134 | 0.066 |
| S2 | new writers | 160 | 0.2592 | 0.2809 | 0.094 | 0.094 |
| S2s2 | new writers | 160 | 0.2702 | 0.2866 | 0.075 | 0.056 |
| S2avg | new writers | 160 | 0.2647 | 0.2838 | 0.084 | 0.075 |
| S3 | new writers | 160 | 0.2515 | 0.2767 | 0.113 | 0.156 |
| S3s2 | new writers | 160 | 0.2678 | 0.2948 | 0.087 | 0.062 |
| S3avg | new writers | 160 | 0.2597 | 0.2858 | 0.100 | 0.109 |
| V1 | held-out long lines only (over 40 chars) | 41 | 0.8871 | 0.8880 | 0.000 | 0.000 |
| pretrained | held-out long lines only (over 40 chars) | 41 | 0.4273 | 0.4281 | 0.000 | 0.000 |
| S0 | held-out long lines only (over 40 chars) | 41 | 0.0324 | 0.0326 | 0.512 | 0.024 |
| S0s2 | held-out long lines only (over 40 chars) | 41 | 0.0325 | 0.0326 | 0.585 | 0.049 |
| S0avg | held-out long lines only (over 40 chars) | 41 | 0.0325 | 0.0326 | 0.549 | 0.037 |
| S1 | held-out long lines only (over 40 chars) | 41 | 0.0350 | 0.0350 | 0.512 | 0.098 |
| S1s2 | held-out long lines only (over 40 chars) | 41 | 0.0355 | 0.0356 | 0.488 | 0.000 |
| S1avg | held-out long lines only (over 40 chars) | 41 | 0.0352 | 0.0353 | 0.500 | 0.049 |
| S2 | held-out long lines only (over 40 chars) | 41 | 0.0299 | 0.0299 | 0.561 | 0.024 |
| S2s2 | held-out long lines only (over 40 chars) | 41 | 0.0342 | 0.0344 | 0.463 | 0.024 |
| S2avg | held-out long lines only (over 40 chars) | 41 | 0.0320 | 0.0322 | 0.512 | 0.024 |
| S3 | held-out long lines only (over 40 chars) | 41 | 0.0282 | 0.0284 | 0.488 | 0.122 |
| S3s2 | held-out long lines only (over 40 chars) | 41 | 0.0345 | 0.0347 | 0.585 | 0.000 |
| S3avg | held-out long lines only (over 40 chars) | 41 | 0.0314 | 0.0316 | 0.537 | 0.061 |

## Primary criterion (fixed before training): whiteboard validation, mean of 2 seeds

A setting **helps** only if the seed-averaged 95% interval is below 0 AND both seeds improve on their own baseline. Whiteboard validation also chose each run's checkpoint, so its absolute values are slightly optimistic, equally for every run.

| setting | seed 1 vs S0 | seed 2 vs S0s2 | mean of seeds | 95% interval | decision |
|---|---|---|---|---|---|
| S1 (S0 + stage-wise backbone learning rates) | -0.0066 | -0.0018 | -0.0042 | -0.0085 .. +0.0012 | **not shown** |
| S2 (S0 + focal CTC loss) | -0.0040 | -0.0014 | -0.0027 | -0.0078 .. +0.0026 | **not shown** |
| S3 (S0 + stage-wise learning rates + focal CTC) | -0.0025 | -0.0040 | -0.0032 | -0.0082 .. +0.0045 | **not shown** |

For scale: the same recipe with a different seed (S0s2 vs S0) differs by +0.0001. Read the IAM and test tables below for what each setting does elsewhere; they do not change the decision.

## Comparisons on whiteboard val (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| S1avg vs S0avg | stage-wise LR vs baseline (mean of 2 seeds) | -0.0042 | -0.0085 .. +0.0012 | 26 / 20 | 26 | no clear difference |
| S2avg vs S0avg | focal CTC vs baseline (mean of 2 seeds) | -0.0027 | -0.0078 .. +0.0026 | 28 / 25 | 26 | no clear difference |
| S3avg vs S0avg | both vs baseline (mean of 2 seeds) | -0.0032 | -0.0082 .. +0.0045 | 30 / 31 | 26 | no clear difference |
| S3avg vs S1avg | adding focal CTC to stage-wise LR (mean of 2 seeds) | +0.0010 | -0.0024 .. +0.0061 | 19 / 26 | 26 | no clear difference |
| S3avg vs S2avg | adding stage-wise LR to focal CTC (mean of 2 seeds) | -0.0005 | -0.0062 .. +0.0067 | 25 / 25 | 26 | no clear difference |
| S1 vs S0 | stage-wise LR, seed 1 | -0.0066 | -0.0112 .. +0.0003 | 24 / 13 | 26 | no clear difference |
| S1s2 vs S0s2 | stage-wise LR, seed 2 | -0.0018 | -0.0103 .. +0.0053 | 16 / 18 | 26 | no clear difference |
| S2 vs S0 | focal CTC, seed 1 | -0.0040 | -0.0106 .. +0.0047 | 24 / 17 | 26 | no clear difference |
| S2s2 vs S0s2 | focal CTC, seed 2 | -0.0014 | -0.0120 .. +0.0066 | 19 / 22 | 26 | no clear difference |
| S3 vs S0 | both, seed 1 | -0.0025 | -0.0080 .. +0.0066 | 25 / 26 | 26 | no clear difference |
| S3s2 vs S0s2 | both, seed 2 | -0.0040 | -0.0146 .. +0.0095 | 21 / 17 | 26 | no clear difference |
| S0s2 vs S0 | seed noise: identical recipe, different seed | +0.0001 | -0.0107 .. +0.0151 | 26 / 20 | 26 | no clear difference |
| S0avg vs pretrained | clean baseline vs no fine-tune | -0.0797 | -0.1098 .. -0.0472 | 75 / 6 | 26 | better |

## Comparisons on CNN test (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| S1avg vs S0avg | stage-wise LR vs baseline (mean of 2 seeds) | -0.0091 | -0.0186 .. +0.0017 | 81 / 56 | 38 | no clear difference |
| S2avg vs S0avg | focal CTC vs baseline (mean of 2 seeds) | -0.0063 | -0.0161 .. +0.0041 | 76 / 68 | 38 | no clear difference |
| S3avg vs S0avg | both vs baseline (mean of 2 seeds) | -0.0047 | -0.0154 .. +0.0067 | 77 / 67 | 38 | no clear difference |
| S3avg vs S1avg | adding focal CTC to stage-wise LR (mean of 2 seeds) | +0.0044 | -0.0026 .. +0.0121 | 65 / 67 | 38 | no clear difference |
| S3avg vs S2avg | adding stage-wise LR to focal CTC (mean of 2 seeds) | +0.0016 | -0.0060 .. +0.0095 | 77 / 67 | 38 | no clear difference |
| S1 vs S0 | stage-wise LR, seed 1 | -0.0247 | -0.0363 .. -0.0110 | 88 / 36 | 38 | better |
| S1s2 vs S0s2 | stage-wise LR, seed 2 | +0.0064 | -0.0084 .. +0.0221 | 50 / 72 | 38 | no clear difference |
| S2 vs S0 | focal CTC, seed 1 | -0.0390 | -0.0535 .. -0.0230 | 100 / 36 | 38 | better |
| S2s2 vs S0s2 | focal CTC, seed 2 | +0.0264 | +0.0133 .. +0.0406 | 41 / 85 | 38 | worse |
| S3 vs S0 | both, seed 1 | -0.0242 | -0.0372 .. -0.0090 | 86 / 41 | 38 | better |
| S3s2 vs S0s2 | both, seed 2 | +0.0148 | -0.0013 .. +0.0316 | 53 / 72 | 38 | no clear difference |
| S0s2 vs S0 | seed noise: identical recipe, different seed | -0.0264 | -0.0404 .. -0.0099 | 85 / 37 | 38 | better |
| S0avg vs pretrained | clean baseline vs no fine-tune | -0.2029 | -0.2401 .. -0.1662 | 158 / 9 | 38 | better |

## Comparisons on long sentences (held out) (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| S1avg vs S0avg | stage-wise LR vs baseline (mean of 2 seeds) | +0.0017 | -0.0012 .. +0.0051 | 4 / 11 | 7 | no clear difference |
| S2avg vs S0avg | focal CTC vs baseline (mean of 2 seeds) | -0.0006 | -0.0018 .. +0.0005 | 8 / 8 | 7 | no clear difference |
| S3avg vs S0avg | both vs baseline (mean of 2 seeds) | -0.0011 | -0.0047 .. +0.0022 | 7 / 7 | 7 | no clear difference |
| S3avg vs S1avg | adding focal CTC to stage-wise LR (mean of 2 seeds) | -0.0028 | -0.0080 .. +0.0006 | 11 / 4 | 7 | no clear difference |
| S3avg vs S2avg | adding stage-wise LR to focal CTC (mean of 2 seeds) | -0.0005 | -0.0044 .. +0.0028 | 10 / 5 | 7 | no clear difference |
| S1 vs S0 | stage-wise LR, seed 1 | +0.0019 | -0.0033 .. +0.0065 | 3 / 8 | 7 | no clear difference |
| S1s2 vs S0s2 | stage-wise LR, seed 2 | +0.0016 | -0.0015 .. +0.0051 | 4 / 8 | 7 | no clear difference |
| S2 vs S0 | focal CTC, seed 1 | -0.0018 | -0.0065 .. +0.0019 | 7 / 4 | 7 | no clear difference |
| S2s2 vs S0s2 | focal CTC, seed 2 | +0.0006 | -0.0023 .. +0.0045 | 6 / 11 | 7 | no clear difference |
| S3 vs S0 | both, seed 1 | -0.0030 | -0.0113 .. +0.0027 | 4 / 5 | 7 | no clear difference |
| S3s2 vs S0s2 | both, seed 2 | +0.0008 | -0.0025 .. +0.0043 | 4 / 6 | 7 | no clear difference |
| S0s2 vs S0 | seed noise: identical recipe, different seed | +0.0007 | -0.0062 .. +0.0051 | 6 / 6 | 7 | no clear difference |
| S0avg vs pretrained | clean baseline vs no fine-tune | -0.2857 | -0.4251 .. -0.1311 | 43 / 0 | 7 | better |

## Comparisons on held-out long lines only (over 40 chars) (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| S1avg vs S0avg | stage-wise LR vs baseline (mean of 2 seeds) | +0.0028 | -0.0010 .. +0.0076 | 3 / 11 | 5 | no clear difference |
| S2avg vs S0avg | focal CTC vs baseline (mean of 2 seeds) | -0.0004 | -0.0019 .. +0.0011 | 7 / 8 | 5 | no clear difference |
| S3avg vs S0avg | both vs baseline (mean of 2 seeds) | -0.0011 | -0.0065 .. +0.0035 | 6 / 7 | 5 | no clear difference |
| S3avg vs S1avg | adding focal CTC to stage-wise LR (mean of 2 seeds) | -0.0039 | -0.0113 .. +0.0010 | 11 / 4 | 5 | no clear difference |
| S3avg vs S2avg | adding stage-wise LR to focal CTC (mean of 2 seeds) | -0.0007 | -0.0063 .. +0.0038 | 10 / 5 | 5 | no clear difference |
| S1 vs S0 | stage-wise LR, seed 1 | +0.0026 | -0.0050 .. +0.0089 | 3 / 8 | 5 | no clear difference |
| S1s2 vs S0s2 | stage-wise LR, seed 2 | +0.0030 | -0.0004 .. +0.0077 | 3 / 8 | 5 | no clear difference |
| S2 vs S0 | focal CTC, seed 1 | -0.0025 | -0.0089 .. +0.0030 | 7 / 4 | 5 | no clear difference |
| S2s2 vs S0s2 | focal CTC, seed 2 | +0.0017 | -0.0018 .. +0.0072 | 5 / 11 | 5 | no clear difference |
| S3 vs S0 | both, seed 1 | -0.0041 | -0.0172 .. +0.0039 | 4 / 5 | 5 | no clear difference |
| S3s2 vs S0s2 | both, seed 2 | +0.0020 | -0.0022 .. +0.0062 | 3 / 6 | 5 | no clear difference |
| S0s2 vs S0 | seed noise: identical recipe, different seed | +0.0001 | -0.0103 .. +0.0057 | 6 / 5 | 5 | no clear difference |
| S0avg vs pretrained | clean baseline vs no fine-tune | -0.3949 | -0.4801 .. -0.3089 | 41 / 0 | 5 | better |

## Comparisons on new writers (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| S1avg vs S0avg | stage-wise LR vs baseline (mean of 2 seeds) | -0.0186 | -0.0339 .. -0.0075 | 84 / 53 | 40 | better |
| S2avg vs S0avg | focal CTC vs baseline (mean of 2 seeds) | -0.0032 | -0.0145 .. +0.0065 | 61 / 68 | 40 | no clear difference |
| S3avg vs S0avg | both vs baseline (mean of 2 seeds) | -0.0082 | -0.0245 .. +0.0029 | 64 / 63 | 40 | no clear difference |
| S3avg vs S1avg | adding focal CTC to stage-wise LR (mean of 2 seeds) | +0.0103 | -0.0020 .. +0.0217 | 49 / 80 | 40 | no clear difference |
| S3avg vs S2avg | adding stage-wise LR to focal CTC (mean of 2 seeds) | -0.0050 | -0.0184 .. +0.0049 | 65 / 59 | 40 | no clear difference |
| S1 vs S0 | stage-wise LR, seed 1 | -0.0326 | -0.0565 .. -0.0130 | 79 / 42 | 40 | better |
| S1s2 vs S0s2 | stage-wise LR, seed 2 | -0.0046 | -0.0217 .. +0.0102 | 68 / 45 | 40 | no clear difference |
| S2 vs S0 | focal CTC, seed 1 | -0.0147 | -0.0331 .. -0.0027 | 62 / 55 | 40 | better |
| S2s2 vs S0s2 | focal CTC, seed 2 | +0.0083 | -0.0074 .. +0.0250 | 60 / 63 | 40 | no clear difference |
| S3 vs S0 | both, seed 1 | -0.0223 | -0.0461 .. -0.0054 | 63 / 53 | 40 | better |
| S3s2 vs S0s2 | both, seed 2 | +0.0059 | -0.0102 .. +0.0180 | 46 / 65 | 40 | no clear difference |
| S0s2 vs S0 | seed noise: identical recipe, different seed | -0.0119 | -0.0322 .. +0.0043 | 59 / 62 | 40 | no clear difference |
| S0avg vs pretrained | clean baseline vs no fine-tune | -0.1629 | -0.1968 .. -0.1183 | 131 / 21 | 40 | better |

## Comparisons on IAM val (rules fixed before training)

**better** only if the whole 95% interval of the CER difference is below 0; resampling whole photos (or whole sentences for the held-out long sentences).

| comparison | what changes | CER difference | 95% interval | crops better / worse | clusters | verdict |
|---|---|---|---|---|---|---|
| S1avg vs S0avg | stage-wise LR vs baseline (mean of 2 seeds) | -0.0016 | -0.0046 .. +0.0011 | 447 / 412 | 976 | no clear difference |
| S2avg vs S0avg | focal CTC vs baseline (mean of 2 seeds) | +0.0265 | +0.0231 .. +0.0298 | 250 / 634 | 976 | worse |
| S3avg vs S0avg | both vs baseline (mean of 2 seeds) | +0.0001 | -0.0029 .. +0.0029 | 430 / 436 | 976 | no clear difference |
| S3avg vs S1avg | adding focal CTC to stage-wise LR (mean of 2 seeds) | +0.0017 | -0.0009 .. +0.0045 | 414 / 430 | 976 | no clear difference |
| S3avg vs S2avg | adding stage-wise LR to focal CTC (mean of 2 seeds) | -0.0264 | -0.0297 .. -0.0232 | 636 / 246 | 976 | better |
| S1 vs S0 | stage-wise LR, seed 1 | -0.0169 | -0.0205 .. -0.0133 | 489 / 292 | 976 | better |
| S1s2 vs S0s2 | stage-wise LR, seed 2 | +0.0136 | +0.0089 .. +0.0180 | 358 / 460 | 976 | worse |
| S2 vs S0 | focal CTC, seed 1 | -0.0102 | -0.0139 .. -0.0064 | 457 / 337 | 976 | better |
| S2s2 vs S0s2 | focal CTC, seed 2 | +0.0632 | +0.0572 .. +0.0690 | 178 / 683 | 976 | worse |
| S3 vs S0 | both, seed 1 | -0.0094 | -0.0132 .. -0.0056 | 427 / 348 | 976 | better |
| S3s2 vs S0s2 | both, seed 2 | +0.0096 | +0.0056 .. +0.0136 | 347 / 443 | 976 | worse |
| S0s2 vs S0 | seed noise: identical recipe, different seed | -0.0057 | -0.0094 .. -0.0021 | 429 / 342 | 976 | better |
| S0avg vs pretrained | clean baseline vs no fine-tune | +0.1169 | +0.1116 .. +0.1218 | 35 / 908 | 976 | worse |

## New writers, by writer (mean line CER)

Teammates' handwriting that no S0-S3 run trained on (the V-runs of versions_2x2 saw 64 teammates' crops of unknown writer). Two writers only: treat as a first look at generalisation, not a population estimate.

| model | writer_DS (70) | writer_JH (90) |
|---|---|---|
| V1 | 0.9175 | 0.8829 |
| pretrained | 0.5116 | 0.3680 |
| S0 | 0.2888 | 0.2622 |
| S0s2 | 0.2907 | 0.2396 |
| S0avg | 0.2898 | 0.2509 |
| S1 | 0.2713 | 0.2179 |
| S1s2 | 0.2827 | 0.2376 |
| S1avg | 0.2770 | 0.2278 |
| S2 | 0.2934 | 0.2325 |
| S2s2 | 0.2934 | 0.2522 |
| S2avg | 0.2934 | 0.2424 |
| S3 | 0.2900 | 0.2216 |
| S3s2 | 0.3045 | 0.2392 |
| S3avg | 0.2972 | 0.2304 |

## By label length (does a version fail on long lines?)

| set | length | V1 | pretrained | S0 | S0s2 | S0avg | S1 | S1s2 | S1avg | S2 | S2s2 | S2avg | S3 | S3s2 | S3avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| whiteboard val | 1-20 chars (86) | 0.998 | 0.133 | 0.060 | 0.066 | 0.063 | 0.054 | 0.060 | 0.057 | 0.056 | 0.060 | 0.058 | 0.060 | 0.057 | 0.059 |
| whiteboard val | 21-40 chars (42) | 0.736 | 0.199 | 0.100 | 0.092 | 0.096 | 0.094 | 0.100 | 0.097 | 0.102 | 0.100 | 0.101 | 0.095 | 0.100 | 0.098 |
| whiteboard val | over 40 chars (8) | 0.803 | 0.152 | 0.102 | 0.083 | 0.093 | 0.092 | 0.071 | 0.081 | 0.074 | 0.081 | 0.077 | 0.089 | 0.070 | 0.079 |
| CNN test | 1-20 chars (57) | 0.851 | 0.407 | 0.242 | 0.225 | 0.234 | 0.228 | 0.217 | 0.223 | 0.185 | 0.236 | 0.211 | 0.221 | 0.252 | 0.236 |
| CNN test | 21-40 chars (117) | 0.795 | 0.483 | 0.281 | 0.250 | 0.265 | 0.251 | 0.263 | 0.257 | 0.251 | 0.284 | 0.267 | 0.255 | 0.259 | 0.257 |
| long sentences (held out) | 21-40 chars (16) | 0.753 | 0.007 | 0.000 | 0.002 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| long sentences (held out) | over 40 chars (41) | 0.887 | 0.427 | 0.032 | 0.033 | 0.032 | 0.035 | 0.036 | 0.035 | 0.030 | 0.034 | 0.032 | 0.028 | 0.034 | 0.031 |
| IAM val | 1-20 chars (36) | 0.883 | 0.222 | 0.361 | 0.368 | 0.364 | 0.348 | 0.351 | 0.349 | 0.342 | 0.416 | 0.379 | 0.349 | 0.377 | 0.363 |
| IAM val | 21-40 chars (308) | 0.797 | 0.108 | 0.236 | 0.233 | 0.235 | 0.220 | 0.245 | 0.232 | 0.230 | 0.296 | 0.263 | 0.228 | 0.247 | 0.237 |
| IAM val | over 40 chars (632) | 0.788 | 0.123 | 0.238 | 0.230 | 0.234 | 0.220 | 0.246 | 0.233 | 0.226 | 0.294 | 0.260 | 0.228 | 0.238 | 0.233 |
| new writers | 1-20 chars (35) | 0.888 | 0.311 | 0.326 | 0.271 | 0.299 | 0.218 | 0.242 | 0.230 | 0.252 | 0.295 | 0.273 | 0.232 | 0.236 | 0.234 |
| new writers | 21-40 chars (56) | 0.850 | 0.358 | 0.186 | 0.177 | 0.181 | 0.160 | 0.186 | 0.173 | 0.189 | 0.179 | 0.184 | 0.176 | 0.200 | 0.188 |
| new writers | over 40 chars (69) | 0.942 | 0.551 | 0.319 | 0.326 | 0.322 | 0.319 | 0.323 | 0.321 | 0.320 | 0.332 | 0.326 | 0.323 | 0.340 | 0.331 |
| held-out long lines only (over 40 chars) | over 40 chars (41) | 0.887 | 0.427 | 0.032 | 0.033 | 0.032 | 0.035 | 0.036 | 0.035 | 0.030 | 0.034 | 0.032 | 0.028 | 0.034 | 0.031 |

## Training lines per run (only hard limits are applied)

| run | lines | left out (reason) | kept although the text covers fewer positions than it needs |
|---|---|---|---|
| S0 | 1359 | {'label longer than the limit': 3} | 28 |
| S0s2 | 1359 | {'label longer than the limit': 3} | 28 |
| S1 | 1359 | {'label longer than the limit': 3} | 28 |
| S1s2 | 1359 | {'label longer than the limit': 3} | 28 |
| S2 | 1359 | {'label longer than the limit': 3} | 28 |
| S2s2 | 1359 | {'label longer than the limit': 3} | 28 |
| S3 | 1359 | {'label longer than the limit': 3} | 28 |
| S3s2 | 1359 | {'label longer than the limit': 3} | 28 |

## Training runs

| run | lines | epochs | updates | minutes | best validation |
|---|---|---|---|---|---|
| S0 | 1359 | 71 | 6035 | 23.2 | norm_edit_dis: 0.921761091159019, is_float16: False, acc: 0.5220587851 |
| S0s2 | 1359 | 71 | 6035 | 23.3 | norm_edit_dis: 0.9214757717634158, is_float16: False, acc: 0.551470547 |
| S1 | 1359 | 71 | 6035 | 23.6 | norm_edit_dis: 0.9245841482255766, is_float16: False, acc: 0.551470547 |
| S1s2 | 1359 | 71 | 6035 | 23.4 | norm_edit_dis: 0.922084020831707, is_float16: False, acc: 0.5367646664 |
| S2 | 1359 | 71 | 6035 | 23.2 | norm_edit_dis: 0.9255565874171453, is_float16: False, acc: 0.485294081 |
| S2s2 | 1359 | 71 | 6035 | 23.5 | norm_edit_dis: 0.9220613130178, is_float16: False, acc: 0.492647022599 |
| S3 | 1359 | 71 | 6035 | 23.2 | norm_edit_dis: 0.9229867008927315, is_float16: False, acc: 0.477941141 |
| S3s2 | 1359 | 71 | 6035 | 23.5 | norm_edit_dis: 0.9230909765074159, is_float16: False, acc: 0.485294081 |

Limits: two seeds for S0-S3 (S0s2 vs S0 shows the size of seed-to-seed change); the intervals reflect evaluation-cluster sampling, not training-seed variability. Validation shares source photos with training (different texts); the CNN test board has few distinct texts (see predictions_all.csv); the held-out long sentences are 7 sentences from one board and writer; the new-writer set is 160 crops from 2 writers (intervals resample whole source photos).
