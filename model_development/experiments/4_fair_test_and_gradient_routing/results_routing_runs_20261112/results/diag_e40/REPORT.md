# Where do the CTC and NRTR losses act? Checkpoints at epoch 40 of every run (same training length)

Readers: S0 (guidance on, seed 1), S0s2 (guidance on, seed 2), G0 (guidance off, seed 1), G0s2 (guidance off, seed 2).
Same training data for all four. Pretrained = PP-OCRv5_server_rec before fine-tuning.

## 4. CTC head vs NRTR head (line CER %, lower is better)

Reading check: 1792 of 1792 CTC readings identical to the saved scoring readings.

| Reader | Set | CTC head | NRTR head |
|---|---|---|---|
| S0 | fresh boards | 9.0 | 28.6 |
| S0 | new writers | 28.1 | 45.5 |
| S0 | whiteboard val | 7.9 | 9.6 |
| S0s2 | fresh boards | 8.3 | 27.7 |
| S0s2 | new writers | 27.2 | 42.3 |
| S0s2 | whiteboard val | 7.4 | 11.9 |
| G0 | fresh boards | 11.6 | 33.6 |
| G0 | new writers | 23.2 | 56.8 |
| G0 | whiteboard val | 7.9 | 14.3 |
| G0s2 | fresh boards | 14.6 | 47.5 |
| G0s2 | new writers | 29.9 | 70.3 |
| G0s2 | whiteboard val | 7.7 | 16.0 |

## 1. How far each part moved from the pretrained weights (relative change, %)

guidance/seed = distance between guidance-on and guidance-off readers of the same seed, divided by the distance between two seeds of the same recipe. About 1 = guidance changed this part no more than a different seed does; clearly above 1 = guidance changed this part.
same direction = cosine between the guidance effect in seed 1 (G0 - S0) and in seed 2 (G0s2 - S0s2): about 0 = the two seeds changed this part in unrelated ways (noise); clearly above 0 = turning guidance off pushes this part the same way every time.

| Part | Params | S0 | S0s2 | G0 | G0s2 | guidance/seed | same direction |
|---|---|---|---|---|---|---|---|
| stem | 0.03 M | 1.01 | 1.15 | 0.68 | 0.88 | 1.05 | 0.115 |
| stage 1 | 0.16 M | 2.51 | 2.55 | 1.38 | 1.73 | 1.06 | 0.062 |
| stage 2 | 0.84 M | 3.30 | 3.35 | 1.73 | 2.17 | 1.06 | 0.051 |
| stage 3 | 5.81 M | 3.43 | 3.47 | 2.37 | 2.56 | 1.06 | 0.097 |
| stage 4 | 6.72 M | 3.46 | 3.50 | 3.60 | 3.63 | 1.11 | 0.185 |
| CTC neck (SVTR) | 5.26 M | 9.07 | 9.25 | 7.48 | 7.73 | 1.09 | 0.023 |
| CTC head | 2.22 M | 7.85 | 7.91 | 9.82 | 9.30 | 1.24 | 0.345 |
| NRTR head | 26.30 M | 3.71 | 3.72 | 3.87 | 3.87 | 1.12 | 0.182 |

## 2. Stage features: similarity between readers (linear CKA, 1 = same)

Guidance effect = S0~G0 and S0s2~G0s2 (same seed). Seed noise = S0~S0s2 and G0~G0s2. If guidance changed a stage, its guidance similarity is clearly lower than its seed similarity.

Control = S0 against G0 with the positions shuffled; it must be far below the others, or CKA is not measuring anything.

| Images | Stage | S0~G0 | S0s2~G0s2 | S0~S0s2 | G0~G0s2 | pre~S0 | pre~G0 | control |
|---|---|---|---|---|---|---|---|---|
| familiar | stem | 0.9988 | 0.9986 | 0.9992 | 0.9981 | 0.9640 | 0.9554 | 0.0001 |
| familiar | stage 1 | 0.9960 | 0.9892 | 0.9951 | 0.9914 | 0.9625 | 0.9572 | 0.0002 |
| familiar | stage 2 | 0.9805 | 0.9842 | 0.9752 | 0.9777 | 0.8352 | 0.8379 | 0.0007 |
| familiar | stage 3 | 0.7974 | 0.7759 | 0.8759 | 0.8533 | 0.5885 | 0.6368 | 0.0024 |
| familiar | stage 4 | 0.5407 | 0.6275 | 0.7551 | 0.6560 | 0.7082 | 0.4758 | 0.0034 |
| unfamiliar | stem | 0.9991 | 0.9978 | 0.9986 | 0.9987 | 0.9740 | 0.9719 | 0.0001 |
| unfamiliar | stage 1 | 0.9928 | 0.9915 | 0.9944 | 0.9917 | 0.9164 | 0.9201 | 0.0011 |
| unfamiliar | stage 2 | 0.9784 | 0.9768 | 0.9794 | 0.9868 | 0.8903 | 0.8775 | 0.0007 |
| unfamiliar | stage 3 | 0.7611 | 0.7451 | 0.8581 | 0.8405 | 0.5371 | 0.6279 | 0.0035 |
| unfamiliar | stage 4 | 0.5456 | 0.6103 | 0.7576 | 0.6815 | 0.6730 | 0.5446 | 0.0036 |

## 3. CTC vs NRTR gradient per part (same fixed training batches)

Sanity check: with guidance on, CTC gradient in the backbone = 0 (must be 0), in the CTC neck = 2.91e+04 (> 0).
24 batches of shape [16, 3, 64, 1024]. CTC measured with guidance off for every reader.
CTC share = CTC gradient size / (CTC + NRTR). cos = direction agreement (+1 same way, 0 unrelated, -1 opposite); per batch and for the gradient summed over all batches.

| Reader | Part | CTC grad | NRTR grad | CTC share | cos per batch | cos summed |
|---|---|---|---|---|---|---|
| pretrained | stem | 7.78e+03 | 54.1 | 0.99 | 0.10 | 0.44 |
| pretrained | stage 1 | 8.03e+03 | 64 | 0.99 | 0.22 | 0.41 |
| pretrained | stage 2 | 4.96e+03 | 35.5 | 0.99 | 0.28 | 0.46 |
| pretrained | stage 3 | 7.35e+03 | 59.1 | 0.99 | 0.18 | 0.41 |
| pretrained | stage 4 | 3.65e+03 | 44.2 | 0.99 | 0.26 | 0.40 |
| pretrained | CTC neck (SVTR) | 724 | 0 | 1.00 | nan | nan |
| pretrained | CTC head | 389 | 0 | 1.00 | nan | nan |
| pretrained | NRTR head | 0 | 274 | 0.00 | nan | nan |
| S0 | stem | 62 | 0.529 | 0.99 | 0.19 | 0.01 |
| S0 | stage 1 | 57.1 | 0.533 | 0.99 | 0.19 | 0.14 |
| S0 | stage 2 | 57.6 | 0.556 | 0.99 | 0.24 | 0.23 |
| S0 | stage 3 | 138 | 1.29 | 0.99 | 0.19 | 0.16 |
| S0 | stage 4 | 130 | 1.07 | 0.99 | 0.08 | 0.05 |
| S0 | CTC neck (SVTR) | 39.2 | 0 | 1.00 | nan | nan |
| S0 | CTC head | 10.5 | 0 | 1.00 | nan | nan |
| S0 | NRTR head | 0 | 4.89 | 0.00 | nan | nan |
| S0s2 | stem | 47 | 0.459 | 0.99 | 0.26 | 0.48 |
| S0s2 | stage 1 | 49.2 | 0.42 | 0.99 | 0.18 | 0.29 |
| S0s2 | stage 2 | 49.4 | 0.457 | 0.99 | 0.23 | 0.26 |
| S0s2 | stage 3 | 128 | 1.11 | 0.99 | 0.19 | 0.16 |
| S0s2 | stage 4 | 138 | 0.947 | 0.99 | 0.08 | 0.09 |
| S0s2 | CTC neck (SVTR) | 41.6 | 0 | 1.00 | nan | nan |
| S0s2 | CTC head | 10.3 | 0 | 1.00 | nan | nan |
| S0s2 | NRTR head | 0 | 5 | 0.00 | nan | nan |
| G0 | stem | 62.6 | 0.636 | 0.99 | 0.19 | 0.40 |
| G0 | stage 1 | 58.8 | 0.678 | 0.99 | 0.19 | 0.41 |
| G0 | stage 2 | 55.8 | 0.627 | 0.99 | 0.20 | 0.22 |
| G0 | stage 3 | 129 | 1.47 | 0.99 | 0.17 | 0.26 |
| G0 | stage 4 | 104 | 1.26 | 0.99 | 0.07 | 0.15 |
| G0 | CTC neck (SVTR) | 32.2 | 0 | 1.00 | nan | nan |
| G0 | CTC head | 10.2 | 0 | 1.00 | nan | nan |
| G0 | NRTR head | 0 | 7.51 | 0.00 | nan | nan |
| G0s2 | stem | 53.7 | 0.556 | 0.99 | 0.15 | 0.68 |
| G0s2 | stage 1 | 52.5 | 0.47 | 0.99 | 0.22 | 0.50 |
| G0s2 | stage 2 | 52.9 | 0.509 | 0.99 | 0.23 | 0.28 |
| G0s2 | stage 3 | 124 | 1.28 | 0.99 | 0.21 | 0.28 |
| G0s2 | stage 4 | 95 | 1.21 | 0.99 | 0.09 | 0.17 |
| G0s2 | CTC neck (SVTR) | 31.4 | 0 | 1.00 | nan | nan |
| G0s2 | CTC head | 10.4 | 0 | 1.00 | nan | nan |
| G0s2 | NRTR head | 0 | 8.29 | 0.00 | nan | nan |

