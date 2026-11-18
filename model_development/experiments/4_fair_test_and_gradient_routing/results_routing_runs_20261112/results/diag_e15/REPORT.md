# Where do the CTC and NRTR losses act? Checkpoints at epoch 15 of every run (same training length)

Readers: S0 (guidance on, seed 1), S0s2 (guidance on, seed 2), G0 (guidance off, seed 1), G0s2 (guidance off, seed 2).
Same training data for all four. Pretrained = PP-OCRv5_server_rec before fine-tuning.

## 1. How far each part moved from the pretrained weights (relative change, %)

guidance/seed = distance between guidance-on and guidance-off readers of the same seed, divided by the distance between two seeds of the same recipe. About 1 = guidance changed this part no more than a different seed does; clearly above 1 = guidance changed this part.
same direction = cosine between the guidance effect in seed 1 (G0 - S0) and in seed 2 (G0s2 - S0s2): about 0 = the two seeds changed this part in unrelated ways (noise); clearly above 0 = turning guidance off pushes this part the same way every time.

| Part | Params | S0 | S0s2 | G0 | G0s2 | guidance/seed | same direction |
|---|---|---|---|---|---|---|---|
| stem | 0.03 M | 0.65 | 0.77 | 0.41 | 0.56 | 1.03 | 0.061 |
| stage 1 | 0.16 M | 1.73 | 1.75 | 0.81 | 0.95 | 1.09 | 0.070 |
| stage 2 | 0.84 M | 2.29 | 2.30 | 1.06 | 1.26 | 1.08 | 0.059 |
| stage 3 | 5.81 M | 2.42 | 2.46 | 1.40 | 1.45 | 1.09 | 0.113 |
| stage 4 | 6.72 M | 2.58 | 2.63 | 2.41 | 2.37 | 1.10 | 0.168 |
| CTC neck (SVTR) | 5.26 M | 5.94 | 6.07 | 4.93 | 5.16 | 1.04 | -0.164 |
| CTC head | 2.22 M | 2.69 | 2.75 | 2.69 | 2.66 | 1.31 | 0.414 |
| NRTR head | 26.30 M | 3.35 | 3.36 | 3.48 | 3.47 | 1.12 | 0.186 |

## 2. Stage features: similarity between readers (linear CKA, 1 = same)

Guidance effect = S0~G0 and S0s2~G0s2 (same seed). Seed noise = S0~S0s2 and G0~G0s2. If guidance changed a stage, its guidance similarity is clearly lower than its seed similarity.

Control = S0 against G0 with the positions shuffled; it must be far below the others, or CKA is not measuring anything.

| Images | Stage | S0~G0 | S0s2~G0s2 | S0~S0s2 | G0~G0s2 | pre~S0 | pre~G0 | control |
|---|---|---|---|---|---|---|---|---|
| familiar | stem | 0.9994 | 0.9996 | 0.9994 | 0.9994 | 0.9557 | 0.9588 | 0.0001 |
| familiar | stage 1 | 0.9957 | 0.9948 | 0.9948 | 0.9967 | 0.9560 | 0.9555 | 0.0002 |
| familiar | stage 2 | 0.9732 | 0.9849 | 0.9399 | 0.9890 | 0.7806 | 0.8668 | 0.0006 |
| familiar | stage 3 | 0.8331 | 0.7686 | 0.8767 | 0.8813 | 0.5728 | 0.6500 | 0.0027 |
| familiar | stage 4 | 0.6276 | 0.6270 | 0.7935 | 0.7255 | 0.7380 | 0.5493 | 0.0034 |
| unfamiliar | stem | 0.9995 | 0.9994 | 0.9997 | 0.9992 | 0.9711 | 0.9748 | 0.0001 |
| unfamiliar | stage 1 | 0.9942 | 0.9949 | 0.9946 | 0.9976 | 0.9062 | 0.9036 | 0.0011 |
| unfamiliar | stage 2 | 0.9740 | 0.9881 | 0.9704 | 0.9944 | 0.8385 | 0.8881 | 0.0007 |
| unfamiliar | stage 3 | 0.8177 | 0.7584 | 0.8693 | 0.8876 | 0.5366 | 0.6631 | 0.0033 |
| unfamiliar | stage 4 | 0.6225 | 0.6277 | 0.7760 | 0.7644 | 0.7153 | 0.5916 | 0.0033 |

## 3. CTC vs NRTR gradient per part (same fixed training batches)

Sanity check: with guidance on, CTC gradient in the backbone = 0 (must be 0), in the CTC neck = 3.15e+04 (> 0).
24 batches of shape [16, 3, 64, 1024]. CTC measured with guidance off for every reader.
CTC share = CTC gradient size / (CTC + NRTR). cos = direction agreement (+1 same way, 0 unrelated, -1 opposite); per batch and for the gradient summed over all batches.

| Reader | Part | CTC grad | NRTR grad | CTC share | cos per batch | cos summed |
|---|---|---|---|---|---|---|
| pretrained | stem | 7.48e+03 | 61.6 | 0.99 | 0.18 | 0.32 |
| pretrained | stage 1 | 7.19e+03 | 65.8 | 0.99 | 0.13 | 0.72 |
| pretrained | stage 2 | 4.43e+03 | 37.2 | 0.99 | 0.19 | 0.41 |
| pretrained | stage 3 | 7.44e+03 | 64.2 | 0.99 | 0.13 | 0.36 |
| pretrained | stage 4 | 3.64e+03 | 44.5 | 0.99 | 0.26 | 0.39 |
| pretrained | CTC neck (SVTR) | 712 | 0 | 1.00 | nan | nan |
| pretrained | CTC head | 384 | 0 | 1.00 | nan | nan |
| pretrained | NRTR head | 0 | 275 | 0.00 | nan | nan |
| S0 | stem | 80 | 1.56 | 0.98 | 0.23 | -0.10 |
| S0 | stage 1 | 60.5 | 1.09 | 0.98 | 0.24 | 0.30 |
| S0 | stage 2 | 64 | 1.27 | 0.98 | 0.21 | 0.25 |
| S0 | stage 3 | 157 | 3.05 | 0.98 | 0.15 | 0.20 |
| S0 | stage 4 | 136 | 2.31 | 0.98 | 0.07 | 0.11 |
| S0 | CTC neck (SVTR) | 41.8 | 0 | 1.00 | nan | nan |
| S0 | CTC head | 12.5 | 0 | 1.00 | nan | nan |
| S0 | NRTR head | 0 | 9.29 | 0.00 | nan | nan |
| S0s2 | stem | 67.6 | 1.22 | 0.98 | 0.20 | 0.36 |
| S0s2 | stage 1 | 66.1 | 1.11 | 0.98 | 0.18 | 0.18 |
| S0s2 | stage 2 | 71.6 | 1.25 | 0.98 | 0.17 | 0.12 |
| S0s2 | stage 3 | 167 | 3 | 0.98 | 0.12 | 0.05 |
| S0s2 | stage 4 | 152 | 2.44 | 0.98 | 0.05 | 0.05 |
| S0s2 | CTC neck (SVTR) | 45.4 | 0 | 1.00 | nan | nan |
| S0s2 | CTC head | 11.9 | 0 | 1.00 | nan | nan |
| S0s2 | NRTR head | 0 | 9.85 | 0.00 | nan | nan |
| G0 | stem | 78.5 | 1.43 | 0.98 | 0.19 | 0.21 |
| G0 | stage 1 | 75 | 1.51 | 0.98 | 0.09 | 0.36 |
| G0 | stage 2 | 72.9 | 1.37 | 0.98 | 0.15 | 0.16 |
| G0 | stage 3 | 166 | 3.15 | 0.98 | 0.13 | 0.19 |
| G0 | stage 4 | 135 | 2.69 | 0.98 | 0.05 | 0.11 |
| G0 | CTC neck (SVTR) | 41.2 | 0 | 1.00 | nan | nan |
| G0 | CTC head | 12.3 | 0 | 1.00 | nan | nan |
| G0 | NRTR head | 0 | 15.7 | 0.00 | nan | nan |
| G0s2 | stem | 60.5 | 1.19 | 0.98 | 0.16 | 0.19 |
| G0s2 | stage 1 | 61.5 | 1.18 | 0.98 | 0.18 | 0.30 |
| G0s2 | stage 2 | 60.6 | 1.23 | 0.98 | 0.18 | 0.24 |
| G0s2 | stage 3 | 142 | 3.02 | 0.98 | 0.17 | 0.19 |
| G0s2 | stage 4 | 110 | 2.84 | 0.97 | 0.07 | 0.05 |
| G0s2 | CTC neck (SVTR) | 33.5 | 0 | 1.00 | nan | nan |
| G0s2 | CTC head | 10.9 | 0 | 1.00 | nan | nan |
| G0s2 | NRTR head | 0 | 18.2 | 0.00 | nan | nan |

