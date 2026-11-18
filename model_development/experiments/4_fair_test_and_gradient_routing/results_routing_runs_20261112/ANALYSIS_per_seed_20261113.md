# Per-seed reading of the routing runs (13 Nov 2026)

Added after the run; nothing in results/ or runs/ was changed. Source: runs/*/epochs/*/rows.json.

Why: the intervals in results/report.md resample photos / writers but NOT seeds. Two seeds of the SAME recipe
differ by about 2-2.7 CER points per epoch on new writers and fresh boards, so a single-epoch "better" there can be
seed luck. Here each recipe is averaged over epochs 10-70 (9 checkpoints) per seed, and an epoch counts only if
BOTH seeds point the same way.

## Mean line CER over epochs 10-70, per seed (%)

| Run | whiteboard val | new writers | fresh boards |
|---|---|---|---|
| S0 | 7.80 | 27.15 | 11.01 |
| S0s2 | 7.72 | 26.29 | 11.37 |
| G0 | 7.64 | 24.93 | 11.37 |
| G0s2 | 7.42 | 25.79 | 13.33 |
| RE | 7.68 | 27.41 | 11.30 |
| REs2 | 7.58 | 27.71 | 13.52 |
| RL | 8.06 | 26.24 | 9.31 |
| RLs2 | 7.97 | 25.76 | 11.43 |

## Differences (A - B, epoch mean), and epochs where both seeds agree

| A - B | set | seed 1 | seed 2 | better in both | worse in both |
|---|---|---|---|---|---|
| G0 - S0 | new writers | -2.22 | -0.51 | 4/9 | 0/9 |
| G0 - S0 | fresh boards | +0.36 | +1.96 | 1/9 | 3/9 |
| RE - G0 | new writers | +2.48 | +1.93 | 0/9 | 6/9 |
| RL - G0 | fresh boards | -2.06 | -1.90 | 6/9 | 2/9 |
| RL - S0 | new writers | -0.91 | -0.53 | 2/9 | 0/9 |
| RL - S0 | fresh boards | -1.70 | +0.05 | 3/9 | 0/9 |
| RE - RL | new writers | +1.17 | +1.95 | 0/9 | 4/9 |
| RE - RL | fresh boards | +1.99 | +2.09 | 0/9 | 5/9 |
| RE - RL | whiteboard val | -0.38 | -0.39 | 6/9 | 0/9 |

Seed noise (mean |seed 1 - seed 2| per epoch): new writers 1.3-2.7, fresh boards 2.0-2.7, whiteboard val 0.4-0.6.

Caution: RL's val-chosen "best" on fresh boards (8.7) averages RL@66 and RLs2@15 (6.8, one early checkpoint);
use the epoch means above, not that number.
