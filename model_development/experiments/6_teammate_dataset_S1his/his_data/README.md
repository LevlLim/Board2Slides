# Teammate dataset for V2his / V3his (10 Nov 2026)

`prep_his_data.py` downloads the teammate's Hugging Face dataset `TYW02/whiteboard-notes-v1` at revision
`75619db2` (his 10 Nov 16:20 SGT upload), builds a leak-free train / validation / test split and writes the lists for
batch B. It needs `huggingface_hub`, `pillow` and `numpy`; it does not touch the training setup.

## Why the split was rebuilt
His whiteboard images are many photos of the same boards. Splitting by file or by exact label left the same
handwriting on both sides (repeat photos, crops cut at the edge, punctuation variants, a crop holding another crop's
line, one writing under three labels). His 16:20 regroup by label fixed most of it; 85 test crops (22 %) still had their
writing in train, 80 of them from the oct17 boards.

## Rule (default `--assign stratified --ratio 80:10:10`)
1. **Group the same writing**: same label in the same source (spacing and punctuation ignored); near-identical labels
   (crop cut at the edge, one-letter slips) and a label contained in another; a JH whole image with its sub-crops;
   near-identical images in the same source (pixel similarity >= 0.85); every augmented copy with its crop.
2. **Reviewed exceptions** (looked at by eye): `Chapter 1..5`, `Classification` vs numbered classification headings,
   `Convolution` vs `Convolution Neural Networks`, `Kernel` vs `Kernel: K×L`, and similar Jy words are different words,
   so they are not joined.
3. **Split whole groups 80:10:10 within every source × line-length band** (short <= 20, medium 21–40, long > 40
   characters), seeded (20261110); ties keep the teammate's side.

## Result (checked 10 Nov, 18:40)
| line length | crops | train | validation | test |
|---|---|---|---|---|
| short | 1,037 | 846 (82 %) | 99 (10 %) | 92 (9 %) |
| medium | 698 | 577 (83 %) | 61 (9 %) | 60 (9 %) |
| long | 173 | 137 (79 %) | 19 (11 %) | 17 (10 %) |
| total | 1,908 | 1,560 (81.8 %) | 179 (9.4 %) | 169 (8.9 %) |

Not exactly 80:10:10 because groups are never split (the CNN board is only 7 groups: 166 / 6 / 7). 621 crops (33 %)
are in a different set from his split (`lists/moves.csv`).

Checks, all three pairs (test–train, val–train, test–val): no byte-identical file, crop, same-source label or
near/contained same-source label; pixel check of every unaugmented image against every image of the other set
(augmented copies included): max 0.697 / 0.723 / 0.694, none at 0.9.

## Use
- V3his: `his_train_full.txt` (8,508 files). V2his: `his_train_orig.txt` (1,557 originals).
- Checkpoint selection: `his_val_orig.txt` (179). Final score, once: `his_test_orig.txt` (169).
- Fixed copy in his format: `Assignment/whiteboard-notes-v1_75619db2_fixed/` (`train/ val/ test/`,
  `rec_gt_train.txt` / `rec_gt_val.txt` / `rec_gt_test.txt`, `moves.csv`). His original download is unchanged in
  `whiteboard-notes-v1_75619db2/`; earlier attempts are in `_old_versions/`.

Expected hashes (first 12 characters): lists `his_train_full` abd894a606cc, `his_train_orig` 73cd8bd2c66f,
`his_val_orig` 33dd43100ae3, `his_test_orig` c7b761c87de4; fixed copy `rec_gt_train` 652847d60de1, `rec_gt_val`
427a9f85731a, `rec_gt_test` e60f462eed75. Script 860f461d8c2b.

## Label error to fix by hand
`lecture_05_035` ('4. ResNet'), `lecture_06_039` ('4.3 Leaky ReLU') and `lecture_07_035` ('4.2 ReLU') are the same
writing; the image reads '4. ResNet', so the last two labels are wrong. The same three crops are in our own dataset.
