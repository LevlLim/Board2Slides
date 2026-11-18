# Paper-notes scalability check: results

Mean line CER (%), case-insensitive, whitespace collapsed. **Reading** = mean over reference lines the detector found (identical crops for every reader). **End to end** = whole page, missed lines fully wrong, extra boxes as insertions; "app order" uses the package reading order, "order-free" matches crops to lines first.

Detector and crops: 9 Nov OCR package settings. `app_prepared` = Board2Slides Automatic preparation run unchanged before detection. This is the OCR package, not the final deployed app.

## page1_angle_medium_a_IMG_5172.HEIC · original (page 1)

16 boxes · detection 1.74 s · detector path: text large enough (45 px) -> quick boxes kept

| reader | reading | reading, spaces ignored | reading, lines >60 chars | end to end (app order) | end to end (order-free) | found / lines | missed | split | extra boxes | read s |
|---|---|---|---|---|---|---|---|---|---|---|
| downloaded | 18.4 | 17.6 | 41.4 | 70.9 | 72.1 | 11 / 25 | [4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25] | – | 5 | 3.53 |
| whiteboard | 13.8 | 14.3 | 34.3 | 69.3 | 72.1 | 11 / 25 | [4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25] | – | 5 | 5.11 |
| paper | 15.8 | 15.5 | 36.2 | 70.3 | 72.1 | 11 / 25 | [4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25] | – | 5 | 5.11 |
| final | 13.7 | 13.8 | 29.9 | 69.1 | 71.3 | 11 / 25 | [4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25] | – | 5 | 5.19 |

Long lines (>60 characters): line 3: downloaded 63.2, whiteboard 35.3, paper 32.4, final 23.5; line 5: downloaded –, whiteboard –, paper –, final –; line 6: downloaded –, whiteboard –, paper –, final –; line 10: downloaded 6.2, whiteboard 3.1, paper 12.3, final 0.0; line 11: downloaded 1.5, whiteboard 3.1, paper 3.1, final 1.5; line 16: downloaded –, whiteboard –, paper –, final –; line 19: downloaded –, whiteboard –, paper –, final –; line 20: downloaded –, whiteboard –, paper –, final –; line 22: downloaded 94.5, whiteboard 95.9, paper 97.3, final 94.5; line 24: downloaded –, whiteboard –, paper –, final –

## page1_angle_medium_a_IMG_5172.HEIC · app_prepared (page 1)

6 boxes · detection 0.29 s · detector path: text large enough (45 px) -> quick boxes kept
Preparation: perspective skipped (text or camera information unavailable: abstain); crop applied (Cropped around detected writing.); rotation skipped (insufficient independent text rows); lighting applied (Gentle correction for low contrast, dimness, or uneven light.)

| reader | reading | reading, spaces ignored | reading, lines >60 chars | end to end (app order) | end to end (order-free) | found / lines | missed | split | extra boxes | read s |
|---|---|---|---|---|---|---|---|---|---|---|
| downloaded | 26.2 | 25.6 | 91.3 | 93.2 | 92.1 | 4 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25] | – | 2 | 1.32 |
| whiteboard | 25.0 | 24.9 | 91.3 | 93.3 | 92.0 | 4 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25] | – | 2 | 1.83 |
| paper | 25.6 | 23.7 | 91.3 | 93.2 | 92.0 | 4 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25] | – | 2 | 1.9 |
| final | 27.2 | 27.4 | 91.3 | 93.4 | 92.1 | 4 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25] | – | 2 | 1.91 |

Long lines (>60 characters): line 3: downloaded –, whiteboard –, paper –, final –; line 5: downloaded –, whiteboard –, paper –, final –; line 6: downloaded –, whiteboard –, paper –, final –; line 10: downloaded –, whiteboard –, paper –, final –; line 11: downloaded –, whiteboard –, paper –, final –; line 16: downloaded 91.3, whiteboard 91.3, paper 91.3, final 91.3; line 19: downloaded –, whiteboard –, paper –, final –; line 20: downloaded –, whiteboard –, paper –, final –; line 22: downloaded –, whiteboard –, paper –, final –; line 24: downloaded –, whiteboard –, paper –, final –

## page1_angle_medium_b_IMG_5173.HEIC · original (page 1)

17 boxes · detection 1.62 s · detector path: text large enough (46 px) -> quick boxes kept

| reader | reading | reading, spaces ignored | reading, lines >60 chars | end to end (app order) | end to end (order-free) | found / lines | missed | split | extra boxes | read s |
|---|---|---|---|---|---|---|---|---|---|---|
| downloaded | 23.2 | 22.3 | 41.7 | 69.8 | 71.5 | 12 / 25 | [4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 23, 24] | – | 5 | 3.75 |
| whiteboard | 19.2 | 18.6 | 33.6 | 68.3 | 72.3 | 12 / 25 | [4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 23, 24] | – | 5 | 5.52 |
| paper | 22.5 | 22.2 | 38.2 | 69.5 | 72.5 | 12 / 25 | [4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 23, 24] | – | 5 | 5.42 |
| final | 19.5 | 19.1 | 32.5 | 68.1 | 71.5 | 12 / 25 | [4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 23, 24] | – | 5 | 5.49 |

Long lines (>60 characters): line 3: downloaded 64.7, whiteboard 39.7, paper 44.1, final 33.8; line 5: downloaded –, whiteboard –, paper –, final –; line 6: downloaded –, whiteboard –, paper –, final –; line 10: downloaded 4.6, whiteboard 0.0, paper 13.9, final 1.5; line 11: downloaded 3.1, whiteboard 0.0, paper 1.5, final 0.0; line 16: downloaded –, whiteboard –, paper –, final –; line 19: downloaded –, whiteboard –, paper –, final –; line 20: downloaded –, whiteboard –, paper –, final –; line 22: downloaded 94.5, whiteboard 94.5, paper 93.2, final 94.5; line 24: downloaded –, whiteboard –, paper –, final –

## page1_angle_medium_b_IMG_5173.HEIC · app_prepared (page 1)

7 boxes · detection 0.3 s · detector path: text large enough (27 px) -> quick boxes kept
Preparation: perspective skipped (text or camera information unavailable: abstain); crop applied (Cropped around detected writing.); rotation skipped (insufficient independent text rows); lighting applied (Gentle correction for low contrast, dimness, or uneven light.)

| reader | reading | reading, spaces ignored | reading, lines >60 chars | end to end (app order) | end to end (order-free) | found / lines | missed | split | extra boxes | read s |
|---|---|---|---|---|---|---|---|---|---|---|
| downloaded | 21.7 | 22.0 | 82.6 | 92.4 | 91.3 | 4 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25] | – | 3 | 1.52 |
| whiteboard | 21.4 | 21.5 | 76.8 | 91.9 | 91.4 | 4 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25] | – | 3 | 2.18 |
| paper | 23.1 | 21.1 | 81.2 | 92.5 | 91.4 | 4 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25] | – | 3 | 2.22 |
| final | 22.8 | 23.1 | 78.3 | 92.4 | 91.2 | 4 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25] | – | 3 | 2.25 |

Long lines (>60 characters): line 3: downloaded –, whiteboard –, paper –, final –; line 5: downloaded –, whiteboard –, paper –, final –; line 6: downloaded –, whiteboard –, paper –, final –; line 10: downloaded –, whiteboard –, paper –, final –; line 11: downloaded –, whiteboard –, paper –, final –; line 16: downloaded 82.6, whiteboard 76.8, paper 81.2, final 78.3; line 19: downloaded –, whiteboard –, paper –, final –; line 20: downloaded –, whiteboard –, paper –, final –; line 22: downloaded –, whiteboard –, paper –, final –; line 24: downloaded –, whiteboard –, paper –, final –

## page1_angle_strong_a_IMG_5170.HEIC · original (page 1)

15 boxes · detection 1.41 s · detector path: text large enough (38 px) -> quick boxes kept

| reader | reading | reading, spaces ignored | reading, lines >60 chars | end to end (app order) | end to end (order-free) | found / lines | missed | split | extra boxes | read s |
|---|---|---|---|---|---|---|---|---|---|---|
| downloaded | 20.5 | 20.1 | 26.4 | 73.8 | 75.6 | 9 / 25 | [2, 4, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25] | [3] | 5 | 3.33 |
| whiteboard | 16.5 | 16.2 | 12.2 | 69.5 | 74.8 | 9 / 25 | [2, 4, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25] | [3] | 5 | 4.92 |
| paper | 22.4 | 21.5 | 15.1 | 71.1 | 75.5 | 9 / 25 | [2, 4, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25] | [3] | 5 | 4.97 |
| final | 16.6 | 16.1 | 8.8 | 70.2 | 73.3 | 9 / 25 | [2, 4, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25] | [3] | 5 | 5.02 |

Long lines (>60 characters): line 3: downloaded 26.5, whiteboard 26.5, paper 30.9, final 22.1; line 5: downloaded 52.9, whiteboard 10.0, paper 14.3, final 4.3; line 6: downloaded 0.0, whiteboard 0.0, paper 0.0, final 0.0; line 10: downloaded –, whiteboard –, paper –, final –; line 11: downloaded –, whiteboard –, paper –, final –; line 16: downloaded –, whiteboard –, paper –, final –; line 19: downloaded –, whiteboard –, paper –, final –; line 20: downloaded –, whiteboard –, paper –, final –; line 22: downloaded –, whiteboard –, paper –, final –; line 24: downloaded –, whiteboard –, paper –, final –

## page1_angle_strong_a_IMG_5170.HEIC · app_prepared (page 1)

3 boxes · detection 0.9 s · detector path: text large enough (52 px) -> quick boxes kept
Preparation: perspective skipped (text or camera information unavailable: abstain); crop applied (Cropped around detected writing.); rotation skipped (insufficient independent text rows); lighting skipped (Lighting check did not find a supported correction.)

| reader | reading | reading, spaces ignored | reading, lines >60 chars | end to end (app order) | end to end (order-free) | found / lines | missed | split | extra boxes | read s |
|---|---|---|---|---|---|---|---|---|---|---|
| downloaded | 47.5 | 44.4 | 0.0 | 100.1 | 98.1 | 1 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] | [14] | 1 | 0.68 |
| whiteboard | 40.0 | 38.9 | 0.0 | 100.0 | 97.8 | 1 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] | [14] | 1 | 0.95 |
| paper | 42.5 | 41.7 | 0.0 | 100.0 | 98.0 | 1 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] | [14] | 1 | 0.94 |
| final | 42.5 | 41.7 | 0.0 | 99.8 | 98.1 | 1 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] | [14] | 1 | 0.95 |

Long lines (>60 characters): line 3: downloaded –, whiteboard –, paper –, final –; line 5: downloaded –, whiteboard –, paper –, final –; line 6: downloaded –, whiteboard –, paper –, final –; line 10: downloaded –, whiteboard –, paper –, final –; line 11: downloaded –, whiteboard –, paper –, final –; line 16: downloaded –, whiteboard –, paper –, final –; line 19: downloaded –, whiteboard –, paper –, final –; line 20: downloaded –, whiteboard –, paper –, final –; line 22: downloaded –, whiteboard –, paper –, final –; line 24: downloaded –, whiteboard –, paper –, final –

## page1_angle_strong_b_IMG_5171.HEIC · original (page 1)

16 boxes · detection 1.36 s · detector path: text large enough (38 px) -> quick boxes kept

| reader | reading | reading, spaces ignored | reading, lines >60 chars | end to end (app order) | end to end (order-free) | found / lines | missed | split | extra boxes | read s |
|---|---|---|---|---|---|---|---|---|---|---|
| downloaded | 23.2 | 22.0 | 30.4 | 70.5 | 73.4 | 9 / 25 | [2, 4, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] | [3, 7, 8] | 4 | 3.58 |
| whiteboard | 22.9 | 22.3 | 31.5 | 69.5 | 75.2 | 9 / 25 | [2, 4, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] | [3, 7, 8] | 4 | 5.26 |
| paper | 27.5 | 26.3 | 31.5 | 70.8 | 74.8 | 9 / 25 | [2, 4, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] | [3, 7, 8] | 4 | 5.38 |
| final | 24.7 | 24.4 | 30.0 | 69.2 | 75.3 | 9 / 25 | [2, 4, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] | [3, 7, 8] | 4 | 5.44 |

Long lines (>60 characters): line 3: downloaded 25.0, whiteboard 25.0, paper 27.9, final 25.0; line 5: downloaded 2.9, whiteboard 5.7, paper 5.7, final 2.9; line 6: downloaded 1.5, whiteboard 1.5, paper 0.0, final 0.0; line 10: downloaded 92.3, whiteboard 93.8, paper 92.3, final 92.3; line 11: downloaded –, whiteboard –, paper –, final –; line 16: downloaded –, whiteboard –, paper –, final –; line 19: downloaded –, whiteboard –, paper –, final –; line 20: downloaded –, whiteboard –, paper –, final –; line 22: downloaded –, whiteboard –, paper –, final –; line 24: downloaded –, whiteboard –, paper –, final –

## page1_angle_strong_b_IMG_5171.HEIC · app_prepared (page 1)

5 boxes · detection 0.95 s · detector path: text large enough (50 px) -> quick boxes kept
Preparation: perspective skipped (text or camera information unavailable: abstain); crop applied (Cropped around detected writing.); rotation skipped (insufficient independent text rows); lighting skipped (Lighting check did not find a supported correction.)

| reader | reading | reading, spaces ignored | reading, lines >60 chars | end to end (app order) | end to end (order-free) | found / lines | missed | split | extra boxes | read s |
|---|---|---|---|---|---|---|---|---|---|---|
| downloaded | 87.7 | 85.3 | 98.5 | 100.3 | 98.7 | 3 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] | – | 2 | 1.12 |
| whiteboard | 85.3 | 83.5 | 93.8 | 100.1 | 98.4 | 3 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] | – | 2 | 1.53 |
| paper | 85.7 | 83.0 | 95.4 | 99.9 | 98.5 | 3 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] | – | 2 | 1.59 |
| final | 86.1 | 84.4 | 93.8 | 99.8 | 98.6 | 3 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] | – | 2 | 1.59 |

Long lines (>60 characters): line 3: downloaded –, whiteboard –, paper –, final –; line 5: downloaded –, whiteboard –, paper –, final –; line 6: downloaded –, whiteboard –, paper –, final –; line 10: downloaded –, whiteboard –, paper –, final –; line 11: downloaded 98.5, whiteboard 93.8, paper 95.4, final 93.8; line 16: downloaded –, whiteboard –, paper –, final –; line 19: downloaded –, whiteboard –, paper –, final –; line 20: downloaded –, whiteboard –, paper –, final –; line 22: downloaded –, whiteboard –, paper –, final –; line 24: downloaded –, whiteboard –, paper –, final –

## page1_flat_a_IMG_5168.HEIC · original (page 1)

26 boxes · detection 1.47 s · detector path: text large enough (49 px) -> quick boxes kept

| reader | reading | reading, spaces ignored | reading, lines >60 chars | end to end (app order) | end to end (order-free) | found / lines | missed | split | extra boxes | read s |
|---|---|---|---|---|---|---|---|---|---|---|
| downloaded | 3.7 | 3.5 | 2.7 | 3.8 | 3.7 | 25 / 25 | – | – | 1 | 5.95 |
| whiteboard | 2.7 | 2.5 | 1.3 | 2.6 | 2.6 | 25 / 25 | – | – | 1 | 8.88 |
| paper | 8.1 | 6.9 | 5.3 | 7.7 | 7.6 | 25 / 25 | – | – | 1 | 8.92 |
| final | 3.7 | 3.9 | 1.7 | 3.5 | 3.4 | 25 / 25 | – | – | 1 | 9.17 |

Long lines (>60 characters): line 3: downloaded 0.0, whiteboard 0.0, paper 1.5, final 0.0; line 5: downloaded 4.3, whiteboard 1.4, paper 4.3, final 0.0; line 6: downloaded 0.0, whiteboard 1.5, paper 0.0, final 0.0; line 10: downloaded 1.5, whiteboard 0.0, paper 15.4, final 1.5; line 11: downloaded 0.0, whiteboard 0.0, paper 3.1, final 1.5; line 16: downloaded 5.8, whiteboard 1.5, paper 2.9, final 1.5; line 19: downloaded 4.1, whiteboard 2.7, paper 5.5, final 4.1; line 20: downloaded 1.5, whiteboard 0.0, paper 9.1, final 1.5; line 22: downloaded 2.7, whiteboard 2.7, paper 5.5, final 4.1; line 24: downloaded 6.9, whiteboard 2.7, paper 5.5, final 2.7

## page1_flat_a_IMG_5168.HEIC · app_prepared (page 1)

3 boxes · detection 0.25 s · detector path: text large enough (63 px) -> quick boxes kept
Preparation: perspective skipped (text or camera information unavailable: abstain); crop applied (Cropped around detected writing.); rotation skipped (insufficient independent text rows); lighting applied (Gentle correction for low contrast, dimness, or uneven light.)

| reader | reading | reading, spaces ignored | reading, lines >60 chars | end to end (app order) | end to end (order-free) | found / lines | missed | split | extra boxes | read s |
|---|---|---|---|---|---|---|---|---|---|---|
| downloaded | 1.0 | 1.2 | 1.5 | 88.2 | 86.2 | 3 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] | – | 0 | 0.7 |
| whiteboard | 0.5 | 0.6 | 0.8 | 88.1 | 86.1 | 3 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] | – | 0 | 1.17 |
| paper | 6.1 | 4.8 | 6.9 | 88.7 | 86.9 | 3 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] | – | 0 | 1.16 |
| final | 3.0 | 3.6 | 2.3 | 88.4 | 86.4 | 3 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] | – | 0 | 1.21 |

Long lines (>60 characters): line 3: downloaded –, whiteboard –, paper –, final –; line 5: downloaded –, whiteboard –, paper –, final –; line 6: downloaded –, whiteboard –, paper –, final –; line 10: downloaded 1.5, whiteboard 1.5, paper 12.3, final 4.6; line 11: downloaded 1.5, whiteboard 0.0, paper 1.5, final 0.0; line 16: downloaded –, whiteboard –, paper –, final –; line 19: downloaded –, whiteboard –, paper –, final –; line 20: downloaded –, whiteboard –, paper –, final –; line 22: downloaded –, whiteboard –, paper –, final –; line 24: downloaded –, whiteboard –, paper –, final –

## page1_flat_b_IMG_5169.HEIC · original (page 1)

27 boxes · detection 1.48 s · detector path: text large enough (49 px) -> quick boxes kept

| reader | reading | reading, spaces ignored | reading, lines >60 chars | end to end (app order) | end to end (order-free) | found / lines | missed | split | extra boxes | read s |
|---|---|---|---|---|---|---|---|---|---|---|
| downloaded | 3.9 | 3.5 | 3.3 | 4.2 | 5.5 | 24 / 25 | [4] | [8] | 2 | 6.16 |
| whiteboard | 3.8 | 3.8 | 1.4 | 3.6 | 4.9 | 24 / 25 | [4] | [8] | 2 | 9.35 |
| paper | 8.1 | 7.3 | 5.6 | 7.8 | 9.1 | 24 / 25 | [4] | [8] | 2 | 9.36 |
| final | 4.1 | 4.4 | 2.2 | 4.1 | 5.4 | 24 / 25 | [4] | [8] | 2 | 9.53 |

Long lines (>60 characters): line 3: downloaded 1.5, whiteboard 0.0, paper 1.5, final 0.0; line 5: downloaded 4.3, whiteboard 1.4, paper 5.7, final 0.0; line 6: downloaded 0.0, whiteboard 1.5, paper 3.0, final 0.0; line 10: downloaded 3.1, whiteboard 0.0, paper 13.9, final 3.1; line 11: downloaded 1.5, whiteboard 0.0, paper 0.0, final 0.0; line 16: downloaded 11.6, whiteboard 1.5, paper 4.3, final 1.5; line 19: downloaded 1.4, whiteboard 4.1, paper 4.1, final 5.5; line 20: downloaded 1.5, whiteboard 0.0, paper 6.1, final 1.5; line 22: downloaded 2.7, whiteboard 2.7, paper 9.6, final 5.5; line 24: downloaded 5.5, whiteboard 2.7, paper 8.2, final 5.5

## page1_flat_b_IMG_5169.HEIC · app_prepared (page 1)

3 boxes · detection 0.25 s · detector path: text large enough (63 px) -> quick boxes kept
Preparation: perspective skipped (text or camera information unavailable: abstain); crop applied (Cropped around detected writing.); rotation skipped (insufficient independent text rows); lighting applied (Gentle correction for low contrast, dimness, or uneven light.)

| reader | reading | reading, spaces ignored | reading, lines >60 chars | end to end (app order) | end to end (order-free) | found / lines | missed | split | extra boxes | read s |
|---|---|---|---|---|---|---|---|---|---|---|
| downloaded | 1.0 | 1.2 | 1.5 | 88.1 | 86.2 | 3 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] | – | 0 | 0.71 |
| whiteboard | 0.0 | 0.0 | 0.0 | 88.0 | 86.0 | 3 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] | – | 0 | 1.16 |
| paper | 4.6 | 4.9 | 6.9 | 88.8 | 86.8 | 3 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] | – | 0 | 1.25 |
| final | 2.5 | 3.0 | 1.5 | 88.3 | 86.3 | 3 / 25 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] | – | 0 | 1.18 |

Long lines (>60 characters): line 3: downloaded –, whiteboard –, paper –, final –; line 5: downloaded –, whiteboard –, paper –, final –; line 6: downloaded –, whiteboard –, paper –, final –; line 10: downloaded 3.1, whiteboard 0.0, paper 13.9, final 3.1; line 11: downloaded 0.0, whiteboard 0.0, paper 0.0, final 0.0; line 16: downloaded –, whiteboard –, paper –, final –; line 19: downloaded –, whiteboard –, paper –, final –; line 20: downloaded –, whiteboard –, paper –, final –; line 22: downloaded –, whiteboard –, paper –, final –; line 24: downloaded –, whiteboard –, paper –, final –
