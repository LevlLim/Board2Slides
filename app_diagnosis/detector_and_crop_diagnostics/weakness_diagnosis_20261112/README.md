# Weakness diagnosis, 12 Nov (no training)

Readers: server S0, S0s2 (guidance on), G0, G0s2 (guidance off), S1his. Read on CPU with the app's recognition runtime.
Stretch = widen the crop horizontally by 1.0 / 1.5 / 2.0 before reading (height unchanged), which gives the CTC head more positions.
"Room" = CTC positions / positions the label needs (characters + doubled letters). Below 1 the head cannot output the whole line.

## Line CER (%) by room, stretch 1.0 / 1.5 / 2.0

New-writer set (160 crops; 2-seed means):

| Reader | All | Room < 1 (30) | 1-1.5 | >= 1.5 |
|---|---|---|---|---|
| S0 | 26.8 / 11.7 / 9.8 | 51.2 / 21.7 / 9.4 | 23.4 / 7.6 / 6.7 | 19.1 / 10.9 / 12.7 |
| G0 | 22.7 / 9.8 / 7.2 | 49.0 / 17.2 / 5.7 | 20.2 / 6.2 / 5.2 | 13.4 / 9.8 / 9.8 |

Fresh boards (152 tightened crops; no crop has room < 1):

| Reader | All | 1-1.5 (11) | >= 1.5 (141) |
|---|---|---|---|
| S0 | 12.4 / 15.5 / 23.6 | 10.9 / 2.4 / 0.7 | 12.5 / 16.6 / 25.4 |
| G0 | 9.1 / 12.7 / 22.3 | 11.1 / 3.4 / 0.7 | 9.0 / 13.4 / 24.0 |
| S1his | 4.6 / 4.2 / 6.1 | 10.5 / 2.6 / 1.8 | 4.1 / 4.3 / 6.5 |

## Crop geometry (median)

| Set | Room | Ink height / crop height | Char width / ink height |
|---|---|---|---|
| New writers | 1.39 | 0.36 | 0.44 |
| Fresh boards | 2.69 | 0.75 | 0.46 |

New-writer crops with room < 1: ink 0.41 of height, char width / ink height 0.24 (narrow, cramped writing).

## Error types, new writers (per 100 reference characters, 2-seed sums)

S0: substitutions 11.0, deletions 12.9, insertions 1.2, spaces 3.1.
G0: substitutions 7.6 (-31%), deletions 14.7, insertions 0.7, spaces 2.4.
Errors rise along the line (about 20% at the start to 37% at the end) for new writers; flat on familiar val.
About 88% of wrong words are non-words.

## Check 1 (12 Nov afternoon): the app's own line detection on the fresh photos (`app_crops/`)

Run the app's real pipeline on the 44 fresh photos: detector PP-OCRv5_mobile_det, unclip 1.2, quick pass at 1600 px,
re-detect at full resolution when the writing looks small, line grouping, quad crop. Match its boxes to the 152
labelled lines by overlap. Read with the same readers. Line CER (%), same lines and labels as the tight crops:

| Reader | Tight crops (hand-checked) | App crops | App crops, detector at 960 px |
|---|---|---|---|
| S0 (2-seed mean) | 12.4 | 31.2 | not read |
| G0 (2-seed mean) | 9.1 | 26.6 | 9.6 |
| S1his | 4.6 | 28.9 | 5.3 |

How the app's boxes matched the labelled lines:

| Detector | One box per line | Line split into 2-4 boxes | Merged with another line |
|---|---|---|---|
| App as is | 115 | 35 | 2 |
| At 960 px | 151 | 1 | 0 |

By board (G0, app as is / tight): A 32.3 / 10.1, B 56.5 / 11.3, C 31.7 / 5.2, D 12.5 / 9.4, E 8.0 / 6.1.
Boards A-C are single large words written with a thin pen (text 330-540 px tall in the photo). D-E are sentences
(about 150 px).

Why (`quick.py`, `app_vs_labelled.jpg`, red = app boxes, green = labelled lines):
- On large thin-stroke words, the 1600 px quick pass finds only small pieces of letters, with a median height of 9 px.
- The app's rule then reads them as "small text" and re-detects at full resolution. There the letters break into even
  more pieces (IMG_5095: 18 boxes for 6 words).
- At 960 px the same photo gives exactly 6 lines.
- One box per line is not always a whole line: some "one box" matches cover only one letter ("letter" read as "V").

Caveats:
- The 960 px setting was found on this set, so these photos are development data for it.
- It must be checked on photos with small writing, where the full-resolution re-detect was designed to help: the frozen
  fresh-photo confirmation and the JH full photos.
- Only then can it go into the app.
