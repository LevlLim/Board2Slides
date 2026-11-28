# 8. English-only reader: the final recipe with a 94-character dictionary (19 Nov 2026)

Question: can the same training recipe give a smaller reader that only outputs English characters?

The downloaded reader outputs 18,383 characters (mostly Chinese). Here only the dictionary changes: `en_dict.txt`
(digits, letters, ASCII symbols) plus space. Every other setting is the final model's (reader 08: 64x1024 input,
width pooling, stage-wise learning rate, teammate dataset `TYW02/whiteboard-notes-v1` @ 75619db2). Only the heads'
character layers change size (CTC output layer, NRTR character embedding and output layer); the backbone and neck
are the same.

Error = mean line character error rate on the teammate's test split, using the 164 test lines whose labels fit the
94 characters (all models scored on the same lines).

| Folder | What differs | Test error |
|---|---|---|
| `results_server_his_en_20261119` | Output layers start from scratch, 11 epochs | 9.4% |
| `results_server_his_en12k_20261119` | Same, 23 epochs | 8.2% |
| `results_server_his_ensl_20261119` | **Shipped.** Output layers start from the downloaded reader's rows for the same 94 characters, 11 epochs | **7.4%** |
| Reference: final model 08, full dictionary, same lines | | 7.0% |
| Reference: downloaded reader, same lines | | 16.9% |

- Shipped English reader vs full-dictionary final model: +0.5 points on test (95% interval -0.4 to +1.2), so no
  clear difference; on validation it is better (5.1% vs 6.2%).
- Slicing check before training: the cut but untrained reader reads like the downloaded one (test 16.6% vs 16.9%,
  validation 18.2% vs 18.2%), so the right rows were kept.
- Weight file: 75.5 MB instead of 84.4 MB. It cannot output Chinese or any character outside the 94.
- The shipped reader is the app's V2 (`models/ocr/recognizer/inference.pdiparams`, SHA-256 615ca77e…, identical to
  the export of `S1his_ensl`). Its weights are also on Release `Reader_V1` as `board2notes_reader_en94.zip`.

Each folder: `source/` (training script `b2n_server_his_en.py`, the shared `b2n_versions.py`, `en_dict.txt`),
`results/report.md`, `summary.json`, `predictions.csv`, `experiment.json` (hashes of data, script and pretrained
weights) and `runs/<run>/` (config, training and export logs, scores). Weights are left out.
