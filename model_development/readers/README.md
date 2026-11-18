# Server readers from model development

Every PP-OCRv5 server recogniser (`PP-OCRv5_server_rec`) exported during model development, one folder per step of the
model-development slides. Each folder here holds the small files only:

- `inference.yml`, `inference.json`: the model definition PaddleOCR loads (input size, dictionary, network).
- `verified.json`: the export check. The exported model reads 30 fixed crops identically to its training checkpoint.
- `SOURCE.txt`: what changed, which experiment trained it, and the SHA-256 of its weight file.

The weight files (`inference.pdiparams`, 84 MB each) are attached to the GitHub Release **readers-v1**
(<https://github.com/LevlLim/Board2Slides/releases/tag/readers-v1>) as two zips:

- `board2notes_final_reader.zip` (76 MB): the final model, folder 08 only.
- `board2notes_all_readers.zip` (934 MB): all 11 folders below.

Each folder in a zip is a complete, loadable model. The final model (08) is step 7's settings trained on the teammate's
larger dataset; it is the reader the app should use.

Error % = mean line character error rate, lower is better, case-insensitive, spaces collapsed. One training run per
model; steps 6 and 7 were trained twice and the slides use the two-run averages.

| Folder | What changed | Whiteboard | CNN board | Long lines | IAM paper | Teammate test |
|---|---|---|---|---|---|---|
| `00_downloaded_PP-OCRv5_server_rec` | PaddleOCR's reader as downloaded (Hugging Face PaddlePaddle/PP-OCRv5_server_rec @ b26c3587, Apache-2.0) | 15.5 | 45.8 | 30.9 | 12.2 | 16.9 |
| `01_finetuned_on_our_whiteboard_lines` | Fine-tuned on our 1,113 whiteboard line crops, PaddleOCR's original setup (48x320 input, 40 CTC columns) | 8.9 | 29.0 | 7.8 | 30.6 | – |
| `02_plus_extra_augmented_images` | 01 plus 3,156 extra augmented images made before training. No clear gain | 9.3 | 28.7 | 8.8 | 31.4 | – |
| `03_wide_input_still_40_columns_BROKEN` | Wider input (64x1024) but still 40 CTC columns: the reader breaks. Control for 04 | 54.5 | 93.9 | 45.3 | 61.0 | – |
| `04_width_pooling_one_column_per_8px` | Same lines and input as 03; only the pooling differs: one CTC column per 8 px (128 at 1,024 px) | 6.4 | 26.6 | 4.5 | 21.7 | – |
| `05_plus_long_lines` | 04 plus the long lines that need more than 40 columns | 7.2 | 24.3 | 2.7 | 25.7 | – |
| `06_control_for_stagewise_lr` | Control: our lines incl. long lines, no extra augmented images, minus 64 teammate lines kept out for the new-writer test. Run 1 of 2 | 7.5 | 26.8 | 2.3 | 24.2 | – |
| `07_plus_stagewise_learning_rate` | 06 plus stage-wise learning rates [0.1, 0.1, 0.25, 0.5, 1.0] (early layers learn 10x slower). Run 1 of 2 | 6.8 | 24.3 | 2.5 | 22.5 | – |
| `08_final_stagewise_lr_on_teammate_dataset` | **Final model**: 07's settings on the teammate's larger, leak-checked 80:10:10 dataset | – | – | – | – | 7.0 |
| `extra_paper_notes_reader_IAM` | Scalability: same training framework, trained on IAM paper handwriting only | 16.8 | 49.8 | 5.8 | 7.8 | – |
| `extra_whiteboard_reader_same_settings_as_paper` | Scalability control: our whiteboard lines, same settings as the paper reader | 7.3 | 23.7 | 2.4 | 23.3 | – |

Notes:
- 00-02 take 48 px tall input; 03-08 and the extras take 64 px tall input padded to at least 1,024 px (see each
  `inference.yml`). We read 00 at 48 px padded to 1,024 px; PaddleOCR's default resize reads long lines worse.
- All folders keep PaddleOCR's 18,383-character dictionary; fine-tuning used English lines only.
- 08 is scored only on the teammate's held-out 169-line test split: his dataset contains most of our validation and
  CNN-board crops, so 08's scores on our sets would not be held out.
