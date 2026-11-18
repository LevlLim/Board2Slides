# Model development: the reader (Khor Zhao Bao)

Everything behind the model-development and scalability slides of our talk: training code, every run's settings,
every scored reading, the notebooks that build the tables, and where each number on the slides comes from.

My part of the project: fine-tuning PaddleOCR's server text reader (`PP-OCRv5_server_rec`) for whiteboard handwriting.
About 50 training runs, each changing one thing against a control; strategy tests were run twice.

Error % on every slide = mean line character error rate (CER), lower is better, case-insensitive, spaces collapsed.
All runs use PaddleOCR's built-in augmentation (`RecAug` in each `config.yml`).

## Where each slide's numbers come from

| Slide | Claim | Evidence (paths from this folder) |
|---|---|---|
| 1 Overview | Steps 0-5: 15.5 / 8.9 / 9.3 / 54.5 / 6.4 / 7.2 on whiteboard, with CNN board and long lines | `experiments/1_data_and_pooling_ladder/results_server_c0_20261111/results/report.md`; table in `notebooks/18_server_reader_progression.ipynb` |
| 1 Overview | Steps 6-7 (2 runs each): 7.5 → 7.1, 25.5 → 24.6, 2.4 → 2.5 | `experiments/2_training_strategies_S0-S3/results_strategies_s0s3/results/report.md` (S0avg, S1avg) |
| 1 Overview | Final model: teammate test 16.9% → 7.0% | `experiments/6_teammate_dataset_S1his/results_server_his_20261110/results/report.md` |
| 1 Overview | About 50 training runs | `experiments/*/results_*/runs/` (28 server, 21+ mobile) |
| 2 Problem: long lines | A 59-letter line gets only 44 columns | `notebooks/19_ctc_columns_and_app_ocr.ipynb` |
| 2 Problem: long lines | Training gives 40 columns; a held-out sentence needs 69 | `notebooks/10_training_results_visual_explorer.ipynb` (minimum positions) |
| 3 Fix: width pooling | 40 columns vs width pooling: 54.5 → 6.4, 93.9 → 26.6, 45.3 → 4.5 | `experiments/1_data_and_pooling_ladder/results_server_c0_20261111/results/report.md` (P40, P128) |
| 3 Fix: width pooling | Against the original setup: 9.3 → 6.4 (-2.9 [-4.9, -1.7]), long lines 8.8 → 4.5 | same report (C1 vs P128) |
| 3 Fix: width pooling | The sentence read by both models | `experiments/1_data_and_pooling_ladder/results_versions_2x2/results/predictions.csv` (P40, P128, `IMG_4944__network_04`) |
| 4 Problem: new writers | Trained writers 7.5% vs new writers 26.8% (control, 2 runs) | `experiments/2_training_strategies_S0-S3/results_strategies_s0s3/results/predictions.csv` (S0, S0s2) |
| 4 Problem: new writers | PowerPac read as Powoerac / Powverpac | same file, `newwriter__images__jhw__train__JH2.1.jpg` |
| 5 Fix: stage-wise LR | Multipliers 0.1, 0.1, 0.25, 0.5, 1.0 | `experiments/2_training_strategies_S0-S3/results_strategies_s0s3/runs/S1/config.yml` (`lr_mult_list`) |
| 5 Fix: stage-wise LR | 7.5 → 7.1; new writers 26.8 → 24.9, better in both runs | `experiments/2_training_strategies_S0-S3/results_strategies_s0s3/results/report.md` |
| 5 Fix: stage-wise LR | Second new-writer set: 7.3 → 5.5, both runs, [-3.0, -1.0] | `experiments/7_second_new_writer_set_Jy/report.md` |
| 5 Fix: stage-wise LR | Fresh boards 12.4 → 11.5 | `../app_diagnosis/fresh_board_test/scores_server_models.csv` |
| 6 Trade-off: CTC trains backbone | First test: new writers 26.8 → 22.8 | `experiments/3_guidance_off_G0/results_guide_vs_s0/` |
| 6 Trade-off: CTC trains backbone | First test: fresh boards 12.4 → 9.1 | `../app_diagnosis/fresh_board_test/scores_server_models.csv` (G0, G0s2) |
| 6 Trade-off: CTC trains backbone | Fair retest at the same epochs: 26.7 → 25.4, 11.2 → 12.3, same direction in both runs | `experiments/4_fair_test_and_gradient_routing/results_routing_runs_20261112/ANALYSIS_per_seed_20261113.md` and `results/report.md` ; per-checkpoint scores in `runs/<run>/epochs.tar.gz` |
| 7 Paper notes | Final model 3.5% on a paper page, 25/25 lines; downloaded reader 3.8% | `scalability_paper_notes/outputs/summary.md` |
| 7 Paper notes | IAM-trained reader: 12.2% → 7.8% on IAM | `readers/README.md`; `experiments/1_data_and_pooling_ladder/results_versions_2x2/results/report.md` (V1b) |

## Folders

| Folder | What is in it |
|---|---|
| `experiments/` | One folder per question. Each `results_*` has `results/report.md`, `summary.json`, `predictions.csv` (every reading), `runs/<run>/config.yml` and training logs, and `source/` (the exact training and scoring scripts) |
| `experiments/5_mobile_reader/` | The same ideas on the smaller mobile reader (pooling repeated: 39.6 → 9.2 on whiteboard) |
| `notebooks/` | 00 overview; 18 every model in one table; 10 training results; 19 CTC columns on a real crop; 13 new-writer review; 17 resize height |
| `readers/` | The 11 exported server readers, one folder per step (weights on the Release) |
| `scalability_paper_notes/` | The paper-notes test: tool, reference text, results, photo overlays of my own notes |
| `scripts/` | Scripts that build notebooks 18 and 19 |
| `PROGRESS_LOG.md` | Dated record of every step, decision and result |

Not included: training datasets and photos (other people's handwriting), checkpoints, and model weights (on the
Release). The work was done in my own repository (`zbkhor/whiteboard-ocr`) and on RunPod, and was copied here on
19 Nov 2026; the dates of each run are in `PROGRESS_LOG.md` and each run's `runs/<run>/started.json`.

Related folders: `../preprocessing_and_super_resolution/` (photo preparation and the super-resolution study) and
`../app_diagnosis/` (OCR app checks, detector and crop diagnostics, the fresh-board test).
