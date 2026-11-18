# Preprocessing and super-resolution

| Folder / file | What it is |
|---|---|
| `preprocessing_package/` | The preprocessing code: board geometry and perspective (`board_geometry.py`), rotation (`fix_rotation.py`, `rotation_consensus.py`), lighting (`fix_lighting.py`, `lighting_conservative.py`), blemish removal, text-line grouping and the team pipeline. Copy of `Assignment/whiteboard-ocr/preprocessing/`. |
| `notebooks/` | 07 preprocessing pipeline, 07a run-order check, 08 preprocessing comparison (decision signals, visual examples), 09 manual vs automatic geometry, section 8 auto-rectify cells |
| `docs/` | Perspective revision, preprocessing audit (30 Oct), rotation rule review (29 Oct), capture-quality assistant handoff, team pipeline README |
| `super_resolution/` | Week 8 super-resolution work and `SUPER_RESOLUTION_RUNS.md` (the full SR and multi-photo fusion record) |

Summary of the decisions (from PROJECT_PROGRESS section 4): perspective correction helps some side views but can
hurt others, so no blanket correction; crop enlargement is kept separate from learned SR; learned SR showed no
reliable OCR benefit (sharper did not mean lower CER) and is paused; multi-photo selection was promising
(12.8% to 8.0% with three photos) but needs automatic line matching.
