# Detector size sweep and rule C (12-13 Nov 2026)

Question: at what image size should the text detector run so that lines come out as one whole box, for big,
medium and small writing? 511 labelled lines on 67 photos (fresh boards, JH, Jy), plus zoomed-out copies
(photo shrunk and padded) to stand in for small writing. Detector sizes 640-4000 px and the current app path,
every box read by the S1 reader. Scripts: build_refs.py, detect.py, score.py, analyse.py, rules.py.

## Band: line height as the detector sees it (analysis_S1.md)

Lines 32-96 px tall at detection size come out as one whole box 99-100% of the time (CER 6-7%). Below about
16 px lines are missed; above 128 px they split into letter pieces.

## Rule C (implemented in text_lines_with_rule_C.py, a copy of the app file; the app itself is NOT changed)

Measure line height with a quick 960 px pass. If lines are taller than 128 px in the full photo, detect at the
size that makes them about 64 px. Otherwise keep the app path unchanged. B2N_DET_SIZE_RULE=app_v1 gives the old
behaviour. Insensitive to the threshold (96-200 px) and target (48-96 px) in rules.py.

## Real code on all 114 photo versions (ruleC_results.md, S1 reader, mean line CER %)

| Photos | App | 960 for all | Rule C |
|---|---|---|---|
| Fresh big words | 36.0 | 12.0 | 12.0 |
| Fresh sentences | 9.3 | 7.2 | 9.1 |
| JH sentences | 0.9 | 1.0 | 0.9 |
| Jy words | 6.2 | 8.0 | 6.2 |
| Zoomed-out sets (small writing) | unchanged | up to 46.1 | unchanged |
| Mean of the 8 sets | 10.8 | 18.8 | 7.7 |

Rule C switched on for 42 of 114 versions; on all 72 others the boxes are identical to the app's.
Thresholds were chosen on these photos; the fair confirmation is the frozen fresh-photo set, not collected yet.

old_ruleC_unitbug/: first run of the implementation, which scaled box heights twice (boxes are already in
full-photo pixels), so the rule fired on 82 of 114 and hurt zoomed sets. Kept as a record; fixed before the
results above. eval_ruleC.py's pre-filter had the same old formula; it only chose which photos to re-detect
(a superset), so the results are unaffected; the labels in det_ruleC.json come from the app code path.

layers.py, buildmap.py, mapverify.py, layers_out/: the earlier weight-change measurement and the exported-
parameter name map (verified; unique.pdparams, 214 MB, not copied).
