# Final-model recipe with a 94-character English dictionary (S1his_en) vs the 10 Nov final model (S1his_fulldict)

Same crops for every model: his-test and his-val lines whose labels use only the 94 characters (+ space). S1his_fulldict = the 10 Nov final model (full 18,383-character dictionary), unchanged, read here.

Experiment `server_his_ensl_20261119`. Data: TYW02/whiteboard-notes-v1 @ 75619db2, split by his_data/prep_his_data.py (groups of the same writing never split; stratified by source and line length). Best checkpoint chosen on HIS validation; HIS test scored once. Not comparable with our-data tables (different training and evaluation data); his pool contains our evaluation crops, so no our-set scores.

| run | what |
|---|---|
| pretrained | PP-OCRv5_server_rec as downloaded |
| S1his_ensl | as S1his_en, but the output layers start from the pretrained rows of the same characters |
| S1his_fulldict | the 10 Nov final model (S1his), full dictionary, read on the same crops |
| pretrained_sliced | the downloaded reader cut to the 94 characters, not trained (slicing check) |

| model | set | crops | mean line CER | corpus CER | exact |
|---|---|---|---|---|---|
| pretrained | his val | 179 | 0.1824 | 0.1819 | 0.352 |
| pretrained_sliced | his val | 179 | 0.1817 | 0.1811 | 0.352 |
| S1his_fulldict | his val | 179 | 0.0621 | 0.0637 | 0.575 |
| S1his_ensl | his val | 179 | 0.0509 | 0.0545 | 0.570 |
| pretrained | his test | 164 | 0.1692 | 0.1776 | 0.317 |
| pretrained_sliced | his test | 164 | 0.1658 | 0.1755 | 0.329 |
| S1his_fulldict | his test | 164 | 0.0696 | 0.0770 | 0.561 |
| S1his_ensl | his test | 164 | 0.0742 | 0.0824 | 0.616 |

By label length (mean line CER):

| set | length | pretrained | pretrained_sliced | S1his_fulldict | S1his_ensl |
|---|---|---|---|---|---|
| his val | 1-20 chars (99) | 0.189 | 0.189 | 0.069 | 0.056 |
| his val | 21-40 chars (61) | 0.175 | 0.173 | 0.048 | 0.039 |
| his val | over 40 chars (19) | 0.173 | 0.173 | 0.069 | 0.060 |
| his test | 1-20 chars (87) | 0.157 | 0.152 | 0.060 | 0.064 |
| his test | 21-40 chars (60) | 0.167 | 0.165 | 0.076 | 0.079 |
| his test | over 40 chars (17) | 0.238 | 0.238 | 0.099 | 0.110 |

| comparison | set | CER difference | 95% interval | verdict |
|---|---|---|---|---|
| S1his_ensl vs pretrained (fine-tuned (94 characters) vs no fine-tune) | his val | -0.1315 | -0.1810 .. -0.0863 | better |
| S1his_ensl vs pretrained (fine-tuned (94 characters) vs no fine-tune) | his test | -0.0950 | -0.1252 .. -0.0635 | better |
| S1his_ensl vs S1his_fulldict (94-character vs full dictionary, same crops) | his val | -0.0113 | -0.0202 .. -0.0018 | better |
| S1his_ensl vs S1his_fulldict (94-character vs full dictionary, same crops) | his test | +0.0046 | -0.0039 .. +0.0123 | no clear difference |

## Training runs

| run | lines | epochs | updates | minutes | best validation |
|---|---|---|---|---|---|
| S1his_ensl | 8448 | 11 | 5808 | 11.6 | norm_edit_dis: 0.9363767617898336, is_float16: False, acc: 0.597765329 |

Lines left out by the shared label limits (98 characters, 128 positions): {'train_S1his_ensl': {'character not in the dictionary': 45, 'label longer than the limit': 15}, 'his_val': {}, 'his_test': {'character not in the dictionary': 5}}.
Limits: one seed each; the two dictionaries differ in the heads' starting weights as well as in size (see source docstring).
