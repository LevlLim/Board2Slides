# Server reader, S1 recipe, on the teammate's leak-fixed 80:10:10 split

Experiment `server_his_20261110`. Data: TYW02/whiteboard-notes-v1 @ 75619db2, split by his_data/prep_his_data.py (groups of the same writing never split; stratified by source and line length). Best checkpoint chosen on HIS validation; HIS test scored once. Not comparable with our-data tables (different training and evaluation data); his pool contains our evaluation crops, so no our-set scores.

| run | what |
|---|---|
| pretrained | PP-OCRv5_server_rec as downloaded |
| S1his | server S1 recipe on the teammate's leak-fixed 80:10:10 train (incl. his augmented copies) |

| model | set | crops | mean line CER | corpus CER | exact |
|---|---|---|---|---|---|
| pretrained | his val | 179 | 0.1824 | 0.1819 | 0.352 |
| S1his | his val | 179 | 0.0621 | 0.0637 | 0.575 |
| pretrained | his test | 169 | 0.1690 | 0.1771 | 0.308 |
| S1his | his test | 169 | 0.0704 | 0.0774 | 0.544 |

By label length (mean line CER):

| set | length | pretrained | S1his |
|---|---|---|---|
| his val | 1-20 chars (99) | 0.189 | 0.069 |
| his val | 21-40 chars (61) | 0.175 | 0.048 |
| his val | over 40 chars (19) | 0.173 | 0.069 |
| his test | 1-20 chars (92) | 0.158 | 0.062 |
| his test | 21-40 chars (60) | 0.167 | 0.076 |
| his test | over 40 chars (17) | 0.238 | 0.099 |

| comparison | set | CER difference | 95% interval | verdict |
|---|---|---|---|---|
| S1his vs pretrained (fine-tuned vs no fine-tune) | his val | -0.1202 | -0.1638 .. -0.0783 | better |
| S1his vs pretrained (fine-tuned vs no fine-tune) | his test | -0.0986 | -0.1296 .. -0.0675 | better |

## Training runs

| run | lines | epochs | updates | minutes | best validation |
|---|---|---|---|---|---|
| S1his | 8493 | 11 | 5841 | 20.6 | norm_edit_dis: 0.9332550740614023, is_float16: False, acc: 0.620111697 |

Lines left out by the shared label limits (98 characters, 128 positions): {'train_S1his': {'label longer than the limit': 15}, 'his_val': {}, 'his_test': {}}.
Limits: one seed; his test is 169 crops (intervals resample whole board photos / items).
