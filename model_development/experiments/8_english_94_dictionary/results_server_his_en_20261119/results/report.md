# Final-model recipe with a 94-character English dictionary (S1his_en) vs the 10 Nov final model (S1his_fulldict)

Same crops for every model: his-test and his-val lines whose labels use only the 94 characters (+ space). S1his_fulldict = the 10 Nov final model (full 18,383-character dictionary), unchanged, read here.

Experiment `server_his_en_20261119`. Data: TYW02/whiteboard-notes-v1 @ 75619db2, split by his_data/prep_his_data.py (groups of the same writing never split; stratified by source and line length). Best checkpoint chosen on HIS validation; HIS test scored once. Not comparable with our-data tables (different training and evaluation data); his pool contains our evaluation crops, so no our-set scores.

| run | what |
|---|---|
| pretrained | PP-OCRv5_server_rec as downloaded |
| S1his_en | final-model recipe (server S1 on the teammate's train incl. his copies), 94-character English dictionary |
| S1his_fulldict | the 10 Nov final model (S1his), full dictionary, read on the same crops |

| model | set | crops | mean line CER | corpus CER | exact |
|---|---|---|---|---|---|
| pretrained | his val | 179 | 0.1824 | 0.1819 | 0.352 |
| S1his_fulldict | his val | 179 | 0.0621 | 0.0637 | 0.575 |
| S1his_en | his val | 179 | 0.0768 | 0.0738 | 0.520 |
| pretrained | his test | 164 | 0.1692 | 0.1776 | 0.317 |
| S1his_fulldict | his test | 164 | 0.0696 | 0.0770 | 0.561 |
| S1his_en | his test | 164 | 0.0935 | 0.0997 | 0.500 |

By label length (mean line CER):

| set | length | pretrained | S1his_fulldict | S1his_en |
|---|---|---|---|---|
| his val | 1-20 chars (99) | 0.189 | 0.069 | 0.090 |
| his val | 21-40 chars (61) | 0.175 | 0.048 | 0.054 |
| his val | over 40 chars (19) | 0.173 | 0.069 | 0.078 |
| his test | 1-20 chars (87) | 0.157 | 0.060 | 0.091 |
| his test | 21-40 chars (60) | 0.167 | 0.076 | 0.087 |
| his test | over 40 chars (17) | 0.238 | 0.099 | 0.128 |

| comparison | set | CER difference | 95% interval | verdict |
|---|---|---|---|---|
| S1his_en vs pretrained (fine-tuned (94 characters) vs no fine-tune) | his val | -0.1056 | -0.1499 .. -0.0616 | better |
| S1his_en vs pretrained (fine-tuned (94 characters) vs no fine-tune) | his test | -0.0756 | -0.1020 .. -0.0496 | better |
| S1his_en vs S1his_fulldict (94-character vs full dictionary, same recipe, same crops) | his val | +0.0147 | +0.0047 .. +0.0292 | worse |
| S1his_en vs S1his_fulldict (94-character vs full dictionary, same recipe, same crops) | his test | +0.0240 | +0.0098 .. +0.0351 | worse |

## Training runs

| run | lines | epochs | updates | minutes | best validation |
|---|---|---|---|---|---|
| S1his_en | 8448 | 11 | 5808 | 11.7 | norm_edit_dis: 0.9171988610558605, is_float16: False, acc: 0.541899411 |

Lines left out by the shared label limits (98 characters, 128 positions): {'train_S1his_en': {'character not in the dictionary': 45, 'label longer than the limit': 15}, 'his_val': {}, 'his_test': {'character not in the dictionary': 5}}.
Limits: one seed each; the two dictionaries differ in the heads' starting weights as well as in size (see source docstring).
