import json, csv
from pathlib import Path
U = Path("/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment")
FR = Path('/tmp/claude-0/-home-claude/02f3df2f-1dbb-533d-b187-71169cf31ef0/scratchpad/unf/board2notes_freshwriter_test_20261111')
refs = []
for r in csv.DictReader(open(FR/'manifest.csv', encoding='utf-8')):
    if r['included'] == 'yes':
        refs.append(dict(set='fresh', group='fresh '+('big words' if r['board'] in 'ABC' else 'sentences'), photo=str(U/'1111_unfamiliar'/f"{r['photo']}.HEIC"), pid=r['photo'], id=r['id'], writing=r['writing'], text=r['label'], quad=json.loads(r['quad'])))
lab = {r['path'].split('/',1)[1]: r['label'] for r in csv.DictReader(open('/home/claude/diag/nw/b2n_data/newwriter/manifest.csv', encoding='utf-8'))}
for q in json.load(open(U/'claude_jh_fullphoto_20261109/dataset_line_quads.json')):
    refs.append(dict(set='jh', group='JH sentences', photo=str(U/f"JH_datasetNov5Sentence/Photo{q['photo']}.jpg"), pid=f"JH{q['photo']}", id=q['id'], writing=f"JH{q['photo']}", text=lab[q['id']], quad=q['line_quad']))
a = json.load(open(U/'whiteboard-ocr/outputs/jy_ocr_evaluation/full_page_evaluation/answer_keys.json'))
for p in a['photos']:
    for l in p['lines']:
        refs.append(dict(set='jy', group='Jy words', photo=str(U/'Jy Dataset'/f"Photo {p['photo_id']}.jpg"), pid=f"Jy{p['photo_id']}", id=l['id'], writing=f"Jy{p['photo_id']}", text=l['text'], quad=l['quad']))
json.dump(refs, open('refs.json', 'w'), ensure_ascii=False)
import collections; print(collections.Counter(r['group'] for r in refs), len({r['pid'] for r in refs}), 'photos')
