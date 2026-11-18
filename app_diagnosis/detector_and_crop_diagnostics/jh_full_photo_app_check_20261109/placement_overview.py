import json, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
A = Path(os.environ['A']); O = A/'claude_jh_fullphoto_20261109'
P = json.load(open(O/'dataset_crop_placement.json'))
F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 60)
tiles = []
for pid in range(1, 8):
    im = Image.open(A/f'JH_datasetNov5Sentence/Photo{pid}.jpg').convert('RGB'); d = ImageDraw.Draw(im)
    for x in P:
        if x['photo'] != pid: continue
        q = [tuple(v) for v in x['quad']]; d.line(q + [q[0]], fill=(220, 38, 38), width=8)
        d.text((q[0][0] + 8, q[0][1] + 4), x['id'].split('__')[-1].replace('.jpg', ''), fill=(220, 38, 38), font=F)
    im.thumbnail((520, 520)); tiles.append(im)
W = sum(t.width for t in tiles[:4]) + 30; H = max(t.height for t in tiles[:4]) + max(t.height for t in tiles[4:]) + 10
sh = Image.new('RGB', (W, H), 'white'); x = y = 0
for i, t in enumerate(tiles):
    if i == 4: x = 0; y = max(t.height for t in tiles[:4]) + 10
    sh.paste(t, (x, y)); x += t.width + 10
sh.save(O/'placement_overview.jpg', quality=75); sh.save('/tmp/pl.jpg', quality=55); print(sh.size, os.path.getsize('/tmp/pl.jpg'))
