import json, math, os
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
A = Path(os.environ['A']); D = A/'claude_margin_diagnostic_20261109'; OUT = D/'persistent_review'; OUT.mkdir(exist_ok=True)
R = json.load(open(D/'RESULTS_MODELS.json')); P = R['persistent_tight']; C = {c['name']: c for c in R['crops']}
B = {b['name']: b for b in json.load(open(D/'boxes_final.json'))}
M = ['S0','S0s2','S1','S1s2','G0','G0s2']
F = lambda s: ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', s)
FM = lambda s: ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', s)
def need(l): return len(l) + sum(1 for a, b in zip(l, l[1:]) if a == b)
def reader_view(img):
    h, w = img.shape[:2]; rw = math.ceil(64*w/h); W = max(1024, rw)
    out = np.full((64, W, 3), 127, np.uint8); out[:, :rw] = cv2.resize(img, (rw, 64)); return out, rw
order = sorted(P, key=lambda n: (-need(C[n]['label'])/(math.ceil(64*(B[n]['box'][2]-B[n]['box'][0])/(B[n]['box'][3]-B[n]['box'][1]))/8)))
meta = []
W = 1060
items = []
for k, n in enumerate(order, 1):
    c, b = C[n], B[n]
    img = cv2.imread(str(D/'crops_tight'/(Path(n).stem + '.png')))
    rv, rw = reader_view(img); ratio = need(c['label'])/(rw/8)
    meta.append(dict(n=k, name=n, label=c['label'], group=c['group'], kept_h=c['kept_h'], tight_wh=list(img.shape[1::-1]), s1_text_px=rw,
                     s1_positions=rw/8, label_chars=len(c['label']), need_over_positions=round(ratio, 3),
                     preds={m: c['m'][m]['pred_tight'] for m in M}, cer={m: round(c['m'][m]['t']['cer'], 3) for m in M},
                     S={m: c['m'][m]['t']['S'] for m in M}, D={m: c['m'][m]['t']['D'] for m in M}, I={m: c['m'][m]['t']['I'] for m in M}))
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); s = min(120/im.height, (W-20)/im.width); im = im.resize((max(1, int(im.width*s)), max(1, int(im.height*s))))
    r = Image.fromarray(cv2.cvtColor(rv, cv2.COLOR_BGR2RGB)).resize((W-20, 64)) if rv.shape[1] > W-20 else Image.fromarray(cv2.cvtColor(rv, cv2.COLOR_BGR2RGB))
    h = 28 + im.height + 8 + 64 + 6 + 20 + 6*19 + 14
    t = Image.new('RGB', (W, h), 'white'); d = ImageDraw.Draw(t)
    d.text((10, 4), f"#{k}  {n}   [{c['group']}]   tight crop {img.shape[1]}x{img.shape[0]}   text covers {rw} of 1024 px = first {rw/8:.0f} of 128 output positions; {len(c['label'])} chars (chars needed / covered positions {ratio:.2f})", fill='black', font=F(15))
    t.paste(im, (10, 28)); y = 28 + im.height + 8
    t.paste(r, (10, y)); d.rectangle([10, y, 10 + r.width, y + 64], outline=(37, 99, 235)); y += 70
    d.text((10, y), f"LABEL  {c['label']}", fill=(21, 128, 61), font=FM(15)); y += 20
    for m in M:
        x = c['m'][m]['t']
        d.text((10, y), f"{m:5s}  {c['m'][m]['pred_tight']}", fill=(30, 30, 30), font=FM(15))
        d.text((W-250, y), f"CER {x['cer']:.2f}  S{x['S']} D{x['D']} I{x['I']}", fill=(185, 28, 28), font=FM(14)); y += 19
    items.append(t)
json.dump(meta, open(OUT/'persistent_26.json', 'w'), indent=1, ensure_ascii=False)
sheets = []; cur = []; H = 0
for t in items:
    if cur and H + t.height > 1700:
        sheets.append(cur); cur = []; H = 0
    cur.append(t); H += t.height + 6
sheets.append(cur)
for i, s in enumerate(sheets, 1):
    sh = Image.new('RGB', (W, sum(t.height + 6 for t in s)), (220, 220, 220)); y = 0
    for t in s:
        sh.paste(t, (0, y)); y += t.height + 6
    sh.save(OUT/f'sheet_{i}.jpg', quality=75)
    sh.convert('L').save(f'/tmp/v{i}.jpg', quality=45)
    print(i, len(s), sh.size, os.path.getsize(f'/tmp/v{i}.jpg'))
