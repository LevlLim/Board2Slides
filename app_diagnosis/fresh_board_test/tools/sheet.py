import sys
from pathlib import Path
from PIL import Image, ImageDraw
C = Path(sys.argv[1]); out = sys.argv[2]; pats = sys.argv[3].split(','); cw, chh = int(sys.argv[4]), int(sys.argv[5]); ncol = int(sys.argv[6])
fs = sorted(f for p in pats for f in C.glob(p + '*.png'))
rows = (len(fs) + ncol - 1) // ncol
sheet = Image.new('RGB', (ncol * (cw + 6), rows * (chh + 22)), (200, 200, 200)); d = ImageDraw.Draw(sheet)
for i, f in enumerate(fs):
    im = Image.open(f).convert('RGB'); im.thumbnail((cw, chh))
    x, y = (i % ncol) * (cw + 6), (i // ncol) * (chh + 22)
    sheet.paste(im, (x, y + 20)); d.text((x + 2, y + 4), f.stem.replace('IMG_', ''), fill=(180, 0, 0))
sheet.save(out, quality=80); print(len(fs))
