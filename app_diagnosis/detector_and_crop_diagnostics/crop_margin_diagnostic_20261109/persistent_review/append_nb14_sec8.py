"""Append section 8 (review of the 26 persistently difficult tightened crops) to notebook 14. Existing cells untouched."""
import nbformat as nbf, sys, shutil
from pathlib import Path
A = Path(sys.argv[1]); p = A / 'whiteboard-ocr/notebooks/14_crop_margin_diagnostic.ipynb'
bak = p.with_name('14_crop_margin_diagnostic_before_section8.ipynb.bak')
if not bak.exists():
    shutil.copy(p, bak)
nb = nbf.read(p, 4)
assert not any('## 8 ·' in c.source for c in nb.cells), 'section 8 already present'
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
## 8 · The 26 crops still hard for all six readers after tightening (review, 9 Nov)

**Which crops.** These are the 26 crops with **≥ 15 % CER for every reader** (S0, S0s2, S1, S1s2, G0, G0s2) on the tightened crops (section 7).

**What this section does.**
- Shows every one of the 26: the tight crop, the exact 64-px input S1 receives, the label and all six predictions.
- Groups the failures by what is visible in the images, and backs each group with measurements.

**What changed for this section.**
- One extra S1 run recorded which output position each character comes from: `persistent_review/s1_position_trace.json`.
- No training was run, and the app, the labels and the rectangles were not changed.
- The categories are the assistant's visual judgement; uncertain items are flagged.

Files are in `claude_margin_diagnostic_20261109/persistent_review/`:
- `REVIEW.json`: one row per crop with its category, measurements and notes;
- `sheet_1..7.jpg`;
- `persistent_26.json`.
""")

code(r"""
from IPython.display import Image as IPImage
RV = json.loads((D / 'persistent_review/REVIEW.json').read_text())
rv = pd.DataFrame(RV['rows'])
print(RV['status'])
display(rv.groupby(['category', 'category_description']).agg(crops=('name', 'size'), mean_CER_6=('mean_cer6', 'mean'),
        mean_deletions=('mean_D6', 'mean'), mean_substitutions=('mean_S6', 'mean')).round(2))
display(rv[['category', 'name', 'label', 'mean_cer6', 'mean_D6', 'mean_S6', 'label_chars', 's1_positions', 'need_over_positions',
            'ink_angle_deg', 'ink_extent_over_local_band', 's1_chars_emitted_in_padding', 'note']])
""")

md(r"""
### 8a · All 26 crops

Crops are ordered from most crowded to least. **Crowded** = characters needed ÷ S1 output positions for the text.
- One output position covers 8 px of the 64-px-high input.
- "Characters needed" counts a CTC blank between doubled letters, which is why a doubled letter costs an extra position.

**Each panel shows:**
- top: the tight crop;
- middle: exactly what S1 receives, resized to 64 px high and padded to 1024 px (mid-grey = padding);
- bottom: the label, then the six predictions with CER and substitution/deletion/insertion (S/D/I) counts.
""")

code(r"""
for i in range(1, 8):
    display(IPImage(filename=str(D / f'persistent_review/sheet_{i}.jpg'), width=1000))
""")

md(r"""
### 8b · What the images show

| group | crops | what is visible |
|---|---|---|
| **A1 · sloped long line in an axis-aligned box** | **16** (all JH sentence crops) | The sentence rises 3–7° from left to right. To contain the slope, the upright box is about 3× as tall as the writing band. Squeezed to 64 px, the letters get small and the line gets short: S1 has only 35–50 output positions for 38–60 characters. Every reader gets the first words right and degrades to the right. Deletions dominate. |
| **A2 · long line with a leftover blank band** | 2 (DS) | The box kept a speck or light patch, so the writing sits in the lower half. Same effect as A1: the writing is compressed in S1's input. |
| C · legible long line, scattered letter confusions | 1 (DS) | m ↔ rn, y ↔ 4. Possible label question ("They" vs written "Thy"). |
| D · short word, unusual letterform read the same wrong way by all six | 5 (JH) | *Legendary* → "Legendain"; *Yuzu Juice* → "Vuzu )uice"; *Giant* → "Ctiant"; *cake* → "cale". The reader mistakes are consistent across all six; one or two wrong characters on a short word already exceed 15 % CER. |
| E · faint and small | 1 | *Panasonic*: very low contrast. |
| F · strongly rotated short text | 1 | *100W* at about 20°. |

**Uncertain diagnoses** (also in the `note` column):
- **Possible label mismatches** (the labels are unchanged; owner to check):
  - JH6.6 *access*: the word may be written *acess*.
  - DS 1.4.2: *They* may be written *Thy*.
  - JH30.2 *Giant*: the G has a T-like bar, so "CTiant" may simply be what is written.
- **Near-duplicates:** JH19.1 and JH22.1 look like the same word photographed twice, so that error counts twice.
- **Mixed causes:**
  - The two A2 crops show both a leftover margin and crowding.
  - *Yuzu Juice* is also a tiny, blurred crop (143 × 35 px).
""")

code(r"""
# Evidence for group A: long lines (label >= 25 chars) - persistent failures versus all other long lines
import math, cv2
def ink_mask(gray):
    h, w = gray.shape; k = max(15, (min(h, w) // 3) | 1); k = min(k, 255) | 1
    diff = cv2.subtract(cv2.medianBlur(gray, k), gray); thr = max(10.0, 0.20 * float(np.percentile(diff, 99.5)))
    m = cv2.morphologyEx((diff > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8); keep = [i for i in range(1, n) if st[i, 4] >= max(6, 0.00005 * h * w)]
    return np.isin(lab, keep)
trace = json.loads((D / 'persistent_review/s1_position_trace.json').read_text())
RM = json.loads((D / 'RESULTS_MODELS.json').read_text()); persist = set(RM['persistent_tight'])
MODS = ['S0', 'S0s2', 'S1', 'S1s2', 'G0', 'G0s2']
ll = []
for c in RM['crops']:
    if len(c['label']) < 25:
        continue
    g = cv2.imread(str(D / 'crops_tight' / (Path(c['name']).stem + '.png')), 0); m = ink_mask(g); ys, xs = np.nonzero(m)
    X = np.stack([xs, ys], 1).astype(float); X -= X.mean(0); _, _, vt = np.linalg.svd(X, full_matrices=False)
    ang = (math.degrees(math.atan2(vt[0, 1], vt[0, 0])) + 90) % 180 - 90
    W = g.shape[1]; band = np.median([s.max() - s.min() for s in (ys[(xs >= a * W / 10) & (xs < (a + 1) * W / 10)] for a in range(10)) if len(s) > 20])
    lab = c['label']; need = len(lab) + sum(a == b for a, b in zip(lab, lab[1:]))
    ll.append(dict(name=c['name'], persistent=c['name'] in persist, writer=c['name'].split('__')[0], abs_angle=abs(ang),
                   extent_over_band=(ys.max() - ys.min()) / band, need_over_positions=need / (math.ceil(64 * g.shape[1] / g.shape[0]) / 8),
                   spills=trace[c['name']]['chars_in_padding'] > 0, mean_cer6=np.mean([c['m'][x]['t']['cer'] for x in MODS])))
LL = pd.DataFrame(ll)
display(LL.groupby('persistent').agg(long_lines=('name', 'size'), median_abs_angle=('abs_angle', 'median'),
        median_extent_over_band=('extent_over_band', 'median'), median_need_over_positions=('need_over_positions', 'median'),
        crowded_ge_1=('need_over_positions', lambda s: int((s >= 1).sum())), S1_text_in_padding=('spills', 'sum'), mean_CER=('mean_cer6', 'mean')).round(2))
bins = pd.cut(LL.need_over_positions, [0, .5, .75, 1.0, 1.25, 9], right=False)
display(LL.groupby(bins, observed=True).agg(long_lines=('name', 'size'), persistent=('persistent', 'sum'), mean_CER=('mean_cer6', 'mean')).round(3))
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for p_, col, lab_ in ((False, '#94a3b8', 'other long lines'), (True, '#dc2626', 'persistent failures')):
    s = LL[LL.persistent == p_]; ax[0].scatter(s.abs_angle, s.need_over_positions, c=col, s=18, label=lab_)
ax[0].axhline(1, ls='--', c='k', lw=.8); ax[0].set_xlabel('|slope of the writing| (degrees)'); ax[0].set_ylabel('characters needed / S1 positions')
ax[0].legend(fontsize=8); ax[0].set_title('Slope inflates the box, crowding the characters')
ax[1].scatter(LL.need_over_positions, LL.mean_cer6, c=np.where(LL.persistent, '#dc2626', '#94a3b8'), s=18)
ax[1].axvline(1, ls='--', c='k', lw=.8); ax[1].set_xlabel('characters needed / S1 positions'); ax[1].set_ylabel('mean CER, six readers'); ax[1].set_title('Errors jump once the line is over-crowded')
plt.tight_layout(); plt.show()
""")

code(r"""
# What S1 does with an over-crowded line: one character per output position ('_' = blank). The input is padded to 1024 px,
# so the text occupies only the first `text_positions` of the 128 positions; the rest is padding.
for n in ['jhs__train__JH1.1.jpg', 'jhs__train__JH6.4.jpg', 'jhs__train__JH3.6.jpg']:
    t = trace[n]
    print(f"{n}: label {[c['label'] for c in RM['crops'] if c['name'] == n][0]!r}")
    print(f"   text occupies {t['text_positions']} of {t['T']} positions; characters emitted inside padding: {t['chars_in_padding']} {t['padding_text']!r}")
    print('   ' + t['per_position'])
""")

md(r"""
### 8c · The most common actionable pattern, and one experiment to test it

**Pattern.** 18 of the 26 crops (groups A1 and A2) fail because **the writing reaches S1 compressed**: small letters, too few output positions per character.
- 16 of these are JH sentence lines written on a slope and cut with an upright box.
- Compared with the other long lines:

| | persistent long-line failures | other long lines |
|---|---|---|
| median slope | ≈ 4° | ≈ 1° |
| box height ÷ writing-band height | ≈ 3.1 | ≈ 1.8 |
| crowded (≥ 1 character per S1 position) | 17 of 19 | 1 of 106 |

- S1 runs out of room. In all 16 A1 crops, and in both A2 crops, the end of the sentence is emitted as garbled text **inside the padding** at the right of the canvas.
- This is not unfamiliar handwriting as such. The same writer's flat lines read well (e.g. JH3.6, JH4.8, 0–1 % CER).
- In training data, fewer than 7 % of long lines are this crowded.

The other 8 crops are a mix: letterform confusions on short words, one faint word, one rotated word, and possible label questions. None of these patterns is large enough to act on yet.

**Recommended experiment: rectify the line, then re-tighten (frozen readers, no training).**

- *Inputs:* the same 160 frozen tight crops and labels.
- *Intervention (label-free, images only):*
  1. Estimate the writing's angle from the ink mask (the principal axis, used only for long lines: ink width ≥ 6 × band height).
  2. Rotate the crop so the line is horizontal.
  3. Re-apply the same notebook-14 tight-box rule.
  4. Freeze angles, boxes and a visual check of every changed crop **before** any reading.
  5. Crops with |angle| < 1° stay byte-identical.
- *Readers:* the same six exports and the same Mac scorer.
- *Pre-declared stratum:* long lines with |angle| ≥ 2°, defined from the images alone.
- *Supported only if all three hold:*
  1. The stratum's mean CER (S1, mean of seeds) falls, with a photo-cluster bootstrap CI below 0.
  2. Deletions fall.
  3. The number of crops emitting text into the padding falls.
- *Safety check:* changed crops outside the stratum get no worse than +1 point.
- *Optional mechanism arm:* stretch the *un-rotated* crop horizontally to the same S1 width.
  - If it recovers as much as rectification does, crowding is the mechanism (relevant for the CTC slot count).
  - If only rotation helps, the slope itself matters.

**Why this experiment first.**
- It targets the largest group with a label-free change.
- It reuses everything already frozen.
- It answers whether the remaining "unfamiliar writer" gap on JH sentences is mostly a crop-geometry artifact.

The app's detector already cuts along rotated quads (`quad_crop`), so if this is supported, the next check is whether app crops of JH's sloped lines are already rectified. JH's full photos (`JH_datasetNov5Sentence/Photo1–7.jpg`) make that possible later. That check, and any training change, waits for this result.
""")

nb.cells.extend(C)
nbf.write(nb, p)
print('appended', len(C), 'cells; backup', bak.name)
