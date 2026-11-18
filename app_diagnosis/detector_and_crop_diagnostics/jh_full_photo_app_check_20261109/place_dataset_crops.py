"""Place each JH sentence dataset crop in its full photo by image matching only (SIFT + RANSAC). No labels, no OCR."""
import json, csv, os, hashlib
from pathlib import Path
import cv2, numpy as np
A = Path(os.environ['A']); J = A/'JH_datasetNov5Sentence'; O = A/'claude_jh_fullphoto_20261109'; O.mkdir(exist_ok=True)
man = [r for r in csv.DictReader(open(A/'board2notes_newwriter_test_20261107/manifest.csv')) if r['id'].startswith('jhs')]
sift = cv2.SIFT_create()
photos = {}
out = []
for r in man:
    pid = int(r['photo'].split('JH')[1]); ph = J/f'Photo{pid}.jpg'
    if pid not in photos:
        g = cv2.imread(str(ph), cv2.IMREAD_GRAYSCALE); kp, de = sift.detectAndCompute(g, None); photos[pid] = (g, kp, de)
    g, kp, de = photos[pid]
    cpath = J/r['source_file']; c = cv2.imread(str(cpath), cv2.IMREAD_GRAYSCALE)
    k2, d2 = sift.detectAndCompute(c, None)
    m = cv2.BFMatcher().knnMatch(d2, de, k=2); good = [a for a, b in m if a.distance < 0.75*b.distance]
    rec = dict(id=r['id'], photo=pid, crop=str(cpath.relative_to(A)), crop_sha256=hashlib.sha256(cpath.read_bytes()).hexdigest(),
               photo_file=str(ph.relative_to(A)), crop_wh=[c.shape[1], c.shape[0]], matches=len(good))
    if len(good) >= 8:
        src = np.float32([k2[a.queryIdx].pt for a in good]); dst = np.float32([kp[a.trainIdx].pt for a in good])
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is not None:
            h, w = c.shape; corners = np.float32([[0,0],[w,0],[w,h],[0,h]])
            quad = cv2.transform(corners[None], M)[0]
            s = float(np.hypot(M[0,0], M[1,0])); ang = float(np.degrees(np.arctan2(M[1,0], M[0,0])))
            # verify: warp the photo region back into crop space and correlate
            Minv = cv2.invertAffineTransform(M); back = cv2.warpAffine(g, Minv, (w, h))
            corr = float(np.corrcoef(back.ravel().astype(float), c.ravel().astype(float))[0, 1])
            rec.update(inliers=int(inl.sum()), scale=round(s, 4), rotation_deg=round(ang, 2), quad=quad.round(1).tolist(), pixel_correlation=round(corr, 4))
    out.append(rec)
json.dump(out, open(O/'dataset_crop_placement.json', 'w'), indent=1)
ok = [x for x in out if x.get('pixel_correlation', 0) > 0.9]
print(len(out), 'placed with corr>0.9:', len(ok)); print('weak:', [(x['id'], x['matches'], x.get('inliers'), x.get('pixel_correlation'), x.get('scale')) for x in out if x.get('pixel_correlation', 0) <= 0.9])
print('scale range', min(x['scale'] for x in ok), max(x['scale'] for x in ok), 'rot range', min(x['rotation_deg'] for x in ok), max(x['rotation_deg'] for x in ok))
