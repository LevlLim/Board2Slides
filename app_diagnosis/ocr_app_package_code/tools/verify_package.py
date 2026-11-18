#!/usr/bin/env python3
"""Verify the Board2Notes OCR package.

    python tools/verify_package.py                       # integrity: every model/code/fixture file present and unchanged
    python tools/verify_package.py --smoke               # + run the packaged pipeline on the fixtures, compare with a reference
    python tools/verify_package.py --smoke --against-app # (dev Mac only) + run the live app on the same fixtures and compare
    python tools/verify_package.py --smoke --record mac-arm64   # store this run as fixtures/expected/mac-arm64.json
    python tools/verify_package.py --smoke --record linux-x86_64 --record-dir outputs/expected   # (container: only outputs/ is writable)

Integrity = every manifest file present and unchanged, and no unexpected file inside the checked folders
(models/, b2n_ocr/, fixtures/photos/; Python caches and .DS_Store ignored). Docs, tools/, deploy/ and requirements/
are not hash-checked.

Exit code 0 = all checks passed, 1 = a check failed. Implementation verification only: this is NOT an accuracy evaluation.
"""
from __future__ import annotations

import argparse, hashlib, json, platform, subprocess, sys
from importlib.metadata import version
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
TESTED = {'paddlepaddle': '3.3.1', 'paddleocr': '3.7.0', 'paddlex': '3.7.2', 'numpy': '2.3.5', 'pillow': '12.3.0',
          'pillow-heif': '1.8.1', 'PyYAML': '6.0.2', 'streamlit': '1.67.0'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


CHECKED_DIRS = ('models', 'b2n_ocr', 'fixtures/photos')


def ignored(p):
    return '__pycache__' in p.parts or p.suffix == '.pyc' or p.name == '.DS_Store'


def integrity():
    m = json.loads((PKG / 'MANIFEST.json').read_text())
    missing, changed = [], []
    for rel, info in m['files'].items():
        p = PKG / rel
        if not p.is_file():
            missing.append(rel)
        elif sha(p) != info['sha256']:
            changed.append(rel)
    extra = [str(p.relative_to(PKG)) for d in CHECKED_DIRS for p in sorted((PKG / d).rglob('*'))
             if p.is_file() and not ignored(p) and p.relative_to(PKG).as_posix() not in m['files']]
    print(f"Integrity: {len(m['files'])} files in manifest | missing {len(missing)} | changed {len(changed)} | "
          f"unexpected files in {', '.join(d + '/' for d in CHECKED_DIRS)} {len(extra)}")
    for x in missing: print('  MISSING ', x)
    for x in changed: print('  CHANGED ', x)
    for x in extra: print('  EXTRA   ', x)
    env_warn = []
    for k, v in TESTED.items():
        try:
            got = version(k)
        except Exception:
            got = None
        if got != v:
            env_warn.append(f'{k}: tested {v}, installed {got}')
    import cv2
    if not cv2.__version__.startswith('5.0.0'):
        env_warn.append(f'cv2 in use {cv2.__version__} (tested 5.0.0; see requirements/README)')
    for w in env_warn: print('  ENV NOTE', w)
    return not (missing or changed or extra), env_warn


def run(code_dir, out, python=None):
    r = subprocess.run([python or sys.executable, str(PKG / 'tools/run_fixtures.py'), '--code-dir', str(code_dir), '--out', str(out)])
    if r.returncode:
        raise SystemExit(f'fixture run failed for {code_dir}')
    return json.loads(Path(out).read_text())


def compare(a, b, label):
    """Exact implementation comparison: same fixtures, same status, same lines, boxes within 0.01 px, identical text."""
    ok, rows = True, []
    for fid, x in a['results'].items():
        y = b['results'].get(fid)
        if y is None:
            rows.append((fid, 'ABSENT in ' + label)); ok = False; continue
        if x['status'] != y['status']:
            rows.append((fid, f"status {x['status']} vs {y['status']}")); ok = False; continue
        if x['status'] != 'ok':
            rows.append((fid, f"both {x['status']}")); continue
        same_n = x['n_lines'] == y['n_lines']
        box = max((abs(p - q) for l, k in zip(x['lines'], y['lines']) for pp, qq in zip(l['box'], k['box']) for p, q in zip(pp, qq)), default=0.0) if same_n else None
        same_text = x['text'] == y['text']
        good = same_n and box is not None and box <= 0.01 and same_text
        ok &= good
        rows.append((fid, f"{'IDENTICAL' if good else 'DIFFERENT'} lines {x['n_lines']}/{y['n_lines']} max box diff {box if box is None else round(box, 4)} text {'same' if same_text else 'differs'}"))
    print(f'Comparison with {label}:')
    for fid, s in rows: print(f'  {fid:26s} {s}')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--against-app', action='store_true')
    ap.add_argument('--app-python', type=Path, default=PKG.parent / '.venv-paddle/bin/python',
                    help='interpreter for the live app (default: Assignment/.venv-paddle/bin/python)')
    ap.add_argument('--record', metavar='NAME'); ap.add_argument('--reference', metavar='NAME', help='reference to compare with (default: this platform)')
    ap.add_argument('--record-dir', type=Path, default=Path('fixtures/expected'),
                    help='folder for --record, relative to the package (default fixtures/expected; use outputs/expected in the container)')
    a = ap.parse_args()
    ok, _ = integrity()
    if not ok:
        print('FAILED: package files are missing, changed or unexpected. Re-copy the package; do not run with substituted weights.'); sys.exit(1)
    if not a.smoke:
        print('PASSED (integrity only).'); return
    outdir = PKG / 'outputs'; outdir.mkdir(exist_ok=True)
    pk = run(PKG / 'b2n_ocr', outdir / 'fixtures_packaged.json')
    bad = [f for f, r in pk['results'].items() if r['status'] == 'error']
    expect_rejected = {'unreadable_bytes', 'truncated_jpeg'}
    wrong = [f for f in expect_rejected if pk['results'][f]['status'] != 'rejected']
    print(f"Packaged run: {sum(r['status'] == 'ok' for r in pk['results'].values())} read, "
          f"{sum(r['status'] == 'rejected' for r in pk['results'].values())} rejected with a message, {len(bad)} unexpected errors {bad}")
    ok &= not bad and not wrong and pk['results']['blank_board']['status'] == 'ok'
    if a.record:
        dst = (PKG / a.record_dir) / f'{a.record}.json'; dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(pk, indent=1, ensure_ascii=False) + '\n'); print('recorded reference', dst)
    ref_name = a.reference or ('mac-arm64' if sys.platform == 'darwin' else f'linux-{platform.machine()}')
    ref = next((d / f'{ref_name}.json' for d in (PKG / 'fixtures/expected', PKG / 'outputs/expected') if (d / f'{ref_name}.json').is_file()),
               PKG / 'fixtures/expected' / f'{ref_name}.json')
    if ref.is_file() and not a.record:
        ok &= compare(pk, json.loads(ref.read_text()), f'reference {ref_name}')
    elif not a.record:
        print(f'No stored reference {ref.name}: outputs saved to outputs/fixtures_packaged.json for review (cross-platform differences are expected; see VERIFICATION.md).')
    if a.against_app:
        app = PKG.parent / 'whiteboard-ocr/ocr_prototype'
        if not a.app_python.is_file():
            raise SystemExit(f'--against-app needs the live app environment; not found: {a.app_python}')
        live = run(app, outdir / 'fixtures_live_app.json', str(a.app_python))
        ok &= compare(pk, live, 'the live app (whiteboard-ocr/ocr_prototype)')
    print('PASSED' if ok else 'FAILED'); sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
