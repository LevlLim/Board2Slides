#!/usr/bin/env python3
"""Repeatable benchmark for the packaged pipeline (local or inside the container).

    python tools/benchmark.py [--repeats 3] [--threads 4] [--out outputs/benchmark_<host>.json]

Measures, on the four fixture photos:
  1. cold start  - a fresh process: imports + reader load + detector load + first photo (separately timed)
  2. warm        - per-photo processing time with models loaded (median of --repeats)
  3. memory      - peak resident memory of the process
  4. concurrency - --threads requests submitted at once to ONE shared OCRService (as Streamlit sessions share the
                   cached service). OCRService holds a lock, so requests are processed one at a time: expect total
                   time ~ sum of single requests and growing latency per waiting request.
Timings depend on the machine and its load; compare like with like.
"""
from __future__ import annotations

import argparse, json, os, platform, resource, statistics, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
CODE = PKG / 'b2n_ocr'


def peak_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == 'darwin' else r / 1024


def png_inputs(tmp):
    sys.path.insert(0, str(CODE))
    from student_flow import decode_photo
    m = json.loads((PKG / 'MANIFEST.json').read_text()); out = []
    for f in m['fixtures']:
        p = Path(tmp) / (Path(f['id']).stem + '.png'); decode_photo((PKG / 'fixtures/photos' / f['id']).read_bytes()).save(p); out.append((f['id'], p))
    return out


def cold(photo):
    code = f"""
import sys, time, json; t0 = time.perf_counter(); sys.path.insert(0, {str(CODE)!r})
import ocr_backend as ob; t1 = time.perf_counter(); s = ob.OCRService('S1'); t2 = time.perf_counter()
r = s.recognise({str(photo)!r}); t3 = time.perf_counter()
print(json.dumps(dict(imports=t1 - t0, reader_load=t2 - t1, detector_load=r['detector_cold_load_seconds'], first_photo_processing=r['seconds'], first_request_total=t3 - t2, process_total=t3 - t0)))"""
    t = time.perf_counter(); out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    d = json.loads(out.stdout.strip().splitlines()[-1]); d['wall_including_interpreter'] = time.perf_counter() - t; return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repeats', type=int, default=3); ap.add_argument('--threads', type=int, default=4); ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix='b2n-bench-'); photos = png_inputs(tmp)
    res = dict(host=platform.node(), platform=platform.platform(), machine=platform.machine(), cpu_count=os.cpu_count(),
               python=sys.version.split()[0], paddle=version('paddlepaddle'), repeats=a.repeats, threads=a.threads)
    res['cold_start'] = cold(photos[0][1])
    import ocr_backend as ob
    svc = ob.OCRService('S1'); svc.recognise(str(photos[0][1]))   # warm-up (not timed)
    warm = {}
    for fid, p in photos:
        ts = []
        for _ in range(a.repeats):
            r = svc.recognise(str(p)); ts.append(r['seconds'])
        warm[fid] = dict(median_seconds=statistics.median(ts), all=ts, regions=len(r['lines']))
    res['warm'] = warm
    jobs = [photos[i % len(photos)][1] for i in range(a.threads)]

    def one(p):
        t = time.perf_counter(); svc.recognise(str(p)); return time.perf_counter() - t
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        lat = list(ex.map(one, jobs))
    res['concurrency'] = dict(requests=a.threads, wall_seconds=time.perf_counter() - t, latencies=sorted(lat),
                              sum_of_single_warm=sum(warm[photos[i % len(photos)][0]]['median_seconds'] for i in range(a.threads)),
                              note='one shared OCRService; its lock serialises recognition')
    res['peak_rss_mb'] = peak_mb()
    out = a.out or PKG / 'outputs' / f"benchmark_{platform.system().lower()}_{platform.machine()}.json"
    out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(res, indent=1) + '\n')
    c = res['cold_start']
    print(f"cold start: imports {c['imports']:.1f}s, reader load {c['reader_load']:.1f}s, detector load {c['detector_load']:.1f}s, first photo {c['first_photo_processing']:.1f}s")
    for fid, w in warm.items(): print(f"warm {fid:16s} {w['median_seconds']:.2f}s ({w['regions']} regions)")
    cc = res['concurrency']
    print(f"concurrency {cc['requests']} at once: wall {cc['wall_seconds']:.1f}s (sum of single {cc['sum_of_single_warm']:.1f}s); latencies {[round(x, 1) for x in cc['latencies']]}")
    print(f"peak RSS {res['peak_rss_mb']:.0f} MB -> {out}")


if __name__ == '__main__':
    main()
