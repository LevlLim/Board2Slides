#!/usr/bin/env python3
"""Run Board2Slides' app/prepare_infer.py UNCHANGED, with one platform workaround.

On Linux aarch64 with Paddle 3.2.x, multithreaded CPU detection crashes (segmentation fault). The app's
Detector does not expose a thread setting, so when B2N_DET_CPU_THREADS is set this wrapper makes
paddleocr.TextDetection default to that many CPU threads. Nothing else changes: same code, same detector
weights, same decisions. On macOS/Windows the variable is unset and the app runs exactly as shipped.

Usage: python3 app_prepare_wrapper.py APP_DIR IMAGE OUT_IMAGE OUT_JSON --mode auto
"""
import os
import runpy
import sys

app_dir = sys.argv.pop(1)
threads = os.environ.get('B2N_DET_CPU_THREADS')
if threads:
    import paddleocr
    original = paddleocr.TextDetection

    def TextDetection(*args, **kwargs):
        kwargs.setdefault('cpu_threads', int(threads))
        return original(*args, **kwargs)

    paddleocr.TextDetection = TextDetection
sys.path.insert(0, app_dir)
os.chdir(app_dir)
sys.argv[0] = os.path.join(app_dir, 'prepare_infer.py')
runpy.run_path(sys.argv[0], run_name='__main__')
