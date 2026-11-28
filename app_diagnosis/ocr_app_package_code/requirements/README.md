# Environment notes

| | tested | status |
|---|---|---|
| macOS arm64, Python 3.12.14, paddlepaddle 3.3.1 | the app's `.venv-paddle` (reference) | reference outputs |
| Linux x86_64 (container), Python 3.12, paddlepaddle 3.3.1 | `deploy/Dockerfile` | **not yet run here**: build and run `verify_package.py --smoke` in the container |
| Linux aarch64, Python 3.12.15, paddlepaddle **3.2.2** | local Linux VM check | 3.3.1 has no aarch64 wheel; used only as a Linux portability check |

- Do not copy `.venv-paddle`: it also holds torch, ultralytics, realesrgan, Jupyter etc. used for training/experiments. `base.txt` lists only what inference and the UI import.
- `torch` is not needed. On the original Mac env the backend imports torch first only to avoid a libiomp clash when both are installed; the packaged backend skips it when torch is absent.
- OpenCV: install `opencv-override.txt` after `base.txt` (see that file). `verify_package.py` warns if `cv2` is not 5.0.0.
- The detector runs with `device='cpu', enable_mkldnn=False` (plain Paddle engine) so a Linux x86 host does not silently switch to oneDNN kernels, which the macOS arm64 reference never uses. S1 already disables oneDNN.
