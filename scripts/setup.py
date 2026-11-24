"""Set up this repo on Windows or macOS: Python environments, Node, and Qwen."""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def env_python(name: str) -> Path:
    return ROOT / name / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(*args: str | Path) -> None:
    command = [str(arg) for arg in args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def ensure_env(name: str) -> Path:
    python = env_python(name)
    if not python.is_file():
        print(f"Creating {name} with Python {sys.version_info.major}.{sys.version_info.minor}", flush=True)
        venv.EnvBuilder(with_pip=True).create(ROOT / name)
    return python


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-only", action="store_true",
                        help="Install only the web app and slide renderer")
    args = parser.parse_args()
    if sys.version_info[:2] not in {(3, 12), (3, 13)}:
        raise RuntimeError("Use Python 3.12 or 3.13 for the pinned model packages.")
    if not shutil.which("node") or not shutil.which("npm"):
        raise RuntimeError("Install Node.js 22 or newer so node and npm are on PATH.")
    web = ensure_env(".venv-web")
    run(web, "-m", "pip", "install", "-r", ROOT / "requirements/web.txt")
    run("npm.cmd" if os.name == "nt" else "npm", "ci", "--omit=dev")
    if args.demo_only:
        print("Demo setup complete. OCR and LLM installation was skipped.", flush=True)
        return

    if not (ROOT / "models/ocr/recognizer/inference.pdiparams").is_file():
        raise RuntimeError("The team's exported OCR recognizer is missing.")
    if not (ROOT / "models/ocr/detector/inference.pdiparams").is_file():
        raise RuntimeError("The official mobile OCR detector is missing.")
    if not (ROOT / "models/ocr/baseline_rec/inference.pdiparams").is_file():
        raise RuntimeError("The English 94-character OCR baseline recognizer is missing.")
    if not (ROOT / "models/ocr/augmented_rec/inference.pdiparams").is_file():
        raise RuntimeError("The multilingual-character OCR recognizer is missing.")
    if not (ROOT / "models/llm/adapter/adapter_model.safetensors").is_file():
        raise RuntimeError("The team's Qwen LoRA adapter is missing.")

    ocr = ensure_env(".venv-ocr")
    run(ocr, "-m", "pip", "install", "-r", ROOT / "requirements/ocr.txt")
    cache_dir = ROOT / ".cache/paddlex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_dir)
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    run(ocr, "-c", "import paddle,paddleocr; print('Paddle',paddle.__version__,'PaddleOCR',paddleocr.__version__)")

    llm = ensure_env(".venv-llm")
    if platform.system() == "Windows" and shutil.which("nvidia-smi"):
        torch_index = "https://download.pytorch.org/whl/cu128"
        run(llm, "-m", "pip", "install", "torch==2.11.0", "--index-url", torch_index)
    elif platform.system() == "Windows":
        run(llm, "-m", "pip", "install", "torch==2.11.0", "--index-url",
            "https://download.pytorch.org/whl/cpu")
    else:
        run(llm, "-m", "pip", "install", "torch==2.11.0")
    run(llm, "-m", "pip", "install", "-r", ROOT / "requirements/llm.txt")
    run(llm, ROOT / "scripts/download_base.py")
    print("Full setup complete. Start with: python scripts/run.py", flush=True)


if __name__ == "__main__":
    main()
