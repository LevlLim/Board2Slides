"""Start the standalone Board2Slides server using its web environment."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_PYTHON = ROOT / ".venv-web" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not WEB_PYTHON.is_file():
        raise RuntimeError("Run python scripts/setup.py first.")
    print(f"Open http://localhost:{args.port} on this computer.", flush=True)
    subprocess.run([
        str(WEB_PYTHON), "-m", "uvicorn", "app:app", "--app-dir", str(ROOT / "app"),
        "--host", args.host, "--port", str(args.port),
    ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
