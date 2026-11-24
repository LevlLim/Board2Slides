"""Preview or upload the application files to an existing Hugging Face Docker Space.

Run from the repository root with the LLM environment's Python. The Hugging
Face token is read from the normal `hf auth login` store, never from this repo.
"""
from __future__ import annotations

import argparse
import fnmatch
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = (
    "README.md", "DEPLOY.md", "Dockerfile", ".dockerignore", ".gitattributes",
    "package.json", "package-lock.json",
    "requirements/*.txt", "app/**", "renderer/**", "scripts/download_base.py",
    "models/ocr/**", "models/llm/adapter/**",
)
EXCLUDE_PARTS = {".git", ".cache", "__pycache__", "data", "node_modules"}
REQUIRED = (
    "README.md", "Dockerfile", "package-lock.json", "scripts/download_base.py",
    "models/llm/adapter/adapter_model.safetensors",
    *(f"models/ocr/{model}/{file}" for model in
      ("detector", "baseline_rec", "recognizer", "augmented_rec")
      for file in ("inference.json", "inference.pdiparams", "inference.yml")),
)


def upload_files() -> list[str]:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True,
    ).stdout.decode("utf-8").split("\0")
    return sorted(path for path in tracked if path and
                  any(fnmatch.fnmatchcase(path, pattern) for pattern in INCLUDE) and
                  not any(part in EXCLUDE_PARTS for part in Path(path).parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("space", nargs="?", help="Hugging Face Space, e.g. Levl07/Board2Notes")
    parser.add_argument("--upload", action="store_true", help="Upload after previewing the file list")
    parser.add_argument("--allow-public", action="store_true", help="Allow upload to a public Space")
    args = parser.parse_args()

    files = upload_files()
    missing = [name for name in REQUIRED if name not in files or not (ROOT / name).is_file()]
    if missing:
        parser.error("Commit the required application files first: " + ", ".join(missing))
    size = sum((ROOT / name).stat().st_size for name in files)
    print(f"Deployment manifest: {len(files)} tracked files, {size / 1024**2:.1f} MiB")
    for name in files:
        print("  " + name)
    print("Qwen base weights, photos, environments, caches, and raw reader archives are excluded.")
    if not args.upload:
        print("Preview only. Add Levl07/Board2Notes --upload to publish to your Space.")
        return
    if not args.space or args.space.count("/") != 1:
        parser.error("Supply the existing Space as USERNAME/SPACE_NAME")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout
    if status.strip():
        parser.error("Commit or discard local changes before uploading the Space")

    from huggingface_hub import HfApi

    api = HfApi()
    space = api.repo_info(repo_id=args.space, repo_type="space")
    if not space.private and not args.allow_public:
        parser.error("This Space is public. Use a private Space for testing or pass --allow-public.")
    api.upload_folder(
        repo_id=args.space,
        repo_type="space",
        folder_path=str(ROOT),
        allow_patterns=files,
        commit_message="Deploy Board2Slides application",
    )
    print(f"Uploaded. Open https://huggingface.co/spaces/{args.space} to watch the build.")


if __name__ == "__main__":
    main()
