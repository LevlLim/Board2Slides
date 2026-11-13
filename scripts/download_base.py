"""Download the pinned public Qwen base model into this standalone app."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models" / "llm" / "base"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
REQUIRED = (
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "model.safetensors.index.json", "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
)


def main() -> None:
    if all((BASE / name).is_file() for name in REQUIRED):
        print("Pinned Qwen base files are already present.", flush=True)
        return
    BASE.mkdir(parents=True, exist_ok=True)
    print("Downloading Qwen/Qwen3-1.7B (~4.1 GB); this happens once.", flush=True)
    snapshot_download(
        repo_id="Qwen/Qwen3-1.7B", revision=REVISION, local_dir=str(BASE),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE"],
    )
    missing = [name for name in REQUIRED if not (BASE / name).is_file()]
    if missing:
        raise RuntimeError("Qwen download is incomplete: " + ", ".join(missing))
    print("Qwen base files are ready.", flush=True)


if __name__ == "__main__":
    main()
