"""Run the saved Qwen3-1.7B adaptive-visual LoRA on reviewed OCR JSON.

This process uses the model environment, separate from the lightweight web venv.
It writes a single JSON result file so model logs cannot corrupt the API protocol.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_BASE = PROJECT / "models" / "llm" / "base"
DEFAULT_ADAPTER = PROJECT / "models" / "llm" / "adapter"

# Keep this prompt in step with adaptive_prompt_v2 in llm_adaptive_visuals.ipynb.
PROMPT = (
    'Convert noisy WHITEBOARD OCR into a concise educational slide plan. Use pretrained knowledge '
    'to explain the same topic, but do not present invented measurements as facts from the board. '
    'Return raw JSON only with exactly title, summary, key_points, example, visual and '
    'uncertain_ocr_terms. Give 2 to 4 unique key_points, each with heading and explanation. example '
    'has exactly title and description. visual has exactly type, title, reason, source_region_ids '
    'and data. Allowed visual types: none, process, comparison_table, formula, line_chart, bar_chart. '
    'Choose comparison_table when two or more named items have parallel properties. Choose process '
    'only for at least three ordered stages. Choose formula for an explicit equation or mathematical '
    'relationship; prefer formula over a chart when the OCR provides only one worked point. Choose '
    'line_chart only for at least three ordered numeric measurements such as time or attempts. Choose '
    'bar_chart only for at least three numeric category measurements. Otherwise choose none. Chart '
    'labels and values must be copied from cited OCR regions. process data has only steps. '
    'comparison_table data has only columns and rows. formula data has only expression, variables and '
    'worked_steps; every variable is an object with exactly symbol and meaning. Chart data has only '
    'labels, series and unit; every series has exactly name and values. none uses empty data. Use only '
    'valid OCR region IDs, do not repeat headings or explanations, and do not ask a question. Return '
    'JSON only. WHITEBOARD OCR: '
)


def model_paths() -> tuple[Path, Path]:
    base = Path(os.environ.get("BOARD2SLIDES_LLM_BASE", str(DEFAULT_BASE)))
    adapter = Path(os.environ.get("BOARD2SLIDES_LLM_ADAPTER", str(DEFAULT_ADAPTER)))
    return base, adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base, adapter = model_paths()
    if not (base / "config.json").is_file() or not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("Qwen base model or v3 LoRA adapter is missing")
    requested_device = os.environ.get("BOARD2SLIDES_LLM_DEVICE", "auto").lower()
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    elif requested_device in {"cuda", "mps", "cpu"}:
        device = requested_device
    else:
        raise ValueError("BOARD2SLIDES_LLM_DEVICE must be auto, cuda, mps or cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but no CUDA device is available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was selected but no Apple GPU is available")
    incoming = json.loads(args.input.read_text(encoding="utf-8"))
    # Training prompts contained only id and text; UI metadata must not change
    # the model prompt or consume the 800-token generation budget.
    ocr = {"regions": [{"id": row["id"], "text": row["text"]}
                       for row in incoming["regions"]]}
    prompt = PROMPT + json.dumps(ocr, ensure_ascii=False, separators=(",", ":"))
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    dtype = torch.bfloat16 if device == "cuda" else torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=dtype, local_files_only=True, low_cpu_mem_usage=True).to(device)
    model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    model.eval()
    tokenized = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
    input_ids = tokenized["input_ids"] if hasattr(tokenized, "keys") else tokenized
    input_ids = input_ids.to(device)
    with torch.inference_mode():
        generated = model.generate(input_ids, max_new_tokens=800, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(generated[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()
    parsed = json.loads(raw)
    args.output.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
