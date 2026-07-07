#!/usr/bin/env python3
"""Fit an Anthropic jlens lens from a prompt JSONL file.

The wrapper records an explicit command boundary for Limes audit runs. It does
not vendor or reimplement Anthropic's reference fitter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path.")
    parser.add_argument("--prompts-jsonl", required=True)
    parser.add_argument("--out", required=True, help="Output lens path.")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    import torch
    import transformers
    import jlens

    device = choose_device(args.device, torch)
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model)
    hf.to(device)
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf, tok)
    prompts = load_prompts(Path(args.prompts_jsonl), args.max_prompts)
    lens = jlens.fit(model, prompts=prompts, checkpoint_path=args.checkpoint_path)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lens.save(args.out)
    print(args.out)
    return 0


def load_prompts(path: Path, max_prompts: int | None) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompts.append(row["text"])
            if max_prompts is not None and len(prompts) >= max_prompts:
                break
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def choose_device(requested: str, torch_module) -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    raise SystemExit(main())
