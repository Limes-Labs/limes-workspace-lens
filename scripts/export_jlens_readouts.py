#!/usr/bin/env python3
"""Export Anthropic jlens readouts into the Limes Workspace Lens schema.

This script is optional because real model runs require model weights, a fitted
Jacobian lens, and ML dependencies. It is kept outside the core package so the
repo remains importable and testable on CPU-only machines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path.")
    parser.add_argument("--lens-repo", required=True, help="Fitted lens repo or local path.")
    parser.add_argument("--lens-file", default="lens.pt")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--positions",
        default="-1",
        help="Comma-separated token positions passed to jlens.apply, for example -1,-2.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    import torch
    import transformers
    import jlens

    from limes_workspace_lens.schema import READOUT_SCHEMA, load_json, validate_audit_spec

    spec = load_json(args.spec)
    errors = validate_audit_spec(spec)
    if errors:
        raise SystemExit("\n".join(errors))

    device = choose_device(args.device, torch)
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model)
    hf.to(device)
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file)

    positions = [int(part) for part in args.positions.split(",") if part.strip()]
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for prompt in spec["prompts"]:
            lens_logits, model_logits, _metadata = lens.apply(
                model,
                prompt["text"],
                positions=positions,
            )
            for layer, logits in sorted(lens_logits.items()):
                for position_index, position in enumerate(positions):
                    values, indices = logits[position_index].topk(args.top_k)
                    rows.append(
                        {
                            "prompt_id": prompt["id"],
                            "position": str(position),
                            "layer": int(layer),
                            "top_tokens": [
                                {
                                    "token": tok.decode([int(token_id)]),
                                    "rank": rank + 1,
                                    "score": float(value),
                                }
                                for rank, (value, token_id) in enumerate(zip(values, indices))
                            ],
                        }
                    )

    output = {
        "schema_version": READOUT_SCHEMA,
        "source": f"jlens:{args.model}:{args.lens_repo}:{args.lens_file}",
        "synthetic": False,
        "model": args.model,
        "lens_repo": args.lens_repo,
        "lens_file": args.lens_file,
        "positions": positions,
        "top_k": args.top_k,
        "readouts": rows,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    return 0


def choose_device(requested: str, torch_module: Any) -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    raise SystemExit(main())
