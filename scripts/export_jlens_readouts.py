#!/usr/bin/env python3
"""Export Anthropic jlens readouts into the Limes Workspace Lens schema.

This script is optional because real model runs require model weights, a fitted
Jacobian lens, and ML dependencies. It is kept outside the core package so the
repo remains importable and testable on CPU-only machines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from limes_workspace_lens.jlens_adapter import (
    AdapterError,
    build_provenance,
    choose_device,
    load_optional_deps,
    parse_positions,
    parse_torch_dtype,
    pretrained_kwargs,
    public_identifier,
    require_positive_int,
    safe_lens_file,
    set_seed,
)


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
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--lens-revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--torch-dtype", default=None)
    args = parser.parse_args()

    try:
        from limes_workspace_lens.schema import (
            READOUT_SCHEMA,
            load_json,
            validate_audit_spec,
            validate_readouts,
        )

        deps = load_optional_deps()
        require_positive_int(args.top_k, "--top-k")
        lens_file = safe_lens_file(args.lens_file)
        positions = parse_positions(args.positions)
        spec_path = Path(args.spec)
        spec = load_json(spec_path)
        errors = validate_audit_spec(spec)
        if errors:
            raise AdapterError("\n".join(errors))

        device = choose_device(args.device, deps.torch)
        set_seed(deps.torch, args.seed)
        torch_dtype = parse_torch_dtype(args.torch_dtype, deps.torch)
        model_kwargs = pretrained_kwargs(
            revision=args.model_revision,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch_dtype,
        )
        tokenizer_kwargs = pretrained_kwargs(
            revision=args.tokenizer_revision or args.model_revision,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        try:
            hf = deps.transformers.AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        except Exception as exc:
            raise AdapterError(f"model load failed for {args.model!r}: {exc}") from exc
        try:
            hf.to(device)
            hf.eval()
        except Exception as exc:
            raise AdapterError(f"model device setup failed on {device!r}: {exc}") from exc
        try:
            tok = deps.transformers.AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
        except Exception as exc:
            raise AdapterError(f"tokenizer load failed for {args.model!r}: {exc}") from exc
        model = deps.jlens.from_hf(hf, tok)
        lens_kwargs: dict[str, Any] = {"filename": lens_file}
        if args.lens_revision:
            lens_kwargs["revision"] = args.lens_revision
        try:
            lens = deps.jlens.JacobianLens.from_pretrained(args.lens_repo, **lens_kwargs)
        except Exception as exc:
            raise AdapterError(f"lens load failed for {args.lens_repo!r}: {exc}") from exc

        rows: list[dict[str, Any]] = []
        with deps.torch.no_grad():
            for prompt in spec["prompts"]:
                lens_logits, _model_logits, _metadata = lens.apply(
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

        provenance = build_provenance(
            model=args.model,
            model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            lens_repo=args.lens_repo,
            lens_file=lens_file,
            lens_revision=args.lens_revision,
            spec_path=spec_path,
            prompt_count=len(spec["prompts"]),
            positions=positions,
            top_k=args.top_k,
            device=device,
            torch_dtype=args.torch_dtype,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            seed=args.seed,
            deps=deps,
        )
        output = {
            "schema_version": READOUT_SCHEMA,
            "source": (
                f"jlens:{public_identifier(args.model)}:"
                f"{public_identifier(args.lens_repo)}:{lens_file}"
            ),
            "synthetic": False,
            "model": public_identifier(args.model),
            "lens_repo": public_identifier(args.lens_repo),
            "lens_file": lens_file,
            "positions": positions,
            "top_k": args.top_k,
            "provenance": provenance,
            "readouts": rows,
        }
        readout_errors = validate_readouts(output, spec)
        if readout_errors:
            raise AdapterError("\n".join(readout_errors))
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(args.out)
        return 0
    except AdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
