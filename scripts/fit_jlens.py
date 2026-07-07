#!/usr/bin/env python3
"""Fit an Anthropic jlens lens from a prompt JSONL file.

The wrapper records an explicit command boundary for Limes audit runs. It does
not vendor or reimplement Anthropic's reference fitter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from limes_workspace_lens.jlens_adapter import (
    AdapterError,
    build_provenance,
    choose_device,
    load_optional_deps,
    load_prompt_texts,
    parse_torch_dtype,
    pretrained_kwargs,
    set_seed,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path.")
    parser.add_argument("--prompts-jsonl", required=True)
    parser.add_argument("--out", required=True, help="Output lens path.")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--metadata-out", default=None, help="Optional provenance JSON path.")
    args = parser.parse_args()

    try:
        deps = load_optional_deps()
        prompts_path = Path(args.prompts_jsonl)
        prompts = load_prompt_texts(prompts_path, args.max_prompts)
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
        try:
            lens = deps.jlens.fit(model, prompts=prompts, checkpoint_path=args.checkpoint_path)
        except Exception as exc:
            raise AdapterError(f"lens fit failed: {exc}") from exc
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        lens.save(str(target))
        if args.metadata_out:
            provenance = build_provenance(
                model=args.model,
                model_revision=args.model_revision,
                tokenizer_revision=args.tokenizer_revision,
                lens_repo=str(target.parent),
                lens_file=target.name,
                lens_revision=None,
                spec_path=None,
                prompt_count=len(prompts),
                positions=None,
                top_k=None,
                device=device,
                torch_dtype=args.torch_dtype,
                local_files_only=args.local_files_only,
                trust_remote_code=args.trust_remote_code,
                seed=args.seed,
                deps=deps,
            )
            write_json(Path(args.metadata_out), provenance)
        print(args.out)
        return 0
    except AdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
