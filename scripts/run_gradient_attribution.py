#!/usr/bin/env python3
"""Run real HF causal-LM gradient attribution for selected readout targets.

The runner emits `limes-workspace-lens/gradient-attribution.v0.1` artifacts.
It intentionally lives outside the default package CLI because it requires
PyTorch, Transformers, model weights, and enough memory for autograd.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from limes_workspace_lens.gradient_attribution_runner import (
    build_attribution_row,
    build_gradient_attribution_artifact,
    compute_gradient_x_activation,
    parse_prompt_ids,
    prompts_by_id,
    resolve_target_token_id,
    select_readout_targets,
    validate_built_artifact,
)
from limes_workspace_lens.jlens_adapter import (
    AdapterError,
    choose_device,
    load_model_deps,
    parse_torch_dtype,
    pretrained_kwargs,
    public_identifier,
    require_positive_int,
    set_seed,
)
from limes_workspace_lens.schema import load_json, validate_audit_spec, validate_readouts, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path.")
    parser.add_argument("--spec", required=True, help="Audit spec JSON.")
    parser.add_argument("--readouts", required=True, help="Readout artifact JSON produced by a lens run.")
    parser.add_argument("--out", required=True, help="Output gradient-attribution artifact JSON.")
    parser.add_argument("--model-revision", default=None, help="Pinned model revision used for loading.")
    parser.add_argument(
        "--model-checkpoint",
        default=None,
        help="Immutable checkpoint label for compatibility metadata. Defaults to model revision, then spec model checkpoint.",
    )
    parser.add_argument(
        "--tokenizer-revision",
        default=None,
        help="Pinned tokenizer revision. Defaults to model revision, then model checkpoint.",
    )
    parser.add_argument("--lens-revision", required=True, help="Pinned lens revision or artifact id.")
    parser.add_argument("--fit-procedure", required=True, help="Human-readable lens fit procedure label.")
    parser.add_argument("--position-policy", required=True, help="Position policy, e.g. positions=-1.")
    parser.add_argument("--layer-policy", default=None, help="Optional layer policy override.")
    parser.add_argument("--prompt-suite-hash", default=None)
    parser.add_argument("--readout-rank", type=int, default=1)
    parser.add_argument("--attribution-top-k", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--prompt-ids", default=None, help="Optional comma-separated prompt ids to include.")
    parser.add_argument(
        "--readout-artifact-id",
        default="readouts",
        help="Evidence-bundle artifact id that this gradient artifact targets.",
    )
    parser.add_argument(
        "--readouts-artifact-path",
        default=None,
        help="Evidence-bundle path label for the readouts artifact. Defaults to a public path label.",
    )
    parser.add_argument(
        "--allow-token-reencode",
        action="store_true",
        help="Allow legacy readouts without token_id when token text re-encodes to exactly one id.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument(
        "--command-label",
        default="python3 scripts/run_gradient_attribution.py",
        help="Safe public command label stored in generation metadata.",
    )
    args = parser.parse_args()

    try:
        require_positive_int(args.readout_rank, "--readout-rank")
        require_positive_int(args.attribution_top_k, "--attribution-top-k")
        require_positive_int(args.max_rows, "--max-rows")
        spec = load_json(args.spec)
        spec_errors = validate_audit_spec(spec)
        if spec_errors:
            raise AdapterError("\n".join(spec_errors))
        readouts_path = Path(args.readouts)
        readouts = load_json(readouts_path)
        readout_errors = validate_readouts(readouts, spec)
        if readout_errors:
            raise AdapterError("\n".join(readout_errors))
        prompt_texts = prompts_by_id(spec)
        prompt_filter = parse_prompt_ids(args.prompt_ids)
        if prompt_filter is not None:
            unknown = sorted(prompt_filter - set(prompt_texts))
            if unknown:
                raise AdapterError(f"--prompt-ids contains ids not present in spec: {unknown}")
        targets = select_readout_targets(
            readouts,
            readout_rank=args.readout_rank,
            prompt_ids=prompt_filter,
            max_rows=args.max_rows,
        )

        deps = load_model_deps()
        device = choose_device(args.device, deps.torch)
        set_seed(deps.torch, args.seed)
        torch_dtype = parse_torch_dtype(args.torch_dtype, deps.torch)
        model_kwargs = pretrained_kwargs(
            revision=args.model_revision,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch_dtype,
        )
        model_checkpoint = (
            args.model_checkpoint
            or args.model_revision
            or spec.get("model", {}).get("checkpoint")
            or public_identifier(args.model)
        )
        tokenizer_load_revision = args.tokenizer_revision or args.model_revision
        tokenizer_revision = args.tokenizer_revision or args.model_revision or model_checkpoint
        tokenizer_kwargs = pretrained_kwargs(
            revision=tokenizer_load_revision,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )

        try:
            model = deps.transformers.AutoModelForCausalLM.from_pretrained(
                args.model,
                **model_kwargs,
            )
        except Exception as exc:
            raise AdapterError(f"model load failed for {args.model!r}: {exc}") from exc
        try:
            model.to(device)
            model.eval()
        except Exception as exc:
            raise AdapterError(f"model device setup failed on {device!r}: {exc}") from exc
        try:
            tokenizer = deps.transformers.AutoTokenizer.from_pretrained(
                args.model,
                **tokenizer_kwargs,
            )
        except Exception as exc:
            raise AdapterError(f"tokenizer load failed for {args.model!r}: {exc}") from exc

        rows = []
        with deps.torch.enable_grad():
            for target in targets:
                prompt_text = prompt_texts.get(target.prompt_id)
                if prompt_text is None:
                    raise AdapterError(f"readout prompt id {target.prompt_id!r} is not in spec")
                target_token_id = resolve_target_token_id(
                    tokenizer,
                    target,
                    allow_token_reencode=args.allow_token_reencode,
                )
                attributions = compute_gradient_x_activation(
                    torch_module=deps.torch,
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    target_token_id=target_token_id,
                    target_position=target.position,
                    attribution_top_k=args.attribution_top_k,
                    device=device,
                )
                rows.append(
                    build_attribution_row(
                        target=target,
                        target_token_id=target_token_id,
                        readout_artifact_id=args.readout_artifact_id,
                        attributions=attributions,
                    )
                )

        artifact = build_gradient_attribution_artifact(
            spec=spec,
            readouts_path=readouts_path,
            readouts_artifact_path=args.readouts_artifact_path,
            readout_artifact_id=args.readout_artifact_id,
            rows=rows,
            model=args.model,
            model_checkpoint=str(model_checkpoint),
            tokenizer_revision=str(tokenizer_revision),
            lens_revision=args.lens_revision,
            fit_procedure=args.fit_procedure,
            position_policy=args.position_policy,
            layer_policy=args.layer_policy,
            prompt_suite_hash=args.prompt_suite_hash,
            attribution_top_k=args.attribution_top_k,
            seed=args.seed,
            device=device,
            torch_dtype=args.torch_dtype,
            model_revision=args.model_revision,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            allow_token_reencode=args.allow_token_reencode,
            readout_rank=args.readout_rank,
            command=args.command_label,
        )
        validate_built_artifact(artifact, spec)
        write_json(args.out, artifact)
        print(args.out)
        return 0
    except AdapterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
