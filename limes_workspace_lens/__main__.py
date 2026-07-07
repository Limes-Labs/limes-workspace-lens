from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import render_markdown_report, score_readouts
from .comparison import compare_reports, compatibility_errors, render_markdown_comparison
from .evidence import VALID_BUNDLE_STATUSES, validate_evidence_bundle
from .eval_artifacts import (
    build_behavior_eval,
    build_compatibility,
    build_control_eval,
    load_jsonl,
    parse_generation_config,
)
from .examples import example_spec
from .intervention import build_intervention_plan
from .manifest import build_manifest, parse_metadata, validate_manifest
from .reflection import build_reflection_rows
from .schema import (
    CONTROL_EVAL_KINDS,
    ensure_valid,
    load_json,
    validate_behavior_eval,
    validate_command_log,
    validate_compute_manifest,
    validate_control_eval,
    validate_lens_artifact,
    validate_audit_spec,
    validate_readouts,
    validate_report,
    write_json,
    write_jsonl,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="limes-workspace-lens")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_spec = subparsers.add_parser("init-spec", help="Write a starter audit spec.")
    init_spec.add_argument("--out", required=True)

    validate_spec = subparsers.add_parser("validate-spec", help="Validate an audit spec.")
    validate_spec.add_argument("spec")

    validate_readouts_parser = subparsers.add_parser(
        "validate-readouts", help="Validate a workspace readout artifact."
    )
    validate_readouts_parser.add_argument("readouts")
    validate_readouts_parser.add_argument("--spec")

    validate_behavior = subparsers.add_parser(
        "validate-behavior-eval", help="Validate a behavior-eval artifact."
    )
    validate_behavior.add_argument("behavior_eval")
    validate_behavior.add_argument("--spec")

    validate_control = subparsers.add_parser(
        "validate-control-eval", help="Validate a control-eval artifact."
    )
    validate_control.add_argument("control_eval")
    validate_control.add_argument("--spec")

    validate_command_log_parser = subparsers.add_parser(
        "validate-command-log", help="Validate a command-log artifact."
    )
    validate_command_log_parser.add_argument("command_log")

    validate_compute_manifest_parser = subparsers.add_parser(
        "validate-compute-manifest", help="Validate a compute-manifest artifact."
    )
    validate_compute_manifest_parser.add_argument("compute_manifest")

    validate_lens_artifact_parser = subparsers.add_parser(
        "validate-lens-artifact", help="Validate a lens-artifact identity artifact."
    )
    validate_lens_artifact_parser.add_argument("lens_artifact")

    validate_bundle = subparsers.add_parser(
        "validate-bundle", help="Validate an evidence-bundle artifact."
    )
    validate_bundle.add_argument("bundle")
    validate_bundle.add_argument("--root", default=".", help="Root for referenced artifact paths.")
    validate_bundle.add_argument(
        "--strict",
        action="store_true",
        help="Require referenced paths and SHA-256 digests to resolve under --root.",
    )
    validate_bundle.add_argument(
        "--expected-status",
        choices=sorted(VALID_BUNDLE_STATUSES),
        help="Fail unless the bundle has this status.",
    )

    export_prompts = subparsers.add_parser(
        "export-prompts", help="Export prompt text from an audit spec as JSONL."
    )
    export_prompts.add_argument("spec")
    export_prompts.add_argument("--out", required=True)

    summarize = subparsers.add_parser("summarize-readouts", help="Score a readout artifact.")
    summarize.add_argument("readouts")
    summarize.add_argument("--spec", required=True)
    summarize.add_argument("--out", required=True, help="Markdown report path.")
    summarize.add_argument("--json-out", required=True, help="Machine-readable report path.")
    summarize.add_argument("--top-k", type=positive_int, default=10)

    reflection = subparsers.add_parser(
        "build-reflection-data",
        help="Create counterfactual-reflection JSONL candidates from a spec.",
    )
    reflection.add_argument("spec")
    reflection.add_argument("--out", required=True)

    intervention = subparsers.add_parser(
        "make-intervention-plan",
        help="Create a JSON intervention plan from a spec.",
    )
    intervention.add_argument("spec")
    intervention.add_argument("--out", required=True)

    compare = subparsers.add_parser(
        "compare-reports", help="Compare two machine-readable audit reports."
    )
    compare.add_argument("--before", required=True)
    compare.add_argument("--after", required=True)
    compare.add_argument("--out", required=True, help="Markdown comparison path.")
    compare.add_argument("--json-out", required=True, help="Machine-readable comparison path.")
    compare.add_argument("--allow-incompatible", action="store_true")

    build_manifest_parser = subparsers.add_parser(
        "build-manifest", help="Build a SHA256 artifact manifest."
    )
    build_manifest_parser.add_argument("files", nargs="+")
    build_manifest_parser.add_argument("--root", default=".")
    build_manifest_parser.add_argument("--out", required=True)
    build_manifest_parser.add_argument("--command", dest="manifest_commands", action="append", default=[])
    build_manifest_parser.add_argument("--metadata", action="append", default=[])

    behavior_eval = subparsers.add_parser(
        "run-behavior-eval",
        help="Build a behavior-eval artifact from saved model-output JSONL.",
    )
    add_eval_generator_args(behavior_eval)
    behavior_eval.add_argument(
        "--include-output-text",
        action="store_true",
        help="Preserve raw output text in the artifact. By default only hashes are stored.",
    )

    control_eval = subparsers.add_parser(
        "run-control-eval",
        help="Build a control-eval artifact from saved control-output JSONL.",
    )
    add_eval_generator_args(control_eval)
    control_eval.add_argument(
        "--control-kind",
        required=True,
        choices=sorted(CONTROL_EVAL_KINDS),
        help="Control family represented by the supplied saved outputs.",
    )
    control_eval.add_argument(
        "--include-output-text",
        action="store_true",
        help="Preserve raw control and output text in the artifact. By default only hashes are stored.",
    )

    validate_manifest_parser = subparsers.add_parser(
        "validate-manifest", help="Validate a SHA256 artifact manifest."
    )
    validate_manifest_parser.add_argument("manifest")
    validate_manifest_parser.add_argument("--root", default=None)

    args = parser.parse_args(argv)

    try:
        if args.command == "init-spec":
            write_json(args.out, example_spec())
            print(args.out)
            return 0
        if args.command == "validate-spec":
            spec = load_json(args.spec)
            ensure_valid(validate_audit_spec(spec))
            print(f"valid: {args.spec}")
            return 0
        if args.command == "validate-readouts":
            readouts = load_json(args.readouts)
            spec = load_json(args.spec) if args.spec else None
            if spec is not None:
                ensure_valid(validate_audit_spec(spec))
            ensure_valid(validate_readouts(readouts, spec))
            print(f"valid: {args.readouts}")
            return 0
        if args.command == "validate-behavior-eval":
            spec = load_json(args.spec) if args.spec else None
            if spec is not None:
                ensure_valid(validate_audit_spec(spec))
            artifact = load_json(args.behavior_eval)
            ensure_valid(validate_behavior_eval(artifact, spec))
            print(f"valid: {args.behavior_eval}")
            return 0
        if args.command == "validate-control-eval":
            spec = load_json(args.spec) if args.spec else None
            if spec is not None:
                ensure_valid(validate_audit_spec(spec))
            artifact = load_json(args.control_eval)
            ensure_valid(validate_control_eval(artifact, spec))
            print(f"valid: {args.control_eval}")
            return 0
        if args.command == "validate-command-log":
            artifact = load_json(args.command_log)
            ensure_valid(validate_command_log(artifact))
            print(f"valid: {args.command_log}")
            return 0
        if args.command == "validate-compute-manifest":
            artifact = load_json(args.compute_manifest)
            ensure_valid(validate_compute_manifest(artifact))
            print(f"valid: {args.compute_manifest}")
            return 0
        if args.command == "validate-lens-artifact":
            artifact = load_json(args.lens_artifact)
            ensure_valid(validate_lens_artifact(artifact))
            print(f"valid: {args.lens_artifact}")
            return 0
        if args.command == "validate-bundle":
            bundle = load_json(args.bundle)
            ensure_valid(
                validate_evidence_bundle(
                    bundle,
                    root=args.root,
                    strict=args.strict,
                    expected_status=args.expected_status,
                )
            )
            print(f"valid: {args.bundle}")
            return 0
        if args.command == "export-prompts":
            spec = load_json(args.spec)
            ensure_valid(validate_audit_spec(spec))
            rows = [
                {
                    "id": prompt["id"],
                    "kind": prompt["kind"],
                    "text": prompt["text"],
                    "expected_workspace_terms": prompt.get("expected_workspace_terms", []),
                }
                for prompt in spec.get("prompts", [])
            ]
            write_jsonl(args.out, rows)
            print(f"{args.out}: {len(rows)} prompts")
            return 0
        if args.command == "summarize-readouts":
            spec = load_json(args.spec)
            readouts = load_json(args.readouts)
            ensure_valid(validate_audit_spec(spec))
            ensure_valid(validate_readouts(readouts, spec))
            report = score_readouts(spec, readouts, top_k=args.top_k)
            write_json(args.json_out, report)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(render_markdown_report(report), encoding="utf-8")
            print(args.out)
            return 0
        if args.command == "build-reflection-data":
            spec = load_json(args.spec)
            ensure_valid(validate_audit_spec(spec))
            rows = build_reflection_rows(spec)
            write_jsonl(args.out, rows)
            print(f"{args.out}: {len(rows)} rows")
            return 0
        if args.command == "make-intervention-plan":
            spec = load_json(args.spec)
            ensure_valid(validate_audit_spec(spec))
            write_json(args.out, build_intervention_plan(spec))
            print(args.out)
            return 0
        if args.command == "compare-reports":
            before = load_json(args.before)
            after = load_json(args.after)
            before_errors = [f"before: {error}" for error in validate_report(before)]
            after_errors = [f"after: {error}" for error in validate_report(after)]
            ensure_valid(before_errors + after_errors)
            errors = compatibility_errors(before, after)
            if errors and not args.allow_incompatible:
                ensure_valid(errors)
            comparison = compare_reports(
                before,
                after,
                compatibility_errors=errors,
                allow_incompatible=args.allow_incompatible,
            )
            write_json(args.json_out, comparison)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(render_markdown_comparison(comparison), encoding="utf-8")
            print(args.out)
            return 0
        if args.command == "build-manifest":
            manifest = build_manifest(
                args.files,
                root=args.root,
                commands=args.manifest_commands,
                metadata=parse_metadata(args.metadata),
            )
            write_json(args.out, manifest)
            print(args.out)
            return 0
        if args.command == "run-behavior-eval":
            spec = load_json(args.spec)
            ensure_valid(validate_audit_spec(spec))
            compatibility = compatibility_from_args(spec, args)
            artifact = build_behavior_eval(
                spec,
                load_jsonl(args.responses),
                compatibility=compatibility,
                responses_path=args.responses,
                model_id=args.model_id or spec["model"]["name"],
                seed=args.seed,
                generation_config=parse_generation_config(args.generation_config),
                command="limes-workspace-lens run-behavior-eval",
                include_output_text=args.include_output_text,
            )
            ensure_valid(validate_behavior_eval(artifact, spec))
            write_json(args.out, artifact)
            print(args.out)
            return 0
        if args.command == "run-control-eval":
            spec = load_json(args.spec)
            ensure_valid(validate_audit_spec(spec))
            compatibility = compatibility_from_args(spec, args)
            artifact = build_control_eval(
                spec,
                load_jsonl(args.responses),
                compatibility=compatibility,
                responses_path=args.responses,
                model_id=args.model_id or spec["model"]["name"],
                control_kind=args.control_kind,
                seed=args.seed,
                generation_config=parse_generation_config(args.generation_config),
                command="limes-workspace-lens run-control-eval",
                include_output_text=args.include_output_text,
            )
            ensure_valid(validate_control_eval(artifact, spec))
            write_json(args.out, artifact)
            print(args.out)
            return 0
        if args.command == "validate-manifest":
            manifest = load_json(args.manifest)
            root = args.root if args.root is not None else Path(args.manifest).resolve().parent
            errors = validate_manifest(manifest, root=root)
            ensure_valid(errors)
            print(f"valid: {args.manifest}")
            return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    parser.error(f"unknown command {args.command}")
    return 2


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def add_eval_generator_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("spec")
    parser.add_argument("--responses", required=True, help="Observed output JSONL file.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-id", help="Model identifier to record; defaults to spec.model.name.")
    parser.add_argument("--model-checkpoint", help="Override spec.model.checkpoint in compatibility.")
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--lens-revision", required=True)
    parser.add_argument("--fit-procedure", required=True)
    parser.add_argument("--prompt-suite-hash")
    parser.add_argument("--layer-policy")
    parser.add_argument("--position-policy", required=True)
    parser.add_argument("--seed", type=non_negative_int)
    parser.add_argument(
        "--generation-config",
        help="JSON object recording generation settings such as temperature and max_new_tokens.",
    )


def compatibility_from_args(spec: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    return build_compatibility(
        spec,
        tokenizer_revision=args.tokenizer_revision,
        lens_revision=args.lens_revision,
        fit_procedure=args.fit_procedure,
        prompt_suite_hash=args.prompt_suite_hash,
        model_checkpoint=args.model_checkpoint,
        layer_policy=args.layer_policy,
        position_policy=args.position_policy,
    )


if __name__ == "__main__":
    raise SystemExit(main())
