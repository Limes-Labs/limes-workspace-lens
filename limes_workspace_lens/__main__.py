from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import render_markdown_report, score_readouts
from .comparison import compare_reports, render_markdown_comparison
from .examples import example_spec
from .intervention import build_intervention_plan
from .manifest import build_manifest, parse_metadata, validate_manifest
from .reflection import build_reflection_rows
from .schema import (
    ensure_valid,
    load_json,
    validate_audit_spec,
    validate_readouts,
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
    summarize.add_argument("--top-k", type=int, default=10)

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

    build_manifest_parser = subparsers.add_parser(
        "build-manifest", help="Build a SHA256 artifact manifest."
    )
    build_manifest_parser.add_argument("files", nargs="+")
    build_manifest_parser.add_argument("--root", default=".")
    build_manifest_parser.add_argument("--out", required=True)
    build_manifest_parser.add_argument("--command", dest="manifest_commands", action="append", default=[])
    build_manifest_parser.add_argument("--metadata", action="append", default=[])

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
            ensure_valid(validate_readouts(readouts))
            print(f"valid: {args.readouts}")
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
            ensure_valid(validate_readouts(readouts))
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
            comparison = compare_reports(before, after)
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


if __name__ == "__main__":
    raise SystemExit(main())
