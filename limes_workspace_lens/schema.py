from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


AUDIT_SPEC_SCHEMA = "limes-workspace-lens/audit-spec.v0.1"
READOUT_SCHEMA = "limes-workspace-lens/readouts.v0.1"
REPORT_SCHEMA = "limes-workspace-lens/report.v0.1"
REFLECTION_SCHEMA = "limes-workspace-lens/reflection-jsonl.v0.1"
INTERVENTION_SCHEMA = "limes-workspace-lens/intervention-plan.v0.1"


class ValidationError(ValueError):
    """Raised when a public artifact does not satisfy the repo schema."""


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValidationError(f"{path} must contain a JSON object")
    return data


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def require_keys(data: dict[str, Any], keys: list[str], where: str) -> list[str]:
    errors: list[str] = []
    for key in keys:
        if key not in data:
            errors.append(f"{where}: missing required key '{key}'")
    return errors


def validate_audit_spec(spec: dict[str, Any]) -> list[str]:
    errors = require_keys(
        spec,
        ["schema_version", "project", "model", "lens", "prompts", "audit_terms"],
        "spec",
    )
    if spec.get("schema_version") != AUDIT_SPEC_SCHEMA:
        errors.append(
            f"spec: schema_version must be {AUDIT_SPEC_SCHEMA!r}, got {spec.get('schema_version')!r}"
        )

    for key in ["project", "model", "lens", "audit_terms"]:
        if key in spec and not isinstance(spec[key], dict):
            errors.append(f"spec.{key}: must be an object")

    if isinstance(spec.get("project"), dict):
        errors.extend(
            _require_non_empty_strings(spec["project"], ["name", "owner"], "spec.project")
        )
    if isinstance(spec.get("model"), dict):
        errors.extend(
            _require_non_empty_strings(
                spec["model"], ["name", "family", "checkpoint"], "spec.model"
            )
        )
    if isinstance(spec.get("lens"), dict):
        errors.extend(_require_non_empty_strings(spec["lens"], ["source"], "spec.lens"))
        for key in ["fit_prompt_count", "sequence_length", "top_k"]:
            if key in spec["lens"] and not _is_positive_int(spec["lens"][key]):
                errors.append(f"spec.lens.{key}: must be a positive integer")
        layer_range = spec["lens"].get("workspace_layer_range")
        if layer_range is not None and not _is_layer_range(layer_range):
            errors.append("spec.lens.workspace_layer_range: must be a two-item increasing integer list")

    prompts = spec.get("prompts")
    prompt_ids: set[str] = set()
    if not isinstance(prompts, list) or not prompts:
        errors.append("spec.prompts: must be a non-empty list")
    else:
        seen: set[str] = set()
        for index, prompt in enumerate(prompts):
            where = f"spec.prompts[{index}]"
            if not isinstance(prompt, dict):
                errors.append(f"{where}: must be an object")
                continue
            errors.extend(require_keys(prompt, ["id", "kind", "text"], where))
            prompt_id = prompt.get("id")
            if isinstance(prompt_id, str) and prompt_id:
                if prompt_id in seen:
                    errors.append(f"{where}.id: duplicate prompt id {prompt_id!r}")
                seen.add(prompt_id)
                prompt_ids.add(prompt_id)
            else:
                errors.append(f"{where}.id: must be a non-empty string")
            for key in ["kind", "text"]:
                if not isinstance(prompt.get(key), str) or not prompt.get(key, "").strip():
                    errors.append(f"{where}.{key}: must be a non-empty string")
            if "expected_workspace_terms" in prompt and not _is_string_list(
                prompt["expected_workspace_terms"]
            ):
                errors.append(f"{where}.expected_workspace_terms: must be a list of strings")

    audit_terms = spec.get("audit_terms")
    if isinstance(audit_terms, dict):
        for name, terms in audit_terms.items():
            if not isinstance(name, str) or not name:
                errors.append("spec.audit_terms: category names must be non-empty strings")
            if not _is_string_list(terms) or not terms:
                errors.append(f"spec.audit_terms.{name}: must be a non-empty list of strings")

    reflection = spec.get("reflection_training")
    if reflection is not None:
        if not isinstance(reflection, dict):
            errors.append("spec.reflection_training: must be an object")
        elif not _is_string_list(reflection.get("principles", [])):
            errors.append("spec.reflection_training.principles: must be a list of strings")

    interventions = spec.get("interventions", [])
    if interventions is not None:
        if not isinstance(interventions, list):
            errors.append("spec.interventions: must be a list")
        else:
            for index, intervention in enumerate(interventions):
                where = f"spec.interventions[{index}]"
                if not isinstance(intervention, dict):
                    errors.append(f"{where}: must be an object")
                    continue
                errors.extend(require_keys(intervention, ["id", "prompt_id", "kind"], where))
                for key in ["id", "prompt_id", "kind"]:
                    if not isinstance(intervention.get(key), str) or not intervention.get(
                        key, ""
                    ).strip():
                        errors.append(f"{where}.{key}: must be a non-empty string")
                prompt_id = intervention.get("prompt_id")
                if isinstance(prompt_id, str) and prompt_ids and prompt_id not in prompt_ids:
                    errors.append(f"{where}.prompt_id: unknown prompt id {prompt_id!r}")

    return errors


def validate_readouts(readouts: dict[str, Any], spec: dict[str, Any] | None = None) -> list[str]:
    errors = require_keys(readouts, ["schema_version", "source", "synthetic", "readouts"], "readouts")
    if readouts.get("schema_version") != READOUT_SCHEMA:
        errors.append(
            f"readouts: schema_version must be {READOUT_SCHEMA!r}, got {readouts.get('schema_version')!r}"
        )
    if not isinstance(readouts.get("source"), str) or not readouts.get("source", "").strip():
        errors.append("readouts.source: must be a non-empty string")
    if not isinstance(readouts.get("synthetic"), bool):
        errors.append("readouts.synthetic: must be a boolean")
    known_prompt_ids = _prompt_ids(spec)
    rows = readouts.get("readouts")
    if not isinstance(rows, list) or not rows:
        errors.append("readouts.readouts: must be a non-empty list")
        return errors
    for index, row in enumerate(rows):
        where = f"readouts.readouts[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{where}: must be an object")
            continue
        errors.extend(require_keys(row, ["prompt_id", "position", "layer", "top_tokens"], where))
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            errors.append(f"{where}.prompt_id: must be a non-empty string")
        elif known_prompt_ids and prompt_id not in known_prompt_ids:
            errors.append(f"{where}.prompt_id: unknown prompt id {prompt_id!r}")
        if not isinstance(row.get("position"), (str, int)) or isinstance(row.get("position"), bool):
            errors.append(f"{where}.position: must be a string or integer")
        if not isinstance(row.get("layer"), int) or isinstance(row.get("layer"), bool):
            errors.append(f"{where}.layer: must be an integer")
        top_tokens = row.get("top_tokens")
        if not isinstance(top_tokens, list) or not top_tokens:
            errors.append(f"{where}.top_tokens: must be a non-empty list")
            continue
        ranks: set[int] = set()
        for token_index, token in enumerate(top_tokens):
            token_where = f"{where}.top_tokens[{token_index}]"
            if not isinstance(token, dict):
                errors.append(f"{token_where}: must be an object")
                continue
            errors.extend(require_keys(token, ["token", "rank"], token_where))
            if not isinstance(token.get("token"), str) or not token.get("token", "").strip():
                errors.append(f"{token_where}.token: must be a non-empty string")
            rank = token.get("rank")
            if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
                errors.append(f"{token_where}.rank: must be a positive integer")
            elif rank in ranks:
                errors.append(f"{token_where}.rank: duplicate rank {rank}")
            else:
                ranks.add(rank)
            for numeric_key in ["score", "logit", "probability"]:
                if numeric_key in token and not _is_finite_number(token[numeric_key]):
                    errors.append(
                        f"{token_where}.{numeric_key}: must be a finite number when present"
                    )
    return errors


def validate_report(report: dict[str, Any]) -> list[str]:
    errors = require_keys(
        report,
        ["schema_version", "project", "model", "lens", "input_readouts", "top_k", "prompt_summaries"],
        "report",
    )
    if report.get("schema_version") != REPORT_SCHEMA:
        errors.append(
            f"report: schema_version must be {REPORT_SCHEMA!r}, got {report.get('schema_version')!r}"
        )
    for key in ["project", "model", "lens", "input_readouts"]:
        if key in report and not isinstance(report[key], dict):
            errors.append(f"report.{key}: must be an object")
    top_k = report.get("top_k")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        errors.append("report.top_k: must be a positive integer")
    category_counts = report.get("category_counts", {})
    if category_counts is not None:
        if not isinstance(category_counts, dict):
            errors.append("report.category_counts: must be an object")
        else:
            for category, count in category_counts.items():
                if not isinstance(category, str):
                    errors.append("report.category_counts: category names must be strings")
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    errors.append(f"report.category_counts.{category}: must be a non-negative integer")
    prompt_summaries = report.get("prompt_summaries")
    if not isinstance(prompt_summaries, list) or not prompt_summaries:
        errors.append("report.prompt_summaries: must be a non-empty list")
    else:
        seen: set[str] = set()
        for index, row in enumerate(prompt_summaries):
            where = f"report.prompt_summaries[{index}]"
            if not isinstance(row, dict):
                errors.append(f"{where}: must be an object")
                continue
            errors.extend(require_keys(row, ["prompt_id", "status"], where))
            prompt_id = row.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id:
                errors.append(f"{where}.prompt_id: must be a non-empty string")
            elif prompt_id in seen:
                errors.append(f"{where}.prompt_id: duplicate prompt id {prompt_id!r}")
            else:
                seen.add(prompt_id)
            for count_key in ["expected_workspace_term_hits", "audit_term_hits"]:
                if count_key in row and (
                    not isinstance(row[count_key], int)
                    or isinstance(row[count_key], bool)
                    or row[count_key] < 0
                ):
                    errors.append(f"{where}.{count_key}: must be a non-negative integer")
    return errors


def ensure_valid(errors: list[str]) -> None:
    if errors:
        raise ValidationError("\n".join(errors))


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_layer_range(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        and value[0] <= value[1]
    )


def _require_non_empty_strings(data: dict[str, Any], keys: list[str], where: str) -> list[str]:
    errors: list[str] = []
    for key in keys:
        if not isinstance(data.get(key), str) or not data.get(key, "").strip():
            errors.append(f"{where}.{key}: must be a non-empty string")
    return errors


def _prompt_ids(spec: dict[str, Any] | None) -> set[str]:
    if not spec:
        return set()
    prompts = spec.get("prompts")
    if not isinstance(prompts, list):
        return set()
    return {
        prompt["id"]
        for prompt in prompts
        if isinstance(prompt, dict) and isinstance(prompt.get("id"), str)
    }
