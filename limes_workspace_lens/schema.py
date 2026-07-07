from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


AUDIT_SPEC_SCHEMA = "limes-workspace-lens/audit-spec.v0.1"
READOUT_SCHEMA = "limes-workspace-lens/readouts.v0.1"
REPORT_SCHEMA = "limes-workspace-lens/report.v0.1"
BEHAVIOR_EVAL_SCHEMA = "limes-workspace-lens/behavior-eval.v0.1"
CONTROL_EVAL_SCHEMA = "limes-workspace-lens/control-eval.v0.1"
CONTROL_EVAL_KINDS = {"random_direction", "neutral_token", "no_op", "prompt_variant"}
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


def validate_behavior_eval(
    artifact: dict[str, Any], spec: dict[str, Any] | None = None
) -> list[str]:
    errors = _validate_eval_common(
        artifact,
        expected_schema=BEHAVIOR_EVAL_SCHEMA,
        where="behavior_eval",
        spec=spec,
        allow_duplicate_prompt_ids=False,
    )
    rows = artifact.get("rows")
    if isinstance(rows, list):
        for index, row in enumerate(rows):
            where = f"behavior_eval.rows[{index}]"
            if isinstance(row, dict):
                _validate_optional_string(row, "response_id", where, errors)
                _validate_optional_string(row, "finish_reason", where, errors)
                _validate_optional_string(row, "output_text", where, errors)
    return errors


def validate_control_eval(
    artifact: dict[str, Any], spec: dict[str, Any] | None = None
) -> list[str]:
    errors = _validate_eval_common(
        artifact,
        expected_schema=CONTROL_EVAL_SCHEMA,
        where="control_eval",
        spec=spec,
        allow_duplicate_prompt_ids=True,
    )
    control = artifact.get("control")
    artifact_control_kind = None
    if not isinstance(control, dict):
        errors.append("control_eval.control: must be an object")
    else:
        errors.extend(require_keys(control, ["kind", "description"], "control_eval.control"))
        kind = control.get("kind")
        if kind not in CONTROL_EVAL_KINDS:
            errors.append(
                "control_eval.control.kind: must be one of "
                f"{sorted(CONTROL_EVAL_KINDS)}"
            )
        elif isinstance(kind, str):
            artifact_control_kind = kind
        if not isinstance(control.get("description"), str) or not control.get(
            "description", ""
        ).strip():
            errors.append("control_eval.control.description: must be a non-empty string")
    rows = artifact.get("rows")
    control_ids: set[str] = set()
    if isinstance(rows, list):
        for index, row in enumerate(rows):
            where = f"control_eval.rows[{index}]"
            if not isinstance(row, dict):
                continue
            errors.extend(
                require_keys(row, ["control_id", "control_kind", "control_text_sha256"], where)
            )
            control_id = row.get("control_id")
            if not isinstance(control_id, str) or not control_id.strip():
                errors.append(f"{where}.control_id: must be a non-empty string")
            elif control_id in control_ids:
                errors.append(f"{where}.control_id: duplicate control id {control_id!r}")
            else:
                control_ids.add(control_id)
            if not isinstance(row.get("control_kind"), str) or not row.get(
                "control_kind", ""
            ).strip():
                errors.append(f"{where}.control_kind: must be a non-empty string")
            elif row.get("control_kind") not in CONTROL_EVAL_KINDS:
                errors.append(
                    f"{where}.control_kind: must be one of {sorted(CONTROL_EVAL_KINDS)}"
                )
            elif artifact_control_kind is not None and row.get("control_kind") != artifact_control_kind:
                errors.append(
                    f"{where}.control_kind: must match control_eval.control.kind "
                    f"{artifact_control_kind!r}"
                )
            if not _valid_sha256(row.get("control_text_sha256")):
                errors.append(f"{where}.control_text_sha256: must be a SHA-256 hex digest")
            _validate_optional_string(row, "control_text", where, errors)
            _validate_optional_string(row, "output_text", where, errors)
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


def _validate_eval_common(
    artifact: dict[str, Any],
    *,
    expected_schema: str,
    where: str,
    spec: dict[str, Any] | None,
    allow_duplicate_prompt_ids: bool,
) -> list[str]:
    errors = require_keys(
        artifact,
        [
            "schema_version",
            "generated_utc",
            "source",
            "model",
            "compatibility",
            "generation",
            "metric_definitions",
            "rows",
        ],
        where,
    )
    if artifact.get("schema_version") != expected_schema:
        errors.append(
            f"{where}: schema_version must be {expected_schema!r}, got {artifact.get('schema_version')!r}"
        )
    if not isinstance(artifact.get("generated_utc"), str) or not artifact.get(
        "generated_utc", ""
    ).strip():
        errors.append(f"{where}.generated_utc: must be a non-empty string")
    if not isinstance(artifact.get("source"), str) or not artifact.get("source", "").strip():
        errors.append(f"{where}.source: must be a non-empty string")
    if not isinstance(artifact.get("model"), dict):
        errors.append(f"{where}.model: must be an object")
    else:
        errors.extend(require_keys(artifact["model"], ["id", "checkpoint"], f"{where}.model"))
        for key in ["id", "checkpoint"]:
            if not isinstance(artifact["model"].get(key), str) or not artifact["model"].get(
                key, ""
            ).strip():
                errors.append(f"{where}.model.{key}: must be a non-empty string")
    _validate_compatibility_object(artifact.get("compatibility"), where, errors)
    _validate_generation_object(artifact.get("generation"), where, errors)
    metric_names = _validate_metric_definitions(artifact.get("metric_definitions"), where, errors)
    _validate_eval_rows(
        artifact.get("rows"),
        where,
        spec,
        metric_names,
        allow_duplicate_prompt_ids,
        errors,
    )
    return errors


def _validate_compatibility_object(value: Any, where: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{where}.compatibility: must be an object")
        return
    required = [
        "model_checkpoint",
        "tokenizer_revision",
        "lens_source",
        "lens_revision",
        "prompt_suite_hash",
        "top_k",
        "layer_policy",
        "position_policy",
        "fit_procedure",
    ]
    errors.extend(require_keys(value, required, f"{where}.compatibility"))
    for key in required:
        if key == "top_k":
            if not _is_positive_int(value.get(key)):
                errors.append(f"{where}.compatibility.top_k: must be a positive integer")
        elif not isinstance(value.get(key), str) or not value.get(key, "").strip():
            errors.append(f"{where}.compatibility.{key}: must be a non-empty string")


def _validate_generation_object(value: Any, where: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{where}.generation: must be an object")
        return
    errors.extend(
        require_keys(
            value,
            [
                "mode",
                "command",
                "dependency_profile",
                "responses_path",
                "responses_sha256",
                "seed",
                "config",
            ],
            f"{where}.generation",
        )
    )
    for key in ["mode", "command", "dependency_profile"]:
        if not isinstance(value.get(key), str) or not value.get(key, "").strip():
            errors.append(f"{where}.generation.{key}: must be a non-empty string")
    if not isinstance(value.get("responses_path"), str) or not value.get(
        "responses_path", ""
    ).strip():
        errors.append(f"{where}.generation.responses_path: must be a non-empty string")
    if not _valid_sha256(value.get("responses_sha256")):
        errors.append(f"{where}.generation.responses_sha256: must be a SHA-256 hex digest")
    seed = value.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        errors.append(f"{where}.generation.seed: must be null or a non-negative integer")
    if not isinstance(value.get("config"), dict):
        errors.append(f"{where}.generation.config: must be an object")


def _validate_metric_definitions(value: Any, where: str, errors: list[str]) -> set[str]:
    metric_names: set[str] = set()
    if not isinstance(value, list) or not value:
        errors.append(f"{where}.metric_definitions: must be a non-empty list")
        return metric_names
    seen: set[str] = set()
    for index, metric in enumerate(value):
        metric_where = f"{where}.metric_definitions[{index}]"
        if not isinstance(metric, dict):
            errors.append(f"{metric_where}: must be an object")
            continue
        errors.extend(
            require_keys(metric, ["name", "description", "pass_condition"], metric_where)
        )
        name = metric.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{metric_where}.name: must be a non-empty string")
        elif name in seen:
            errors.append(f"{metric_where}.name: duplicate metric name {name!r}")
        else:
            seen.add(name)
            metric_names.add(name)
        for key in ["description", "pass_condition"]:
            if not isinstance(metric.get(key), str) or not metric.get(key, "").strip():
                errors.append(f"{metric_where}.{key}: must be a non-empty string")
    return metric_names


def _validate_eval_rows(
    rows: Any,
    where: str,
    spec: dict[str, Any] | None,
    metric_names: set[str],
    allow_duplicate_prompt_ids: bool,
    errors: list[str],
) -> None:
    known_prompt_ids = _prompt_ids(spec)
    if not isinstance(rows, list) or not rows:
        errors.append(f"{where}.rows: must be a non-empty list")
        return
    seen_prompt_ids: set[str] = set()
    for index, row in enumerate(rows):
        row_where = f"{where}.rows[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{row_where}: must be an object")
            continue
        errors.extend(
            require_keys(
                row,
                ["prompt_id", "output_sha256", "output_chars", "metrics", "passed"],
                row_where,
            )
        )
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            errors.append(f"{row_where}.prompt_id: must be a non-empty string")
        else:
            if not allow_duplicate_prompt_ids and prompt_id in seen_prompt_ids:
                errors.append(f"{row_where}.prompt_id: duplicate prompt id {prompt_id!r}")
            seen_prompt_ids.add(prompt_id)
            if known_prompt_ids and prompt_id not in known_prompt_ids:
                errors.append(f"{row_where}.prompt_id: unknown prompt id {prompt_id!r}")
        if not _valid_sha256(row.get("output_sha256")):
            errors.append(f"{row_where}.output_sha256: must be a SHA-256 hex digest")
        if not isinstance(row.get("output_chars"), int) or isinstance(
            row.get("output_chars"), bool
        ) or row.get("output_chars") < 0:
            errors.append(f"{row_where}.output_chars: must be a non-negative integer")
        if not isinstance(row.get("passed"), bool):
            errors.append(f"{row_where}.passed: must be a boolean")
        metric_passes = _validate_metric_results(row.get("metrics"), row_where, metric_names, errors)
        if isinstance(row.get("passed"), bool) and metric_passes is not None:
            expected_passed = all(metric_passes.values())
            if row["passed"] != expected_passed:
                errors.append(
                    f"{row_where}.passed: must equal all metric passed values "
                    f"({expected_passed!r})"
                )
    if known_prompt_ids:
        missing = sorted(known_prompt_ids - seen_prompt_ids)
        if missing:
            errors.append(f"{where}.rows: missing rows for prompt ids {missing}")


def _validate_metric_results(
    value: Any,
    where: str,
    metric_names: set[str],
    errors: list[str],
) -> dict[str, bool] | None:
    if not isinstance(value, dict) or not value:
        errors.append(f"{where}.metrics: must be a non-empty object")
        return None
    if metric_names:
        missing = sorted(metric_names - set(value))
        if missing:
            errors.append(f"{where}.metrics: missing metric results {missing}")
    metric_passes: dict[str, bool] = {}
    for metric_name, metric in value.items():
        metric_where = f"{where}.metrics.{metric_name}"
        if not isinstance(metric_name, str) or not metric_name:
            errors.append(f"{where}.metrics: metric names must be non-empty strings")
        elif metric_names and metric_name not in metric_names:
            errors.append(f"{metric_where}: metric is not defined in metric_definitions")
        if not isinstance(metric, dict):
            errors.append(f"{metric_where}: must be an object")
            continue
        if not isinstance(metric.get("passed"), bool):
            errors.append(f"{metric_where}.passed: must be a boolean")
        else:
            metric_passes[str(metric_name)] = metric["passed"]
    return metric_passes


def _validate_optional_string(
    row: dict[str, Any],
    key: str,
    where: str,
    errors: list[str],
) -> None:
    if key in row and row[key] is not None and not isinstance(row[key], str):
        errors.append(f"{where}.{key}: must be a string when present")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value)
    )
