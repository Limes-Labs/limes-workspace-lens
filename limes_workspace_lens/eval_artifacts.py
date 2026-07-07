from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import (
    BEHAVIOR_EVAL_SCHEMA,
    CONTROL_EVAL_KINDS,
    CONTROL_EVAL_SCHEMA,
    ValidationError,
)


DEFAULT_BEHAVIOR_METRICS = [
    {
        "name": "nonempty_output",
        "description": "The observed model output is not empty after trimming whitespace.",
        "pass_condition": "output.strip() is non-empty",
    },
    {
        "name": "forbidden_surface_terms_absent",
        "description": "Prompt-specific terms marked as surface_output_should_not_contain do not appear in the observed output.",
        "pass_condition": "no forbidden term occurs as a case-insensitive substring",
    },
]

DEFAULT_CONTROL_METRICS = [
    {
        "name": "nonempty_output",
        "description": "The observed control-run output is not empty after trimming whitespace.",
        "pass_condition": "output.strip() is non-empty",
    },
    {
        "name": "control_text_recorded",
        "description": "The control prompt text that produced the output is preserved by hash.",
        "pass_condition": "control_text is a non-empty string",
    },
]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValidationError(f"{path}:{line_number}: JSONL rows must be objects")
            rows.append(value)
    if not rows:
        raise ValidationError(f"{path}: must contain at least one JSONL row")
    return rows


def build_compatibility(
    spec: dict[str, Any],
    *,
    tokenizer_revision: str,
    lens_revision: str,
    fit_procedure: str,
    prompt_suite_hash: str | None = None,
    model_checkpoint: str | None = None,
    layer_policy: str | None = None,
    position_policy: str,
) -> dict[str, Any]:
    lens = spec.get("lens", {})
    return {
        "model_checkpoint": model_checkpoint or spec.get("model", {}).get("checkpoint", ""),
        "tokenizer_revision": tokenizer_revision,
        "lens_source": lens.get("source", ""),
        "lens_revision": lens_revision,
        "prompt_suite_hash": prompt_suite_hash or prompt_hash(spec),
        "top_k": lens.get("top_k"),
        "layer_policy": layer_policy or _layer_policy(lens),
        "position_policy": position_policy,
        "fit_procedure": fit_procedure,
    }


def build_behavior_eval(
    spec: dict[str, Any],
    response_rows: list[dict[str, Any]],
    *,
    compatibility: dict[str, Any],
    responses_path: str,
    model_id: str,
    seed: int | None = None,
    generation_config: dict[str, Any] | None = None,
    command: str = "run-behavior-eval",
    include_output_text: bool = False,
) -> dict[str, Any]:
    prompts = _prompts_by_id(spec)
    responses = _unique_rows_by_prompt(response_rows, prompts, rows_name="behavior responses")
    rows = [
        _behavior_row(
            prompt,
            responses[prompt_id],
            include_output_text=include_output_text,
        )
        for prompt_id, prompt in prompts.items()
    ]
    return {
        "schema_version": BEHAVIOR_EVAL_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "observed-model-outputs-jsonl",
        "model": {
            "id": model_id,
            "checkpoint": compatibility.get("model_checkpoint", ""),
        },
        "compatibility": compatibility,
        "generation": {
            "mode": "evaluate-saved-model-outputs",
            "command": command,
            "dependency_profile": "stdlib-only-no-model-execution",
            "responses_path": responses_path,
            "responses_sha256": sha256_file(Path(responses_path)),
            "seed": seed,
            "config": generation_config or {},
        },
        "metric_definitions": DEFAULT_BEHAVIOR_METRICS,
        "rows": rows,
    }


def build_control_eval(
    spec: dict[str, Any],
    response_rows: list[dict[str, Any]],
    *,
    compatibility: dict[str, Any],
    responses_path: str,
    model_id: str,
    control_kind: str,
    seed: int | None = None,
    generation_config: dict[str, Any] | None = None,
    command: str = "run-control-eval",
    include_output_text: bool = False,
) -> dict[str, Any]:
    if control_kind not in CONTROL_EVAL_KINDS:
        raise ValidationError(f"control kind must be one of {sorted(CONTROL_EVAL_KINDS)}")
    prompts = _prompts_by_id(spec)
    grouped = _control_rows_by_prompt(response_rows, prompts)
    rows = []
    for prompt_id, prompt in prompts.items():
        for row in grouped[prompt_id]:
            rows.append(
                _control_row(
                    prompt,
                    row,
                    default_control_kind=control_kind,
                    include_output_text=include_output_text,
                )
            )
    return {
        "schema_version": CONTROL_EVAL_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "observed-control-outputs-jsonl",
        "model": {
            "id": model_id,
            "checkpoint": compatibility.get("model_checkpoint", ""),
        },
        "compatibility": compatibility,
        "generation": {
            "mode": "evaluate-saved-control-outputs",
            "command": command,
            "dependency_profile": "stdlib-only-no-model-execution",
            "responses_path": responses_path,
            "responses_sha256": sha256_file(Path(responses_path)),
            "seed": seed,
            "config": generation_config or {},
        },
        "control": {
            "kind": control_kind,
            "description": "Control outputs were supplied by an external runner and evaluated without model dependencies.",
        },
        "metric_definitions": DEFAULT_CONTROL_METRICS,
        "rows": rows,
    }


def parse_generation_config(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValidationError("generation config must decode to a JSON object")
    return parsed


def prompt_hash(spec: dict[str, Any]) -> str:
    prompts = spec.get("prompts", [])
    stable = [
        {
            "id": prompt.get("id"),
            "kind": prompt.get("kind"),
            "text": prompt.get("text"),
            "expected_workspace_terms": prompt.get("expected_workspace_terms", []),
            "surface_output_should_not_contain": prompt.get(
                "surface_output_should_not_contain", []
            ),
        }
        for prompt in prompts
        if isinstance(prompt, dict)
    ]
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _behavior_row(
    prompt: dict[str, Any],
    response: dict[str, Any],
    *,
    include_output_text: bool,
) -> dict[str, Any]:
    output = _required_string(response, "output", "behavior response")
    forbidden_terms = [
        term
        for term in prompt.get("surface_output_should_not_contain", [])
        if isinstance(term, str) and term
    ]
    forbidden_hits = _substring_hits(output, forbidden_terms)
    metrics = {
        "nonempty_output": {
            "passed": bool(output.strip()),
            "value": bool(output.strip()),
        },
        "forbidden_surface_terms_absent": {
            "passed": not forbidden_hits,
            "applicable": bool(forbidden_terms),
            "forbidden_terms": forbidden_terms,
            "matched_terms": forbidden_hits,
        },
    }
    row = {
        "prompt_id": prompt["id"],
        "kind": prompt.get("kind"),
        "response_id": response.get("response_id"),
        "output_sha256": sha256_text(output),
        "output_chars": len(output),
        "finish_reason": response.get("finish_reason"),
        "metrics": metrics,
        "passed": all(metric["passed"] for metric in metrics.values()),
    }
    if include_output_text:
        row["output_text"] = output
    return row


def _control_row(
    prompt: dict[str, Any],
    response: dict[str, Any],
    *,
    default_control_kind: str,
    include_output_text: bool,
) -> dict[str, Any]:
    output = _required_string(response, "output", "control response")
    control_text = _required_string(response, "control_text", "control response")
    control_kind = response.get("control_kind") or default_control_kind
    if not isinstance(control_kind, str) or not control_kind.strip():
        raise ValidationError("control response.control_kind: must be a non-empty string")
    if control_kind not in CONTROL_EVAL_KINDS:
        raise ValidationError(f"control response.control_kind: must be one of {sorted(CONTROL_EVAL_KINDS)}")
    if control_kind != default_control_kind:
        raise ValidationError(
            "control response.control_kind: must match the requested control kind "
            f"{default_control_kind!r}"
        )
    control_id = response.get("control_id") or f"{prompt['id']}:{control_kind}"
    if not isinstance(control_id, str) or not control_id.strip():
        raise ValidationError("control response.control_id: must be a non-empty string")
    metrics = {
        "nonempty_output": {
            "passed": bool(output.strip()),
            "value": bool(output.strip()),
        },
        "control_text_recorded": {
            "passed": bool(control_text.strip()),
            "value": bool(control_text.strip()),
        },
    }
    row = {
        "prompt_id": prompt["id"],
        "kind": prompt.get("kind"),
        "control_id": control_id,
        "control_kind": control_kind,
        "control_text_sha256": sha256_text(control_text),
        "output_sha256": sha256_text(output),
        "output_chars": len(output),
        "finish_reason": response.get("finish_reason"),
        "metrics": metrics,
        "passed": all(metric["passed"] for metric in metrics.values()),
    }
    if include_output_text:
        row["control_text"] = control_text
        row["output_text"] = output
    return row


def _unique_rows_by_prompt(
    rows: list[dict[str, Any]],
    prompts: dict[str, dict[str, Any]],
    *,
    rows_name: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValidationError(f"{rows_name}[{index}].prompt_id: must be a non-empty string")
        if prompt_id not in prompts:
            raise ValidationError(f"{rows_name}[{index}].prompt_id: unknown prompt id {prompt_id!r}")
        if prompt_id in grouped:
            raise ValidationError(f"{rows_name}: duplicate row for prompt id {prompt_id!r}")
        grouped[prompt_id] = row
    missing = sorted(set(prompts) - set(grouped))
    if missing:
        raise ValidationError(f"{rows_name}: missing rows for prompt ids {missing}")
    return grouped


def _control_rows_by_prompt(
    rows: list[dict[str, Any]], prompts: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {prompt_id: [] for prompt_id in prompts}
    seen_control_ids: set[str] = set()
    for index, row in enumerate(rows):
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValidationError(f"control responses[{index}].prompt_id: must be a non-empty string")
        if prompt_id not in prompts:
            raise ValidationError(
                f"control responses[{index}].prompt_id: unknown prompt id {prompt_id!r}"
            )
        control_id = row.get("control_id") or f"{prompt_id}:{row.get('control_kind', 'control')}"
        if not isinstance(control_id, str) or not control_id.strip():
            raise ValidationError(
                f"control responses[{index}].control_id: must be a non-empty string"
            )
        if control_id in seen_control_ids:
            raise ValidationError(f"control responses: duplicate control id {control_id!r}")
        seen_control_ids.add(control_id)
        grouped[prompt_id].append(row)
    missing = sorted(prompt_id for prompt_id, prompt_rows in grouped.items() if not prompt_rows)
    if missing:
        raise ValidationError(f"control responses: missing rows for prompt ids {missing}")
    return grouped


def _prompts_by_id(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {prompt["id"]: prompt for prompt in spec.get("prompts", [])}


def _required_string(row: dict[str, Any], key: str, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValidationError(f"{where}.{key}: must be a string")
    return value


def _substring_hits(output: str, terms: list[str]) -> list[str]:
    lowered = output.lower()
    return sorted({term for term in terms if term.lower() in lowered})


def _layer_policy(lens: dict[str, Any]) -> str:
    layer_range = lens.get("workspace_layer_range")
    if isinstance(layer_range, list) and len(layer_range) == 2:
        return f"workspace_layer_range={layer_range[0]}-{layer_range[1]}"
    return "unspecified-layer-policy"
