from __future__ import annotations

import json
import math
import re
from pathlib import Path
from pathlib import PurePosixPath
from pathlib import PureWindowsPath
from typing import Any


AUDIT_SPEC_SCHEMA = "limes-workspace-lens/audit-spec.v0.1"
READOUT_SCHEMA = "limes-workspace-lens/readouts.v0.1"
REPORT_SCHEMA = "limes-workspace-lens/report.v0.1"
BEHAVIOR_EVAL_SCHEMA = "limes-workspace-lens/behavior-eval.v0.1"
CONTROL_EVAL_SCHEMA = "limes-workspace-lens/control-eval.v0.1"
CONTROL_EVAL_KINDS = {"random_direction", "neutral_token", "no_op", "prompt_variant"}
GRADIENT_ATTRIBUTION_SCHEMA = "limes-workspace-lens/gradient-attribution.v0.1"
TOKENIZER_TERM_MAP_SCHEMA = "limes-workspace-lens/tokenizer-term-map.v0.1"
GRADIENT_ATTRIBUTION_OPERATORS = {
    "attention_gradient",
    "gradient_x_activation",
    "input_gradient",
    "integrated_gradients",
    "logit_lens_gradient",
    "saliency",
}
GRADIENT_ATTRIBUTION_FEATURE_TYPES = {
    "activation_coordinate",
    "attention_head",
    "input_token",
    "neuron",
    "readout_token",
    "residual_stream",
}
REFLECTION_SCHEMA = "limes-workspace-lens/reflection-jsonl.v0.1"
INTERVENTION_SCHEMA = "limes-workspace-lens/intervention-plan.v0.1"
COMMAND_LOG_SCHEMA = "limes-workspace-lens/command-log.v0.1"
COMPUTE_MANIFEST_SCHEMA = "limes-workspace-lens/compute-manifest.v0.1"
LENS_ARTIFACT_SCHEMA = "limes-workspace-lens/lens-artifact.v0.1"

SECRET_KEY_NAMES = {
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "client_secret",
    "credential",
    "credentials",
    "hf_token",
    "openai_api_key",
    "password",
    "private_key",
    "secret",
}
REDACTED_VALUES = {"", "<redacted>", "redacted", "***", "null", "none"}
SECRET_VALUE_PATTERNS = [
    re.compile(r"\bBearer\s+[A-Za-z0-9._=-]{8,}"),
    re.compile(r"\bAuthorization:\s*(?:Bearer|token)\s+[A-Za-z0-9._=-]{8,}", re.IGNORECASE),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{12,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(
        r"\b(?:HF_TOKEN|OPENAI_API_KEY|ANTHROPIC_API_KEY|WANDB_API_KEY|AWS_SECRET_ACCESS_KEY)"
        r"\s*=\s*(?!<redacted|\$|\$\{|<env:|REDACTED|\*\*\*)\S+",
        re.IGNORECASE,
    ),
    re.compile(
        r"[?&](?:token|api_key|access_token)="
        r"(?!<redacted|\$|\$\{|<env:|REDACTED|\*\*\*)[^&\s]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"--(?:token|api-key|password|secret)(?:=|\s+)"
        r"(?!<redacted|\$|\$\{|<env:|REDACTED|\*\*\*)\S+",
        re.IGNORECASE,
    ),
]
ABSOLUTE_LOCAL_PATH_PATTERN = re.compile(
    r"(^|[\s:='\"])(/Users/|/home/|/mnt/|/nfs/|/private/|/scratch/|/tmp/|/var/folders/|/Volumes/|/gpfs/|~[/\\]|file://|\\\\|[A-Za-z]:\\)"
)
LENS_IDENTITY_KINDS = {"revision", "artifact", "adapter"}
MUTABLE_REVISION_LABELS = {"head", "latest", "main", "master", "trunk"}


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


def public_artifact_path_label(value: str | Path) -> str:
    """Return a publishable label for a path while preserving relative artifact paths."""
    text = str(value).strip()
    if not text:
        return text
    if _looks_like_absolute_local_path(text):
        return f"<local:{_path_leaf(text)}>"
    return text


def validate_public_artifact_strings(value: Any, where: str) -> list[str]:
    return _validate_public_artifact_strings(value, where)


def is_safe_relative_artifact_path(value: Any) -> bool:
    return _safe_relative_path(value)


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

    spec_metadata = {
        key: spec[key]
        for key in ["project", "model", "lens"]
        if isinstance(spec.get(key), dict)
    }
    if spec_metadata:
        errors.extend(validate_public_artifact_strings(spec_metadata, "spec.metadata"))
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
            if "token_id" in token and (
                not isinstance(token.get("token_id"), int)
                or isinstance(token.get("token_id"), bool)
                or token.get("token_id") < 0
            ):
                errors.append(f"{token_where}.token_id: must be a non-negative integer when present")
    metadata = {
        key: readouts[key]
        for key in ["source", "model", "lens_repo", "lens_file", "positions", "top_k"]
        if key in readouts
    }
    if metadata:
        errors.extend(validate_public_artifact_strings(metadata, "readouts.metadata"))
    if "lens_file" in readouts and not _safe_relative_path(readouts.get("lens_file")):
        errors.append("readouts.lens_file: must be a safe relative path")
    if "provenance" in readouts:
        provenance = readouts.get("provenance")
        if not isinstance(provenance, dict):
            errors.append("readouts.provenance: must be an object when present")
        else:
            lens = provenance.get("lens")
            if isinstance(lens, dict) and "file" in lens and not _safe_relative_path(lens.get("file")):
                errors.append("readouts.provenance.lens.file: must be a safe relative path")
            errors.extend(validate_public_artifact_strings(provenance, "readouts.provenance"))
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
    input_readouts = report.get("input_readouts")
    if isinstance(input_readouts, dict):
        _validate_report_tokenizer_term_map_summary(
            input_readouts.get("tokenizer_term_map"), errors
        )
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
    report_metadata = {
        key: report[key]
        for key in ["project", "model", "lens", "input_readouts"]
        if isinstance(report.get(key), dict)
    }
    if report_metadata:
        errors.extend(validate_public_artifact_strings(report_metadata, "report.metadata"))
    return errors


def _validate_report_tokenizer_term_map_summary(value: Any, errors: list[str]) -> None:
    if value is None:
        return
    where = "report.input_readouts.tokenizer_term_map"
    if not isinstance(value, dict):
        errors.append(f"{where}: must be null or an object")
        return
    errors.extend(
        require_keys(
            value,
            ["source", "synthetic", "path", "sha256", "tokenizer", "term_count"],
            where,
        )
    )
    if not isinstance(value.get("source"), str) or not value.get("source", "").strip():
        errors.append(f"{where}.source: must be a non-empty string")
    if not isinstance(value.get("synthetic"), bool):
        errors.append(f"{where}.synthetic: must be a boolean")
    path = value.get("path")
    if path is not None and not _safe_relative_path_or_local_label(path):
        errors.append(f"{where}.path: must be null, a safe relative path, or a local path label")
    sha256 = value.get("sha256")
    if sha256 is not None and not _valid_sha256(sha256):
        errors.append(f"{where}.sha256: must be null or a SHA-256 hex digest")
    tokenizer = value.get("tokenizer")
    if not isinstance(tokenizer, dict):
        errors.append(f"{where}.tokenizer: must be an object")
    else:
        errors.extend(require_keys(tokenizer, ["id", "revision"], f"{where}.tokenizer"))
        if not isinstance(tokenizer.get("id"), str) or not tokenizer.get("id", "").strip():
            errors.append(f"{where}.tokenizer.id: must be a non-empty string")
        revision = tokenizer.get("revision")
        if revision is not None and (
            not isinstance(revision, str) or not revision.strip()
        ):
            errors.append(f"{where}.tokenizer.revision: must be a non-empty string or null")
    if not _is_non_negative_int(value.get("term_count")):
        errors.append(f"{where}.term_count: must be a non-negative integer")


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
                errors.extend(_validate_optional_public_fields(row, ["output_text"], where))
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
            errors.extend(_validate_optional_public_fields(row, ["control_text", "output_text"], where))
    return errors


def validate_gradient_attribution(
    artifact: dict[str, Any], spec: dict[str, Any] | None = None
) -> list[str]:
    errors = require_keys(
        artifact,
        [
            "schema_version",
            "generated_utc",
            "source",
            "synthetic",
            "model",
            "compatibility",
            "attribution_compatibility",
            "generation",
            "input_artifacts",
            "rows",
        ],
        "gradient_attribution",
    )
    if artifact.get("schema_version") != GRADIENT_ATTRIBUTION_SCHEMA:
        errors.append(
            "gradient_attribution: schema_version must be "
            f"{GRADIENT_ATTRIBUTION_SCHEMA!r}, got {artifact.get('schema_version')!r}"
        )
    _validate_generated_utc(artifact.get("generated_utc"), "gradient_attribution", errors)
    if not isinstance(artifact.get("source"), str) or not artifact.get("source", "").strip():
        errors.append("gradient_attribution.source: must be a non-empty string")
    else:
        errors.extend(
            validate_public_artifact_strings(
                {"source": artifact["source"]}, "gradient_attribution"
            )
        )
    if not isinstance(artifact.get("synthetic"), bool):
        errors.append("gradient_attribution.synthetic: must be a boolean")

    model = artifact.get("model")
    if not isinstance(model, dict):
        errors.append("gradient_attribution.model: must be an object")
    else:
        errors.extend(require_keys(model, ["id", "checkpoint"], "gradient_attribution.model"))
        errors.extend(_require_non_empty_strings(model, ["id", "checkpoint"], "gradient_attribution.model"))
        errors.extend(validate_public_artifact_strings(model, "gradient_attribution.model"))

    _validate_compatibility_object(artifact.get("compatibility"), "gradient_attribution", errors)
    if isinstance(artifact.get("compatibility"), dict):
        errors.extend(
            validate_public_artifact_strings(
                artifact["compatibility"], "gradient_attribution.compatibility"
            )
        )
    if isinstance(model, dict) and isinstance(artifact.get("compatibility"), dict):
        model_checkpoint = model.get("checkpoint")
        compatibility_checkpoint = artifact["compatibility"].get("model_checkpoint")
        if (
            isinstance(model_checkpoint, str)
            and isinstance(compatibility_checkpoint, str)
            and model_checkpoint != compatibility_checkpoint
        ):
            errors.append(
                "gradient_attribution.model.checkpoint: must match "
                "gradient_attribution.compatibility.model_checkpoint"
            )

    _validate_attribution_compatibility(artifact.get("attribution_compatibility"), errors)
    _validate_gradient_generation(artifact.get("generation"), errors)
    _validate_input_artifacts(artifact.get("input_artifacts"), "gradient_attribution.input_artifacts", errors)

    attribution_compatibility = artifact.get("attribution_compatibility")
    declared_feature_types = _declared_gradient_feature_types(attribution_compatibility)
    attribution_top_k = (
        attribution_compatibility.get("attribution_top_k")
        if isinstance(attribution_compatibility, dict)
        and _is_positive_int(attribution_compatibility.get("attribution_top_k"))
        else None
    )
    _validate_gradient_rows(
        artifact.get("rows"),
        spec,
        declared_feature_types,
        attribution_top_k,
        errors,
    )
    return errors


def validate_tokenizer_term_map(
    term_map: dict[str, Any], spec: dict[str, Any] | None = None
) -> list[str]:
    errors = require_keys(
        term_map,
        [
            "schema_version",
            "generated_utc",
            "source",
            "synthetic",
            "tokenizer",
            "input_spec",
            "normalization",
            "generation",
            "terms",
        ],
        "tokenizer_term_map",
    )
    if term_map.get("schema_version") != TOKENIZER_TERM_MAP_SCHEMA:
        errors.append(
            "tokenizer_term_map: schema_version must be "
            f"{TOKENIZER_TERM_MAP_SCHEMA!r}, got {term_map.get('schema_version')!r}"
        )
    _validate_generated_utc(term_map.get("generated_utc"), "tokenizer_term_map", errors)
    if not isinstance(term_map.get("source"), str) or not term_map.get("source", "").strip():
        errors.append("tokenizer_term_map.source: must be a non-empty string")
    if not isinstance(term_map.get("synthetic"), bool):
        errors.append("tokenizer_term_map.synthetic: must be a boolean")

    tokenizer = term_map.get("tokenizer")
    if not isinstance(tokenizer, dict):
        errors.append("tokenizer_term_map.tokenizer: must be an object")
    else:
        errors.extend(require_keys(tokenizer, ["id", "revision"], "tokenizer_term_map.tokenizer"))
        if not isinstance(tokenizer.get("id"), str) or not tokenizer.get("id", "").strip():
            errors.append("tokenizer_term_map.tokenizer.id: must be a non-empty string")
        revision = tokenizer.get("revision")
        if revision is not None and (
            not isinstance(revision, str) or not revision.strip()
        ):
            errors.append(
                "tokenizer_term_map.tokenizer.revision: must be a non-empty string or null"
            )

    input_spec = term_map.get("input_spec")
    if not isinstance(input_spec, dict):
        errors.append("tokenizer_term_map.input_spec: must be an object")
    else:
        errors.extend(require_keys(input_spec, ["path", "sha256"], "tokenizer_term_map.input_spec"))
        path = input_spec.get("path")
        if path is not None and not _safe_relative_path_or_local_label(path):
            errors.append(
                "tokenizer_term_map.input_spec.path: must be null, a safe relative path, "
                "or a local path label"
            )
        sha256 = input_spec.get("sha256")
        if sha256 is not None and not _valid_sha256(sha256):
            errors.append("tokenizer_term_map.input_spec.sha256: must be null or a SHA-256 hex digest")

    normalization = term_map.get("normalization")
    if not isinstance(normalization, dict):
        errors.append("tokenizer_term_map.normalization: must be an object")
    else:
        if not isinstance(normalization.get("casefold"), bool):
            errors.append("tokenizer_term_map.normalization.casefold: must be a boolean")
        if not isinstance(normalization.get("strip"), bool):
            errors.append("tokenizer_term_map.normalization.strip: must be a boolean")
        if not _is_string_list(normalization.get("variant_policy", [])):
            errors.append("tokenizer_term_map.normalization.variant_policy: must be a list of strings")

    generation = term_map.get("generation")
    if not isinstance(generation, dict):
        errors.append("tokenizer_term_map.generation: must be an object")
    else:
        errors.extend(
            require_keys(
                generation,
                ["adapter_version", "dependency_profile", "local_files_only", "trust_remote_code"],
                "tokenizer_term_map.generation",
            )
        )
        for key in ["adapter_version", "dependency_profile"]:
            if not isinstance(generation.get(key), str) or not generation.get(key, "").strip():
                errors.append(f"tokenizer_term_map.generation.{key}: must be a non-empty string")
        for key in ["local_files_only", "trust_remote_code"]:
            if not isinstance(generation.get(key), bool):
                errors.append(f"tokenizer_term_map.generation.{key}: must be a boolean")

    _validate_tokenizer_terms(term_map.get("terms"), spec, errors)

    public_metadata = {
        key: term_map[key]
        for key in ["source", "tokenizer", "input_spec", "normalization", "generation"]
        if key in term_map
    }
    errors.extend(validate_public_artifact_strings(public_metadata, "tokenizer_term_map"))
    return errors


def validate_command_log(artifact: dict[str, Any]) -> list[str]:
    errors = require_keys(
        artifact,
        ["schema_version", "generated_utc", "compatibility", "redaction", "commands"],
        "command_log",
    )
    if artifact.get("schema_version") != COMMAND_LOG_SCHEMA:
        errors.append(
            "command_log: schema_version must be "
            f"{COMMAND_LOG_SCHEMA!r}, got {artifact.get('schema_version')!r}"
        )
    _validate_generated_utc(artifact.get("generated_utc"), "command_log", errors)
    _validate_compatibility_object(artifact.get("compatibility"), "command_log", errors)

    redaction = artifact.get("redaction")
    if not isinstance(redaction, dict):
        errors.append("command_log.redaction: must be an object")
    else:
        errors.extend(
            require_keys(redaction, ["secrets_redacted", "rules"], "command_log.redaction")
        )
        if redaction.get("secrets_redacted") is not True:
            errors.append("command_log.redaction.secrets_redacted: must be true")
        if not _is_string_list(redaction.get("rules")) or not redaction.get("rules"):
            errors.append("command_log.redaction.rules: must be a non-empty list of strings")

    commands = artifact.get("commands")
    if not isinstance(commands, list) or not commands:
        errors.append("command_log.commands: must be a non-empty list")
    else:
        seen: set[str] = set()
        for index, command in enumerate(commands):
            where = f"command_log.commands[{index}]"
            if not isinstance(command, dict):
                errors.append(f"{where}: must be an object")
                continue
            errors.extend(require_keys(command, ["id", "purpose", "command", "cwd", "exit_code"], where))
            command_id = command.get("id")
            if not isinstance(command_id, str) or not command_id.strip():
                errors.append(f"{where}.id: must be a non-empty string")
            elif command_id in seen:
                errors.append(f"{where}.id: duplicate command id {command_id!r}")
            else:
                seen.add(command_id)
            for key in ["purpose", "command"]:
                if not isinstance(command.get(key), str) or not command.get(key, "").strip():
                    errors.append(f"{where}.{key}: must be a non-empty string")
            if not _safe_relative_path(command.get("cwd")):
                errors.append(f"{where}.cwd: must be a safe relative path")
            exit_code = command.get("exit_code")
            if not isinstance(exit_code, int) or isinstance(exit_code, bool):
                errors.append(f"{where}.exit_code: must be an integer")
            if "started_utc" in command:
                _validate_iso_datetime(command.get("started_utc"), where, "started_utc", errors)
            if "finished_utc" in command:
                _validate_iso_datetime(command.get("finished_utc"), where, "finished_utc", errors)
            if "environment" in command and not isinstance(command.get("environment"), dict):
                errors.append(f"{where}.environment: must be an object when present")

    errors.extend(validate_public_artifact_strings(artifact, "command_log"))
    return errors


def validate_compute_manifest(artifact: dict[str, Any]) -> list[str]:
    errors = require_keys(
        artifact,
        ["schema_version", "generated_utc", "runtime", "hardware", "dependencies"],
        "compute_manifest",
    )
    if artifact.get("schema_version") != COMPUTE_MANIFEST_SCHEMA:
        errors.append(
            "compute_manifest: schema_version must be "
            f"{COMPUTE_MANIFEST_SCHEMA!r}, got {artifact.get('schema_version')!r}"
        )
    _validate_generated_utc(artifact.get("generated_utc"), "compute_manifest", errors)

    runtime = artifact.get("runtime")
    if not isinstance(runtime, dict):
        errors.append("compute_manifest.runtime: must be an object")
    else:
        errors.extend(require_keys(runtime, ["python", "platform"], "compute_manifest.runtime"))
        errors.extend(
            _require_non_empty_strings(runtime, ["python", "platform"], "compute_manifest.runtime")
        )

    hardware = artifact.get("hardware")
    if not isinstance(hardware, dict):
        errors.append("compute_manifest.hardware: must be an object")
    else:
        errors.extend(require_keys(hardware, ["accelerator", "device_count"], "compute_manifest.hardware"))
        if not isinstance(hardware.get("accelerator"), str) or not hardware.get(
            "accelerator", ""
        ).strip():
            errors.append("compute_manifest.hardware.accelerator: must be a non-empty string")
        device_count = hardware.get("device_count")
        if not isinstance(device_count, int) or isinstance(device_count, bool) or device_count < 0:
            errors.append("compute_manifest.hardware.device_count: must be a non-negative integer")

    dependencies = artifact.get("dependencies")
    if not isinstance(dependencies, dict) or not dependencies:
        errors.append("compute_manifest.dependencies: must be a non-empty object")
    else:
        for name, version in dependencies.items():
            if not isinstance(name, str) or not name.strip():
                errors.append("compute_manifest.dependencies: dependency names must be non-empty strings")
            if not isinstance(version, str) or not version.strip():
                errors.append(f"compute_manifest.dependencies.{name}: must be a non-empty string")

    if "compatibility" in artifact:
        _validate_compatibility_object(artifact.get("compatibility"), "compute_manifest", errors)
    if "resource_limits" in artifact and not isinstance(artifact.get("resource_limits"), dict):
        errors.append("compute_manifest.resource_limits: must be an object when present")
    if "notes" in artifact and not _is_string_list(artifact.get("notes")):
        errors.append("compute_manifest.notes: must be a list of strings when present")

    errors.extend(validate_public_artifact_strings(artifact, "compute_manifest"))
    return errors


def validate_lens_artifact(artifact: dict[str, Any]) -> list[str]:
    errors = require_keys(
        artifact,
        ["schema_version", "generated_utc", "compatibility", "lens"],
        "lens_artifact",
    )
    if artifact.get("schema_version") != LENS_ARTIFACT_SCHEMA:
        errors.append(
            "lens_artifact: schema_version must be "
            f"{LENS_ARTIFACT_SCHEMA!r}, got {artifact.get('schema_version')!r}"
        )
    _validate_generated_utc(artifact.get("generated_utc"), "lens_artifact", errors)
    _validate_compatibility_object(artifact.get("compatibility"), "lens_artifact", errors)

    lens = artifact.get("lens")
    if not isinstance(lens, dict):
        errors.append("lens_artifact.lens: must be an object")
    else:
        errors.extend(
            require_keys(lens, ["identity_kind", "source", "revision"], "lens_artifact.lens")
        )
        if lens.get("identity_kind") not in LENS_IDENTITY_KINDS:
            errors.append(
                "lens_artifact.lens.identity_kind: must be one of "
                f"{sorted(LENS_IDENTITY_KINDS)}"
            )
        errors.extend(_require_non_empty_strings(lens, ["source", "revision"], "lens_artifact.lens"))
        if "artifact_path" in lens and not _safe_relative_path(lens.get("artifact_path")):
            errors.append("lens_artifact.lens.artifact_path: must be a safe relative path")
        if "sha256" in lens and not _valid_sha256(lens.get("sha256")):
            errors.append("lens_artifact.lens.sha256: must be a SHA-256 hex digest")
        if lens.get("identity_kind") == "artifact" and not _valid_sha256(lens.get("sha256")):
            errors.append("lens_artifact.lens.sha256: artifact identities require SHA-256")
        revision = lens.get("revision")
        if isinstance(revision, str) and revision.strip().lower() in MUTABLE_REVISION_LABELS:
            errors.append("lens_artifact.lens.revision: must be immutable, not a mutable label")

    errors.extend(validate_public_artifact_strings(artifact, "lens_artifact"))
    return errors


def ensure_valid(errors: list[str]) -> None:
    if errors:
        raise ValidationError("\n".join(errors))


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _int_not_bool(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _string_or_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        or isinstance(value, str)
        and bool(value.strip())
    )


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _gradient_scores_are_consistent(
    signed_score: float,
    abs_score: float,
    direction: str,
) -> bool:
    tolerance = 1e-9
    signed_abs = abs(signed_score)
    if direction == "mixed":
        return abs_score + tolerance >= signed_abs and abs_score > tolerance
    if abs(abs_score - signed_abs) > tolerance:
        return False
    if direction == "positive":
        return signed_score > tolerance
    if direction == "negative":
        return signed_score < -tolerance
    return signed_abs <= tolerance and abs_score <= tolerance


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
    if isinstance(artifact.get("model"), dict):
        errors.extend(validate_public_artifact_strings(artifact["model"], f"{where}.model"))
    if isinstance(artifact.get("compatibility"), dict):
        errors.extend(
            validate_public_artifact_strings(
                artifact["compatibility"], f"{where}.compatibility"
            )
        )
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
    elif not _safe_relative_path_or_local_label(value.get("responses_path")):
        errors.append(
            f"{where}.generation.responses_path: must be a safe relative path or local path label"
        )
    if not _valid_sha256(value.get("responses_sha256")):
        errors.append(f"{where}.generation.responses_sha256: must be a SHA-256 hex digest")
    seed = value.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        errors.append(f"{where}.generation.seed: must be null or a non-negative integer")
    if not isinstance(value.get("config"), dict):
        errors.append(f"{where}.generation.config: must be an object")
    errors.extend(validate_public_artifact_strings(value, f"{where}.generation"))


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


def _validate_attribution_compatibility(value: Any, errors: list[str]) -> None:
    where = "gradient_attribution.attribution_compatibility"
    if not isinstance(value, dict):
        errors.append(f"{where}: must be an object")
        return
    errors.extend(
        require_keys(
            value,
            [
                "operator",
                "target_policy",
                "feature_types",
                "attribution_top_k",
                "rank_by",
                "normalization",
                "baseline_policy",
                "hook_policy",
                "autograd_backend",
                "dtype",
            ],
            where,
        )
    )
    for key in [
        "target_policy",
        "rank_by",
        "normalization",
        "baseline_policy",
        "hook_policy",
        "autograd_backend",
        "dtype",
    ]:
        if not isinstance(value.get(key), str) or not value.get(key, "").strip():
            errors.append(f"{where}.{key}: must be a non-empty string")
    operator = value.get("operator")
    if operator not in GRADIENT_ATTRIBUTION_OPERATORS:
        errors.append(
            f"{where}.operator: must be one of {sorted(GRADIENT_ATTRIBUTION_OPERATORS)}"
        )
    feature_types = value.get("feature_types")
    if not isinstance(feature_types, list) or not feature_types:
        errors.append(f"{where}.feature_types: must be a non-empty list")
    else:
        seen_feature_types: set[str] = set()
        for index, feature_type in enumerate(feature_types):
            if feature_type not in GRADIENT_ATTRIBUTION_FEATURE_TYPES:
                errors.append(
                    f"{where}.feature_types[{index}]: must be one of "
                    f"{sorted(GRADIENT_ATTRIBUTION_FEATURE_TYPES)}"
                )
            elif feature_type in seen_feature_types:
                errors.append(f"{where}.feature_types[{index}]: duplicate feature type {feature_type!r}")
            else:
                seen_feature_types.add(feature_type)
    if not _is_positive_int(value.get("attribution_top_k")):
        errors.append(f"{where}.attribution_top_k: must be a positive integer")
    if "steps" in value and not _is_positive_int(value.get("steps")):
        errors.append(f"{where}.steps: must be a positive integer when present")
    if operator == "integrated_gradients":
        baseline_policy = value.get("baseline_policy")
        if not isinstance(baseline_policy, str) or baseline_policy.strip().lower() in {"", "none"}:
            errors.append(f"{where}.baseline_policy: integrated_gradients requires a non-empty baseline policy")
        if not _is_positive_int(value.get("steps")):
            errors.append(f"{where}.steps: integrated_gradients requires a positive integer")
    errors.extend(validate_public_artifact_strings(value, where))


def _declared_gradient_feature_types(value: Any) -> set[str]:
    if not isinstance(value, dict):
        return set()
    feature_types = value.get("feature_types")
    if not isinstance(feature_types, list):
        return set()
    return {
        item
        for item in feature_types
        if isinstance(item, str) and item in GRADIENT_ATTRIBUTION_FEATURE_TYPES
    }


def _validate_gradient_generation(value: Any, errors: list[str]) -> None:
    where = "gradient_attribution.generation"
    if not isinstance(value, dict):
        errors.append(f"{where}: must be an object")
        return
    errors.extend(
        require_keys(value, ["mode", "command", "dependency_profile", "seed", "config"], where)
    )
    for key in ["mode", "command", "dependency_profile"]:
        if not isinstance(value.get(key), str) or not value.get(key, "").strip():
            errors.append(f"{where}.{key}: must be a non-empty string")
    seed = value.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        errors.append(f"{where}.seed: must be null or a non-negative integer")
    if not isinstance(value.get("config"), dict):
        errors.append(f"{where}.config: must be an object")
    errors.extend(validate_public_artifact_strings(value, where))


def _validate_input_artifacts(value: Any, where: str, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append(f"{where}: must be a non-empty list")
        return
    seen_paths: set[str] = set()
    for index, item in enumerate(value):
        item_where = f"{where}[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{item_where}: must be an object")
            continue
        errors.extend(require_keys(item, ["kind", "path", "sha256"], item_where))
        if not isinstance(item.get("kind"), str) or not item.get("kind", "").strip():
            errors.append(f"{item_where}.kind: must be a non-empty string")
        path = item.get("path")
        if not _safe_relative_path_or_local_label(path):
            errors.append(f"{item_where}.path: must be a safe relative path or local path label")
        elif path in seen_paths:
            errors.append(f"{item_where}.path: duplicate input artifact path {path!r}")
        else:
            seen_paths.add(path)
        if not _valid_sha256(item.get("sha256")):
            errors.append(f"{item_where}.sha256: must be a SHA-256 hex digest")
        errors.extend(validate_public_artifact_strings(item, item_where))


def _validate_gradient_target(value: Any, where: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{where}: must be an object")
        return
    errors.extend(require_keys(value, ["kind"], where))
    if not isinstance(value.get("kind"), str) or not value.get("kind", "").strip():
        errors.append(f"{where}.kind: must be a non-empty string")
    if "token" in value and value["token"] is not None and not isinstance(value["token"], str):
        errors.append(f"{where}.token: must be a string or null when present")
    if "rank" in value and not _is_positive_int(value.get("rank")):
        errors.append(f"{where}.rank: must be a positive integer when present")
    if "score" in value and not _is_finite_number(value.get("score")):
        errors.append(f"{where}.score: must be a finite number when present")
    if "position" in value and not _string_or_int(value.get("position")):
        errors.append(f"{where}.position: must be a string or integer when present")
    if "layer" in value and not _int_not_bool(value.get("layer")):
        errors.append(f"{where}.layer: must be an integer when present")
    public_target = {
        key: value[key]
        for key in ["kind", "description", "artifact_ref"]
        if key in value
    }
    if public_target:
        errors.extend(validate_public_artifact_strings(public_target, where))


def _validate_gradient_condition(value: Any, where: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{where}: must be an object")
        return
    errors.extend(require_keys(value, ["kind"], where))
    kind = value.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        errors.append(f"{where}.kind: must be a non-empty string")
    control_id = value.get("control_id")
    if control_id is not None and (not isinstance(control_id, str) or not control_id.strip()):
        errors.append(f"{where}.control_id: must be a non-empty string or null when present")
    if isinstance(kind, str) and kind.strip() not in {"observed", "baseline"} and control_id is None:
        errors.append(f"{where}.control_id: control conditions must name their control_id")
    if "alignment_policy" in value and (
        not isinstance(value.get("alignment_policy"), str) or not value.get("alignment_policy", "").strip()
    ):
        errors.append(f"{where}.alignment_policy: must be a non-empty string when present")
    public_condition = {
        key: value[key]
        for key in ["kind", "control_id", "alignment_policy"]
        if key in value
    }
    errors.extend(validate_public_artifact_strings(public_condition, where))


def _validate_gradient_rows(
    rows: Any,
    spec: dict[str, Any] | None,
    declared_feature_types: set[str],
    attribution_top_k: int | None,
    errors: list[str],
) -> None:
    known_prompt_ids = _prompt_ids(spec)
    if not isinstance(rows, list) or not rows:
        errors.append("gradient_attribution.rows: must be a non-empty list")
        return
    seen_row_ids: set[str] = set()
    for index, row in enumerate(rows):
        where = f"gradient_attribution.rows[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{where}: must be an object")
            continue
        errors.extend(
            require_keys(
                row,
                [
                    "row_id",
                    "prompt_id",
                    "position",
                    "layer",
                    "target",
                    "condition",
                    "attributions",
                    "quality",
                ],
                where,
            )
        )
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not row_id.strip():
            errors.append(f"{where}.row_id: must be a non-empty string")
        elif row_id in seen_row_ids:
            errors.append(f"{where}.row_id: duplicate row id {row_id!r}")
        else:
            seen_row_ids.add(row_id)
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            errors.append(f"{where}.prompt_id: must be a non-empty string")
        else:
            if known_prompt_ids and prompt_id not in known_prompt_ids:
                errors.append(f"{where}.prompt_id: unknown prompt id {prompt_id!r}")
        if not _string_or_int(row.get("position")):
            errors.append(f"{where}.position: must be a string or integer")
        if not _int_not_bool(row.get("layer")):
            errors.append(f"{where}.layer: must be an integer")
        _validate_gradient_target(row.get("target"), f"{where}.target", errors)
        _validate_gradient_condition(row.get("condition"), f"{where}.condition", errors)
        abs_total = _validate_gradient_attribution_entries(
            row.get("attributions"),
            where,
            declared_feature_types,
            attribution_top_k,
            errors,
        )
        _validate_gradient_quality(row.get("quality"), where, abs_total, errors)


def _validate_gradient_attribution_entries(
    entries: Any,
    where: str,
    declared_feature_types: set[str],
    attribution_top_k: int | None,
    errors: list[str],
) -> float | None:
    if not isinstance(entries, list) or not entries:
        errors.append(f"{where}.attributions: must be a non-empty list")
        return None
    valid = True
    if attribution_top_k is not None and len(entries) > attribution_top_k:
        errors.append(f"{where}.attributions: must not exceed attribution_top_k {attribution_top_k}")
        valid = False
    ranks: set[int] = set()
    abs_total = 0.0
    for index, item in enumerate(entries):
        item_where = f"{where}.attributions[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{item_where}: must be an object")
            valid = False
            continue
        errors.extend(
            require_keys(
                item,
                [
                    "rank",
                    "feature_type",
                    "feature_id",
                    "signed_score",
                    "abs_score",
                    "normalized_abs",
                    "direction",
                ],
                item_where,
            )
        )
        feature_type = item.get("feature_type")
        if feature_type not in GRADIENT_ATTRIBUTION_FEATURE_TYPES:
            errors.append(
                f"{item_where}.feature_type: must be one of "
                f"{sorted(GRADIENT_ATTRIBUTION_FEATURE_TYPES)}"
            )
            valid = False
        elif declared_feature_types and feature_type not in declared_feature_types:
            errors.append(
                f"{item_where}.feature_type: must be declared in "
                "gradient_attribution.attribution_compatibility.feature_types"
            )
            valid = False
        if not _string_or_int(item.get("feature_id")):
            errors.append(f"{item_where}.feature_id: must be a non-empty string or integer")
            valid = False
        elif isinstance(item.get("feature_id"), str) and not item.get("feature_id", "").strip():
            errors.append(f"{item_where}.feature_id: must be a non-empty string or integer")
            valid = False
        rank = item.get("rank")
        if not _is_positive_int(rank):
            errors.append(f"{item_where}.rank: must be a positive integer")
            valid = False
        elif rank in ranks:
            errors.append(f"{item_where}.rank: duplicate rank {rank}")
            valid = False
        else:
            ranks.add(rank)
        signed_score = item.get("signed_score")
        abs_score = item.get("abs_score")
        direction = item.get("direction")
        if not _is_finite_number(signed_score):
            errors.append(f"{item_where}.signed_score: must be a finite number")
            valid = False
        if not _is_finite_number(abs_score):
            errors.append(f"{item_where}.abs_score: must be a finite number")
            valid = False
        elif abs_score < 0:
            errors.append(f"{item_where}.abs_score: must be non-negative")
            valid = False
        else:
            abs_total += float(abs_score)
        if not _is_finite_number(item.get("normalized_abs")):
            errors.append(f"{item_where}.normalized_abs: must be a finite number")
            valid = False
        elif not 0 <= item["normalized_abs"] <= 1:
            errors.append(f"{item_where}.normalized_abs: must be between 0 and 1")
            valid = False
        if direction not in {"positive", "negative", "zero", "mixed"}:
            errors.append(f"{item_where}.direction: must be one of ['mixed', 'negative', 'positive', 'zero']")
            valid = False
        elif _is_finite_number(signed_score) and _is_finite_number(abs_score):
            if not _gradient_scores_are_consistent(float(signed_score), float(abs_score), str(direction)):
                errors.append(
                    f"{item_where}.direction: must be consistent with signed_score and abs_score"
                )
                valid = False
        if "feature_position" in item and not _string_or_int(item.get("feature_position")):
            errors.append(f"{item_where}.feature_position: must be a string or integer when present")
            valid = False
        if "feature_token_id" in item and not _int_not_bool(item.get("feature_token_id")):
            errors.append(f"{item_where}.feature_token_id: must be an integer when present")
            valid = False
        if "feature_text_sha256" in item and not _valid_sha256(item.get("feature_text_sha256")):
            errors.append(f"{item_where}.feature_text_sha256: must be a SHA-256 hex digest when present")
            valid = False
        if "layer" in item and not _int_not_bool(item.get("layer")):
            errors.append(f"{item_where}.layer: must be an integer when present")
            valid = False
        if "token" in item and item["token"] is not None and not isinstance(item["token"], str):
            errors.append(f"{item_where}.token: must be a string or null when present")
            valid = False
    return abs_total if valid else None


def _validate_gradient_quality(
    value: Any,
    where: str,
    abs_total: float | None,
    errors: list[str],
) -> None:
    quality_where = f"{where}.quality"
    if not isinstance(value, dict):
        errors.append(f"{quality_where}: must be an object")
        return
    errors.extend(
        require_keys(
            value,
            [
                "finite_values",
                "target_found_in_readouts",
                "autograd_enabled",
                "nonzero_total_attribution",
                "completeness_delta",
            ],
            quality_where,
        )
    )
    for key in ["finite_values", "target_found_in_readouts", "autograd_enabled", "nonzero_total_attribution"]:
        if not isinstance(value.get(key), bool):
            errors.append(f"{quality_where}.{key}: must be a boolean")
    if value.get("finite_values") is not True:
        errors.append(f"{quality_where}.finite_values: must be true")
    if value.get("target_found_in_readouts") is not True:
        errors.append(f"{quality_where}.target_found_in_readouts: must be true")
    if value.get("autograd_enabled") is not True:
        errors.append(f"{quality_where}.autograd_enabled: must be true")
    completeness_delta = value.get("completeness_delta")
    if completeness_delta is not None and not _is_finite_number(completeness_delta):
        errors.append(f"{quality_where}.completeness_delta: must be null or a finite number")
    if abs_total is not None and isinstance(value.get("nonzero_total_attribution"), bool):
        if value["nonzero_total_attribution"] and abs_total == 0:
            errors.append(f"{quality_where}.nonzero_total_attribution: must be false when total attribution is zero")
        if not value["nonzero_total_attribution"] and abs_total > 0:
            errors.append(f"{quality_where}.nonzero_total_attribution: must be true when total attribution is non-zero")


def _validate_tokenizer_terms(
    terms: Any,
    spec: dict[str, Any] | None,
    errors: list[str],
) -> None:
    if not isinstance(terms, list) or not terms:
        errors.append("tokenizer_term_map.terms: must be a non-empty list")
        return
    known_prompt_ids = _prompt_ids(spec)
    known_categories = set(spec.get("audit_terms", {}).keys()) if isinstance(spec, dict) else set()
    expected_keys = _expected_tokenizer_term_keys(spec)
    seen: set[tuple[str, str | None, str | None, str]] = set()
    for index, row in enumerate(terms):
        where = f"tokenizer_term_map.terms[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{where}: must be an object")
            continue
        errors.extend(
            require_keys(
                row,
                [
                    "scope",
                    "term",
                    "normalized",
                    "variants",
                    "single_token_token_ids",
                    "multi_token_variant_count",
                ],
                where,
            )
        )
        scope = row.get("scope")
        if scope not in {"audit_terms", "expected_workspace_terms"}:
            errors.append(f"{where}.scope: must be 'audit_terms' or 'expected_workspace_terms'")
        term = row.get("term")
        if not isinstance(term, str) or not term.strip():
            errors.append(f"{where}.term: must be a non-empty string")
            term_key = ""
        else:
            term_key = term
        normalized = row.get("normalized")
        if not isinstance(normalized, str) or not normalized.strip():
            errors.append(f"{where}.normalized: must be a non-empty string")
        elif term_key and normalized != _normalize_tokenizer_text(term_key):
            errors.append(
                f"{where}.normalized: must equal normalized term "
                f"{_normalize_tokenizer_text(term_key)!r}"
            )
        category = row.get("category")
        prompt_id = row.get("prompt_id")
        if scope == "audit_terms":
            if not isinstance(category, str) or not category.strip():
                errors.append(f"{where}.category: audit_terms rows must name a category")
            elif known_categories and category not in known_categories:
                errors.append(f"{where}.category: unknown audit category {category!r}")
            if prompt_id is not None:
                errors.append(f"{where}.prompt_id: audit_terms rows must not set prompt_id")
        if scope == "expected_workspace_terms":
            if not isinstance(prompt_id, str) or not prompt_id.strip():
                errors.append(f"{where}.prompt_id: expected rows must name a prompt_id")
            elif known_prompt_ids and prompt_id not in known_prompt_ids:
                errors.append(f"{where}.prompt_id: unknown prompt id {prompt_id!r}")
            if category is not None:
                errors.append(f"{where}.category: expected rows must not set category")
        dedupe_key = (
            str(scope),
            category if isinstance(category, str) else None,
            prompt_id if isinstance(prompt_id, str) else None,
            term_key,
        )
        if dedupe_key in seen:
            errors.append(f"{where}: duplicate term mapping {dedupe_key!r}")
        else:
            seen.add(dedupe_key)
        single_token_ids = row.get("single_token_token_ids")
        _validate_non_negative_int_list(
            single_token_ids,
            f"{where}.single_token_token_ids",
            errors,
            allow_empty=True,
        )
        multi_token_variant_count = row.get("multi_token_variant_count")
        if not _is_non_negative_int(multi_token_variant_count):
            errors.append(f"{where}.multi_token_variant_count: must be a non-negative integer")
        _validate_tokenizer_variants(row.get("variants"), where, errors)
        if isinstance(row.get("variants"), list):
            derived_single_token_ids = sorted(
                {
                    token_id
                    for variant in row["variants"]
                    if isinstance(variant, dict) and variant.get("single_token") is True
                    for token_id in variant.get("token_ids", [])
                    if _is_non_negative_int(token_id)
                }
            )
            if (
                isinstance(single_token_ids, list)
                and all(_is_non_negative_int(item) for item in single_token_ids)
                and sorted(single_token_ids) != derived_single_token_ids
            ):
                errors.append(
                    f"{where}.single_token_token_ids: must equal derived single-token "
                    f"variant token ids {derived_single_token_ids!r}"
                )
            derived_multi_count = sum(
                1
                for variant in row["variants"]
                if isinstance(variant, dict) and variant.get("single_token") is False
            )
            if (
                _is_non_negative_int(multi_token_variant_count)
                and multi_token_variant_count != derived_multi_count
            ):
                errors.append(
                    f"{where}.multi_token_variant_count: must equal derived multi-token "
                    f"variant count {derived_multi_count}"
                )
    if expected_keys:
        for missing in sorted(expected_keys - seen):
            errors.append(f"tokenizer_term_map.terms: missing spec term mapping {missing!r}")
        for extra in sorted(seen - expected_keys):
            errors.append(f"tokenizer_term_map.terms: term mapping not present in spec {extra!r}")


def _expected_tokenizer_term_keys(
    spec: dict[str, Any] | None,
) -> set[tuple[str, str | None, str | None, str]]:
    if not isinstance(spec, dict):
        return set()
    expected: set[tuple[str, str | None, str | None, str]] = set()
    audit_terms = spec.get("audit_terms", {})
    if isinstance(audit_terms, dict):
        for category, terms in audit_terms.items():
            if not isinstance(category, str) or not isinstance(terms, list):
                continue
            for term in terms:
                if isinstance(term, str) and term.strip():
                    expected.add(("audit_terms", category, None, term))
    prompts = spec.get("prompts", [])
    if isinstance(prompts, list):
        for prompt in prompts:
            if not isinstance(prompt, dict):
                continue
            prompt_id = prompt.get("id")
            if not isinstance(prompt_id, str) or not prompt_id.strip():
                continue
            expected_terms = prompt.get("expected_workspace_terms", [])
            if not isinstance(expected_terms, list):
                continue
            for term in expected_terms:
                if isinstance(term, str) and term.strip():
                    expected.add(("expected_workspace_terms", None, prompt_id, term))
    return expected


def _validate_tokenizer_variants(value: Any, where: str, errors: list[str]) -> None:
    variants_where = f"{where}.variants"
    if not isinstance(value, list) or not value:
        errors.append(f"{variants_where}: must be a non-empty list")
        return
    seen_kinds: set[str] = set()
    for index, variant in enumerate(value):
        item_where = f"{variants_where}[{index}]"
        if not isinstance(variant, dict):
            errors.append(f"{item_where}: must be an object")
            continue
        errors.extend(
            require_keys(
                variant,
                ["kind", "text", "normalized", "token_ids", "tokens", "single_token"],
                item_where,
            )
        )
        kind = variant.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            errors.append(f"{item_where}.kind: must be a non-empty string")
        elif kind in seen_kinds:
            errors.append(f"{item_where}.kind: duplicate variant kind {kind!r}")
        else:
            seen_kinds.add(kind)
        for key in ["text", "normalized"]:
            if not isinstance(variant.get(key), str) or not variant.get(key, "").strip():
                errors.append(f"{item_where}.{key}: must be a non-empty string")
        text = variant.get("text")
        normalized = variant.get("normalized")
        if isinstance(text, str) and text.strip() and isinstance(normalized, str) and normalized.strip():
            expected_normalized = _normalize_tokenizer_text(text)
            if normalized != expected_normalized:
                errors.append(
                    f"{item_where}.normalized: must equal normalized text "
                    f"{expected_normalized!r}"
                )
        token_ids = variant.get("token_ids")
        _validate_non_negative_int_list(token_ids, f"{item_where}.token_ids", errors)
        tokens = variant.get("tokens")
        if not _is_string_list(tokens) or not tokens:
            errors.append(f"{item_where}.tokens: must be a non-empty list of strings")
        if isinstance(token_ids, list) and isinstance(tokens, list) and len(token_ids) != len(tokens):
            errors.append(f"{item_where}.tokens: length must match token_ids")
        single_token = variant.get("single_token")
        if not isinstance(single_token, bool):
            errors.append(f"{item_where}.single_token: must be a boolean")
        elif isinstance(token_ids, list) and single_token != (len(token_ids) == 1):
            errors.append(f"{item_where}.single_token: must match whether token_ids has length 1")


def _validate_non_negative_int_list(
    value: Any,
    where: str,
    errors: list[str],
    *,
    allow_empty: bool = False,
) -> None:
    if not isinstance(value, list) or (not value and not allow_empty):
        errors.append(f"{where}: must be a {'possibly empty ' if allow_empty else ''}list of non-negative integers")
        return
    seen: set[int] = set()
    for index, item in enumerate(value):
        if not _is_non_negative_int(item):
            errors.append(f"{where}[{index}]: must be a non-negative integer")
        elif item in seen:
            errors.append(f"{where}[{index}]: duplicate token id {item}")
        else:
            seen.add(item)


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _normalize_tokenizer_text(value: str) -> str:
    return value.strip().casefold()


def _validate_generated_utc(value: Any, where: str, errors: list[str]) -> None:
    _validate_iso_datetime(value, where, "generated_utc", errors)


def _validate_iso_datetime(value: Any, where: str, field: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{where}.{field}: must be a non-empty ISO-8601 datetime string")
        return
    try:
        from datetime import datetime

        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{where}.{field}: must be a non-empty ISO-8601 datetime string")


def _validate_public_artifact_strings(value: Any, where: str) -> list[str]:
    errors: list[str] = []

    def walk(item: Any, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                child_path = f"{path}.{key}" if path else str(key)
                if isinstance(key, str) and _is_secret_key_name(key) and not _is_redacted(child):
                    errors.append(f"{child_path}: contains unredacted secret-like field")
                walk(child, child_path)
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")
            return
        if isinstance(item, str):
            if _contains_secret_like_value(item):
                errors.append(f"{path}: contains unredacted secret-like value")
            if _looks_like_absolute_local_path(item):
                errors.append(f"{path}: contains an absolute local path")

    walk(value, where)
    return errors


def _contains_secret_like_value(value: str) -> bool:
    if _is_redacted(value):
        return False
    return any(pattern.search(value) for pattern in SECRET_VALUE_PATTERNS)


def _is_redacted(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    lowered = stripped.lower()
    return (
        lowered in REDACTED_VALUES
        or lowered.startswith("<redacted:")
        or (stripped.startswith("$") and len(stripped) > 1)
        or (stripped.startswith("${") and stripped.endswith("}"))
        or (lowered.startswith("<env:") and stripped.endswith(">"))
    )


def _is_secret_key_name(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    compact = normalized.replace("_", "")
    if "redact" in compact:
        return False
    return (
        normalized in SECRET_KEY_NAMES
        or normalized == "token"
        or normalized.endswith("_token")
        or normalized.endswith("_secret")
        or ("secret" in compact)
        or ("api" in compact and "key" in compact)
        or ("auth" in compact and "token" in compact)
        or ("private" in compact and "key" in compact)
    )


def _safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    if "\\" in value or ":" in value or value.startswith("~") or re.match(r"^[A-Za-z]:", value):
        return False
    pure = PurePosixPath(value)
    return not pure.is_absolute() and ".." not in pure.parts


def _safe_relative_path_or_local_label(value: Any) -> bool:
    if isinstance(value, str) and _is_local_path_label(value):
        return True
    return _safe_relative_path(value)


def _is_local_path_label(value: str) -> bool:
    if not (value.startswith("<local:") and value.endswith(">")):
        return False
    label = value.removeprefix("<local:").removesuffix(">")
    return bool(label) and "/" not in label and "\\" not in label and ":" not in label


def _looks_like_absolute_local_path(value: str) -> bool:
    if ABSOLUTE_LOCAL_PATH_PATTERN.search(value):
        return True
    stripped = value.strip().strip("'\"")
    if Path(stripped).is_absolute():
        return True
    if PureWindowsPath(stripped).is_absolute():
        return True
    return False


def _path_leaf(value: str) -> str:
    stripped = value.strip().strip("'\"")
    if stripped.startswith("file://"):
        stripped = stripped.removeprefix("file://")
    if "\\" in stripped or PureWindowsPath(stripped).is_absolute():
        name = PureWindowsPath(stripped).name
    else:
        name = PurePosixPath(stripped).name
    return name or "path"


def _validate_optional_string(
    row: dict[str, Any],
    key: str,
    where: str,
    errors: list[str],
) -> None:
    if key in row and row[key] is not None and not isinstance(row[key], str):
        errors.append(f"{where}.{key}: must be a string when present")


def _validate_optional_public_fields(
    row: dict[str, Any],
    keys: list[str],
    where: str,
) -> list[str]:
    public_fields = {key: row[key] for key in keys if key in row and isinstance(row[key], str)}
    if not public_fields:
        return []
    return validate_public_artifact_strings(public_fields, where)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value)
    )
