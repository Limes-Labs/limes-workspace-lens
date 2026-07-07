from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .eval_artifacts import build_compatibility
from .jlens_adapter import AdapterError, public_identifier, sha256_file
from .schema import (
    GRADIENT_ATTRIBUTION_SCHEMA,
    public_artifact_path_label,
    validate_gradient_attribution,
)


@dataclass(frozen=True)
class ReadoutTarget:
    prompt_id: str
    position: str | int
    layer: int
    token: str
    rank: int
    score: float | None
    token_id: int | None
    readout_row_index: int


def prompts_by_id(spec: dict[str, Any]) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for prompt in spec.get("prompts", []):
        if isinstance(prompt, dict):
            prompt_id = prompt.get("id")
            text = prompt.get("text")
            if isinstance(prompt_id, str) and isinstance(text, str):
                prompts[prompt_id] = text
    return prompts


def parse_prompt_ids(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    prompt_ids = {part.strip() for part in value.split(",") if part.strip()}
    if not prompt_ids:
        raise AdapterError("--prompt-ids must contain at least one prompt id")
    return prompt_ids


def select_readout_targets(
    readouts: dict[str, Any],
    *,
    readout_rank: int,
    prompt_ids: set[str] | None = None,
    max_rows: int | None = None,
) -> list[ReadoutTarget]:
    if readout_rank <= 0:
        raise AdapterError("--readout-rank must be a positive integer")
    if max_rows is not None and max_rows <= 0:
        raise AdapterError("--max-rows must be a positive integer")
    rows = readouts.get("readouts")
    if not isinstance(rows, list):
        raise AdapterError("readouts artifact must contain a readouts list")
    selected: list[ReadoutTarget] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or (prompt_ids is not None and prompt_id not in prompt_ids):
            continue
        token = _top_token_at_rank(row, readout_rank)
        if token is None:
            continue
        score = token.get("score")
        token_id = token.get("token_id")
        selected.append(
            ReadoutTarget(
                prompt_id=prompt_id,
                position=row["position"],
                layer=row["layer"],
                token=token["token"],
                rank=token["rank"],
                score=float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
                token_id=token_id if isinstance(token_id, int) and not isinstance(token_id, bool) else None,
                readout_row_index=row_index,
            )
        )
        if max_rows is not None and len(selected) >= max_rows:
            break
    if not selected:
        raise AdapterError(
            f"no readout rows matched rank {readout_rank}"
            + (f" and prompt ids {sorted(prompt_ids)}" if prompt_ids else "")
        )
    return selected


def resolve_target_token_id(
    tokenizer: Any,
    target: ReadoutTarget,
    *,
    allow_token_reencode: bool,
) -> int:
    if target.token_id is not None:
        return target.token_id
    if not allow_token_reencode:
        raise AdapterError(
            "readout target is missing token_id. Re-run scripts/export_jlens_readouts.py "
            "with this version, or pass --allow-token-reencode for legacy readouts."
        )
    try:
        token_ids = tokenizer.encode(target.token, add_special_tokens=False)
    except Exception as exc:
        raise AdapterError(f"failed to encode readout token {target.token!r}: {exc}") from exc
    if not isinstance(token_ids, list) or len(token_ids) != 1 or not isinstance(token_ids[0], int):
        raise AdapterError(
            f"readout token {target.token!r} does not re-encode to exactly one token id; "
            "use a readout artifact that includes token_id"
        )
    return token_ids[0]


def resolve_sequence_position(position: str | int, sequence_length: int) -> int:
    if sequence_length <= 0:
        raise AdapterError("model produced an empty sequence")
    try:
        raw_position = int(position)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"readout position {position!r} is not an integer") from exc
    resolved = sequence_length + raw_position if raw_position < 0 else raw_position
    if resolved < 0 or resolved >= sequence_length:
        raise AdapterError(
            f"readout position {position!r} resolves to {resolved}, outside sequence length {sequence_length}"
        )
    return resolved


def build_gradient_attribution_artifact(
    *,
    spec: dict[str, Any],
    readouts_path: str | Path,
    readouts_artifact_path: str | None,
    readout_artifact_id: str,
    rows: list[dict[str, Any]],
    model: str,
    model_checkpoint: str,
    tokenizer_revision: str,
    lens_revision: str,
    fit_procedure: str,
    position_policy: str,
    layer_policy: str | None,
    prompt_suite_hash: str | None,
    attribution_top_k: int,
    seed: int | None,
    device: str,
    torch_dtype: str | None,
    model_revision: str | None,
    local_files_only: bool,
    trust_remote_code: bool,
    allow_token_reencode: bool,
    readout_rank: int,
    command: str = "python3 scripts/run_gradient_attribution.py",
) -> dict[str, Any]:
    readouts_file = Path(readouts_path)
    readouts_sha256 = sha256_file(readouts_file)
    if readouts_sha256 is None:
        raise AdapterError(f"readouts file {readouts_file} could not be hashed")
    compatibility = build_compatibility(
        spec,
        tokenizer_revision=tokenizer_revision,
        lens_revision=lens_revision,
        fit_procedure=fit_procedure,
        prompt_suite_hash=prompt_suite_hash,
        model_checkpoint=model_checkpoint,
        layer_policy=layer_policy,
        position_policy=position_policy,
    )
    artifact = {
        "schema_version": GRADIENT_ATTRIBUTION_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": f"hf-causal-lm-gradient-attribution:{public_identifier(model)}",
        "synthetic": False,
        "model": {
            "id": public_identifier(model),
            "checkpoint": model_checkpoint,
        },
        "compatibility": compatibility,
        "attribution_compatibility": {
            "operator": "gradient_x_activation",
            "target_policy": f"selected_readout_token_rank_{readout_rank}",
            "feature_types": ["input_token"],
            "attribution_top_k": attribution_top_k,
            "rank_by": "abs_score",
            "normalization": "l1_abs",
            "baseline_policy": "not_applicable",
            "hook_policy": "input_embeddings",
            "autograd_backend": "torch.autograd",
            "dtype": torch_dtype or "model_default",
        },
        "generation": {
            "mode": "hf_causal_lm_input_embedding_gradient_x_activation",
            "command": command,
            "dependency_profile": "torch-transformers-real-model",
            "seed": seed,
            "config": {
                "adapter_version": __version__,
                "model": public_identifier(model),
                "model_revision": model_revision,
                "tokenizer_revision": tokenizer_revision,
                "device": device,
                "torch_dtype": torch_dtype or "model_default",
                "local_files_only": local_files_only,
                "trust_remote_code": trust_remote_code,
                "allow_token_reencode": allow_token_reencode,
                "readout_artifact_id": readout_artifact_id,
                "readouts_path": readouts_artifact_path
                or public_artifact_path_label(readouts_file),
            },
        },
        "input_artifacts": [
            {
                "kind": "readouts",
                "path": readouts_artifact_path or public_artifact_path_label(readouts_file),
                "sha256": readouts_sha256,
            }
        ],
        "rows": rows,
    }
    return artifact


def build_attribution_row(
    *,
    target: ReadoutTarget,
    target_token_id: int,
    readout_artifact_id: str,
    attributions: list[dict[str, Any]],
) -> dict[str, Any]:
    row_id = (
        f"{target.prompt_id}:readout-row-{target.readout_row_index}:"
        f"layer-{target.layer}:rank-{target.rank}:gradient-x-activation"
    )
    abs_total = sum(float(item["abs_score"]) for item in attributions)
    normalized_total = sum(float(item.get("normalized_abs", 0.0)) for item in attributions)
    omitted_mass = 0.0 if abs_total == 0 else max(0.0, 1.0 - min(1.0, normalized_total))
    target_payload = {
        "kind": "readout_token",
        "token": target.token,
        "token_id": target_token_id,
        "rank": target.rank,
        "description": "Selected readout token logit used as the gradient target.",
        "artifact_ref": readout_artifact_id,
    }
    if target.score is not None:
        target_payload["score"] = target.score
    return {
        "row_id": row_id,
        "prompt_id": target.prompt_id,
        "position": target.position,
        "layer": target.layer,
        "target": target_payload,
        "condition": {
            "kind": "observed",
            "control_id": None,
            "alignment_policy": "same_prompt_position",
        },
        "attributions": attributions,
        "quality": {
            "finite_values": True,
            "target_found_in_readouts": True,
            "autograd_enabled": True,
            "nonzero_total_attribution": abs_total > 0,
            "completeness_delta": omitted_mass,
        },
    }


def compute_gradient_x_activation(
    *,
    torch_module: Any,
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    target_token_id: int,
    target_position: str | int,
    attribution_top_k: int,
    device: str,
) -> list[dict[str, Any]]:
    encoded = tokenizer(prompt_text, return_tensors="pt")
    if not isinstance(encoded, dict):
        try:
            encoded = dict(encoded)
        except Exception as exc:
            raise AdapterError("tokenizer output must be mapping-like") from exc
    if "input_ids" not in encoded:
        raise AdapterError("tokenizer output did not include input_ids")
    input_ids = _to_device(encoded["input_ids"], device)
    attention_mask = _to_device(encoded["attention_mask"], device) if "attention_mask" in encoded else None
    shape = getattr(input_ids, "shape", None)
    if shape is None or len(shape) != 2 or int(shape[0]) != 1:
        raise AdapterError("gradient runner expects one prompt at a time with 2D input_ids")

    try:
        embedding_layer = model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids).detach().clone()
        inputs_embeds.requires_grad_(True)
    except Exception as exc:
        raise AdapterError(f"failed to build differentiable input embeddings: {exc}") from exc

    _zero_grad(model)
    forward_kwargs = {"inputs_embeds": inputs_embeds}
    if attention_mask is not None:
        forward_kwargs["attention_mask"] = attention_mask
    try:
        outputs = model(**forward_kwargs)
    except Exception as exc:
        raise AdapterError(f"model forward pass failed: {exc}") from exc
    logits = getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, dict):
        logits = outputs.get("logits")
    if logits is None:
        raise AdapterError("model output did not include logits")
    logits_shape = getattr(logits, "shape", None)
    if logits_shape is None or len(logits_shape) != 3:
        raise AdapterError("model logits must have shape [batch, sequence, vocabulary]")
    if int(logits_shape[0]) != 1:
        raise AdapterError("gradient runner expects batch size 1")
    sequence_length = int(logits_shape[1])
    vocabulary_size = int(logits_shape[2])
    if target_token_id < 0 or target_token_id >= vocabulary_size:
        raise AdapterError(
            f"target token id {target_token_id} is outside vocabulary size {vocabulary_size}"
        )
    position_index = resolve_sequence_position(target_position, sequence_length)
    try:
        target_logit = logits[0, position_index, target_token_id]
        target_logit.backward()
    except Exception as exc:
        raise AdapterError(f"backward pass failed for selected target logit: {exc}") from exc
    gradient = getattr(inputs_embeds, "grad", None)
    if gradient is None:
        raise AdapterError("input embedding gradients were not populated")
    scores = (gradient * inputs_embeds).sum(dim=-1)[0]
    return _rank_input_token_scores(
        torch_module=torch_module,
        tokenizer=tokenizer,
        input_ids=input_ids,
        scores=scores,
        top_k=attribution_top_k,
    )


def validate_built_artifact(artifact: dict[str, Any], spec: dict[str, Any]) -> None:
    errors = validate_gradient_attribution(artifact, spec)
    if errors:
        raise AdapterError("\n".join(errors))


def _top_token_at_rank(row: dict[str, Any], readout_rank: int) -> dict[str, Any] | None:
    for token in row.get("top_tokens", []):
        if isinstance(token, dict) and token.get("rank") == readout_rank:
            if not isinstance(token.get("token"), str) or not isinstance(token.get("rank"), int):
                return None
            return token
    return None


def _rank_input_token_scores(
    *,
    torch_module: Any,
    tokenizer: Any,
    input_ids: Any,
    scores: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise AdapterError("--attribution-top-k must be a positive integer")
    try:
        detached_scores = scores.detach()
        if hasattr(torch_module, "isfinite") and not bool(torch_module.isfinite(detached_scores).all().item()):
            raise AdapterError("gradient scores contained non-finite values")
        abs_scores = detached_scores.abs()
        full_abs_total = float(abs_scores.sum().detach().cpu().item())
        limit = min(top_k, int(abs_scores.numel()))
        _values, indices = torch_module.topk(abs_scores, limit)
    except AdapterError:
        raise
    except Exception as exc:
        raise AdapterError(f"failed to rank input token attributions: {exc}") from exc
    raw_items: list[dict[str, Any]] = []
    for index_value in indices.detach().cpu().tolist():
        token_position = int(index_value)
        signed_score = float(detached_scores[token_position].detach().cpu().item())
        abs_score = abs(signed_score)
        token_id = int(input_ids[0, token_position].detach().cpu().item())
        try:
            token_text = tokenizer.decode([token_id])
        except Exception:
            token_text = None
        raw_items.append(
            {
                "feature_position": token_position,
                "feature_token_id": token_id,
                "token": token_text,
                "signed_score": signed_score,
                "abs_score": abs_score,
            }
        )
    ranked: list[dict[str, Any]] = []
    for rank, item in enumerate(raw_items, start=1):
        token_text = item["token"]
        attribution = {
            "rank": rank,
            "feature_type": "input_token",
            "feature_id": f"input_token:{item['feature_position']}:{item['feature_token_id']}",
            "feature_position": item["feature_position"],
            "feature_token_id": item["feature_token_id"],
            "signed_score": item["signed_score"],
            "abs_score": item["abs_score"],
            "normalized_abs": item["abs_score"] / full_abs_total if full_abs_total > 0 else 0.0,
            "direction": _direction(item["signed_score"], item["abs_score"]),
        }
        if token_text is not None:
            attribution["token"] = token_text
            attribution["feature_text_sha256"] = hashlib.sha256(
                token_text.encode("utf-8")
            ).hexdigest()
        _assert_finite_attribution(attribution)
        ranked.append(attribution)
    return ranked


def _direction(signed_score: float, abs_score: float) -> str:
    if abs_score == 0:
        return "zero"
    if signed_score > 0:
        return "positive"
    if signed_score < 0:
        return "negative"
    return "mixed"


def _assert_finite_attribution(attribution: dict[str, Any]) -> None:
    for key in ["signed_score", "abs_score", "normalized_abs"]:
        value = attribution[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise AdapterError(f"computed attribution {key} was not finite")


def _to_device(value: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    return value


def _zero_grad(model: Any) -> None:
    if not hasattr(model, "zero_grad"):
        return
    try:
        model.zero_grad(set_to_none=True)
    except TypeError:
        model.zero_grad()
