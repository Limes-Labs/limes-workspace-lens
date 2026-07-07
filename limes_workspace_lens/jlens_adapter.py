from __future__ import annotations

import hashlib
import importlib
import json
import platform
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from types import ModuleType
from typing import Any

from . import __version__


class AdapterError(RuntimeError):
    """Raised for user-actionable real-model adapter failures."""


@dataclass(frozen=True)
class ModelDeps:
    torch: ModuleType
    transformers: ModuleType


@dataclass(frozen=True)
class OptionalDeps(ModelDeps):
    jlens: ModuleType


def load_model_deps() -> ModelDeps:
    missing: list[str] = []
    modules: dict[str, ModuleType] = {}
    for name in ["torch", "transformers"]:
        try:
            modules[name] = importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise AdapterError(
            "missing optional real-model dependencies: "
            + ", ".join(missing)
            + ". Install the package with the real-model extra, for example "
            + "`python3 -m pip install '.[real-model]'`."
        )
    return ModelDeps(
        torch=modules["torch"],
        transformers=modules["transformers"],
    )


def load_optional_deps() -> OptionalDeps:
    missing: list[str] = []
    modules: dict[str, ModuleType] = {}
    for name in ["torch", "transformers", "jlens"]:
        try:
            modules[name] = importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise AdapterError(
            "missing optional real-model dependencies: "
            + ", ".join(missing)
            + ". Install torch and transformers, then install anthropics/jacobian-lens."
        )
    return OptionalDeps(
        torch=modules["torch"],
        transformers=modules["transformers"],
        jlens=modules["jlens"],
    )


def parse_positions(raw: str) -> list[int]:
    if not isinstance(raw, str) or not raw.strip():
        raise AdapterError("--positions must contain at least one integer")
    positions: list[int] = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            raise AdapterError(f"invalid --positions value {raw!r}: empty position")
        try:
            positions.append(int(stripped))
        except ValueError as exc:
            raise AdapterError(
                f"invalid --positions value {raw!r}: {stripped!r} is not an integer"
            ) from exc
    return positions


def require_positive_int(value: int | None, name: str) -> None:
    if value is not None and value <= 0:
        raise AdapterError(f"{name} must be a positive integer")


def require_non_negative_int(value: int | None, name: str) -> None:
    if value is not None and value < 0:
        raise AdapterError(f"{name} must be a non-negative integer")


def load_prompt_texts(path: Path, max_prompts: int | None = None) -> list[str]:
    require_positive_int(max_prompts, "--max-prompts")
    prompts: list[str] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AdapterError(f"{path}:{line_number}: invalid JSONL row: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise AdapterError(f"{path}:{line_number}: prompt row must be an object")
            prompt_id = row.get("id")
            if prompt_id is not None:
                if not isinstance(prompt_id, str) or not prompt_id.strip():
                    raise AdapterError(f"{path}:{line_number}: prompt id must be a non-empty string")
                if prompt_id in seen_ids:
                    raise AdapterError(f"{path}:{line_number}: duplicate prompt id {prompt_id!r}")
                seen_ids.add(prompt_id)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise AdapterError(f"{path}:{line_number}: prompt text must be a non-empty string")
            prompts.append(text)
            if max_prompts is not None and len(prompts) >= max_prompts:
                break
    if not prompts:
        raise AdapterError(f"no prompts found in {path}")
    return prompts


def choose_device(requested: str, torch_module: Any) -> str:
    if requested == "auto":
        if _cuda_available(torch_module):
            return "cuda"
        if _mps_available(torch_module):
            return "mps"
        return "cpu"
    if requested == "cuda" and not _cuda_available(torch_module):
        raise AdapterError("requested --device cuda, but CUDA is not available")
    if requested == "mps" and not _mps_available(torch_module):
        raise AdapterError("requested --device mps, but MPS is not available")
    return requested


def parse_torch_dtype(raw: str | None, torch_module: Any) -> Any:
    if raw is None:
        return None
    if raw == "auto":
        return "auto"
    attr = raw.removeprefix("torch.")
    if attr not in {"float16", "bfloat16", "float32", "float64"}:
        raise AdapterError("--torch-dtype must be one of auto, float16, bfloat16, float32, float64")
    if not hasattr(torch_module, attr):
        raise AdapterError(f"torch does not expose dtype {attr!r}")
    return getattr(torch_module, attr)


def pretrained_kwargs(
    *,
    revision: str | None,
    local_files_only: bool,
    trust_remote_code: bool,
    torch_dtype: Any = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if revision:
        kwargs["revision"] = revision
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    return kwargs


def set_seed(torch_module: Any, seed: int | None) -> None:
    require_non_negative_int(seed, "--seed")
    if seed is None:
        return
    if hasattr(torch_module, "manual_seed"):
        torch_module.manual_seed(seed)
    if hasattr(torch_module, "cuda") and hasattr(torch_module.cuda, "manual_seed_all"):
        torch_module.cuda.manual_seed_all(seed)


def build_provenance(
    *,
    model: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    lens_repo: str | None,
    lens_file: str | None,
    lens_revision: str | None,
    spec_path: Path | None,
    prompt_count: int,
    positions: list[int] | None,
    top_k: int | None,
    device: str,
    torch_dtype: str | None,
    local_files_only: bool,
    trust_remote_code: bool,
    seed: int | None,
    deps: OptionalDeps,
) -> dict[str, Any]:
    safe_file = safe_lens_file(lens_file)
    lens_path = _lens_path(lens_repo, safe_file)
    return {
        "schema_version": "limes-workspace-lens/jlens-adapter-provenance.v0.1",
        "adapter_version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "requested": public_identifier(model),
            "revision": model_revision,
        },
        "tokenizer": {
            "revision": tokenizer_revision or model_revision,
        },
        "lens": {
            "repo": public_identifier(lens_repo) if lens_repo else None,
            "file": safe_file,
            "revision": lens_revision,
            "sha256": sha256_file(lens_path) if lens_path and lens_path.exists() else None,
        },
        "spec_sha256": sha256_file(spec_path) if spec_path and spec_path.exists() else None,
        "prompt_count": prompt_count,
        "positions": positions,
        "top_k": top_k,
        "device": device,
        "torch_dtype": torch_dtype,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        "seed": seed,
        "versions": {
            "python": platform.python_version(),
            "torch": _module_version(deps.torch),
            "transformers": _module_version(deps.transformers),
            "jlens": _module_version(deps.jlens),
        },
    }


def public_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return f"<local:{path.name}>"
    return value


def safe_lens_file(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AdapterError("--lens-file must be a safe relative path")
    stripped = value.strip()
    if (
        "\\" in stripped
        or ":" in stripped
        or stripped.startswith("~")
        or "://" in stripped
        or re.match(r"^[A-Za-z]:", stripped)
    ):
        raise AdapterError("--lens-file must be a safe relative path")
    path = PurePosixPath(stripped)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise AdapterError("--lens-file must be a safe relative path")
    return path.as_posix()


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _lens_path(lens_repo: str | None, lens_file: str | None) -> Path | None:
    if not lens_repo or not lens_file:
        return None
    return Path(lens_repo) / safe_lens_file(lens_file)


def _cuda_available(torch_module: Any) -> bool:
    return bool(getattr(getattr(torch_module, "cuda", None), "is_available", lambda: False)())


def _mps_available(torch_module: Any) -> bool:
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    return bool(mps and getattr(mps, "is_available", lambda: False)())


def _module_version(module: Any) -> str | None:
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None
