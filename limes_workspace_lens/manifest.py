from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "limes-workspace-lens/artifact-manifest.v0.1"


def build_manifest(
    files: list[str | Path],
    *,
    root: str | Path = ".",
    commands: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    records = []
    for file_path in files:
        path = Path(file_path).resolve()
        relative = _relative_to_root(path, root_path)
        if not path.is_file():
            raise FileNotFoundError(f"manifest input is not a file: {path}")
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return {
        "schema_version": MANIFEST_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": ".",
        "git_commit": current_git_commit(root_path),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "commands": commands or [],
        "metadata": metadata or {},
        "files": sorted(records, key=lambda item: item["path"]),
    }


def validate_manifest(manifest: dict[str, Any], *, root: str | Path | None = None) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append(
            f"manifest.schema_version must be {MANIFEST_SCHEMA!r}, got {manifest.get('schema_version')!r}"
        )
    manifest_root = Path(root or manifest.get("root", ".")).resolve()
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        errors.append("manifest.files must be a non-empty list")
        return errors

    seen: set[str] = set()
    for index, record in enumerate(files):
        where = f"manifest.files[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{where} must be an object")
            continue
        path_value = record.get("path")
        if not isinstance(path_value, str) or not path_value:
            errors.append(f"{where}.path must be a non-empty string")
            continue
        if path_value in seen:
            errors.append(f"{where}.path duplicates {path_value!r}")
        seen.add(path_value)
        try:
            path = _resolve_manifest_path(manifest_root, path_value)
        except ValueError as exc:
            errors.append(f"{where}.path: {exc}")
            continue
        if not path.exists():
            errors.append(f"{where}.path missing: {path_value}")
            continue
        if not path.is_file():
            errors.append(f"{where}.path is not a file: {path_value}")
            continue
        expected_size = record.get("size_bytes")
        if not isinstance(expected_size, int) or expected_size < 0:
            errors.append(f"{where}.size_bytes must be a non-negative integer")
        elif path.stat().st_size != expected_size:
            errors.append(
                f"{where}.size_bytes mismatch for {path_value}: expected {expected_size}, got {path.stat().st_size}"
            )
        expected_sha = record.get("sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            errors.append(f"{where}.sha256 must be a 64-character hex digest")
        elif sha256_file(path) != expected_sha:
            errors.append(f"{where}.sha256 mismatch for {path_value}")
    return errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def parse_metadata(values: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"metadata must be KEY=VALUE, got {value!r}")
        key, item_value = value.split("=", 1)
        if not key:
            raise ValueError(f"metadata key cannot be empty in {value!r}")
        parsed[key] = item_value
    return parsed


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{path} is outside manifest root {root}") from exc


def _resolve_manifest_path(root: Path, path_value: str) -> Path:
    candidate = (root / path_value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{path_value!r} escapes manifest root") from exc
    return candidate
