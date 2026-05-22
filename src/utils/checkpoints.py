"""Checkpoint hashing and evaluation-artifact alias helpers."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


_EVAL_ARTIFACT_KEYS = (
    "summary_path",
    "predictions_path",
    "checkpoint_jsonl_path",
    "progress_path",
    "judge_ready_path",
)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_sha256(adapter_dir: str | Path | None) -> Optional[str]:
    if adapter_dir is None:
        return None
    weights_path = Path(adapter_dir) / "adapter_model.safetensors"
    if not weights_path.exists():
        return None
    return sha256_file(weights_path)


def collect_adapter_sha256(
    *,
    best_adapter_dir: str | Path | None,
    final_adapter_dir: str | Path | None,
) -> Dict[str, Optional[str]]:
    return {
        "best": adapter_sha256(best_adapter_dir),
        "final": adapter_sha256(final_adapter_dir),
    }


def adapters_match(adapter_sha_map: Dict[str, Optional[str]]) -> bool:
    best_sha = adapter_sha_map.get("best")
    final_sha = adapter_sha_map.get("final")
    return bool(best_sha) and best_sha == final_sha


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        relative_src = os.path.relpath(src, dst.parent)
        dst.symlink_to(relative_src)
    except OSError:
        shutil.copy2(src, dst)


def materialize_eval_aliases(result: Dict[str, Any], *alias_tags: str) -> None:
    source_tag = str(result.get("checkpoint_tag") or "")
    if not source_tag:
        return

    for key in _EVAL_ARTIFACT_KEYS:
        raw_path = result.get(key)
        if not raw_path:
            continue
        src = Path(str(raw_path))
        if not src.exists() or source_tag not in src.name:
            continue
        for alias_tag in alias_tags:
            if not alias_tag or alias_tag == source_tag:
                continue
            dst = src.with_name(src.name.replace(source_tag, alias_tag))
            _safe_link_or_copy(src, dst)
