"""MTVQA loading helpers for single-sample and paired training flows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_manifest(root: Path, splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload

    all_rows = [row for split_rows in splits.values() for row in split_rows]
    return {
        "dataset_name": "mtvqa_qa",
        "languages": sorted({str(row.get("language", "")).lower() for row in all_rows if row.get("language")}),
        "counts": {split: len(rows) for split, rows in splits.items()},
        "source": str(root),
    }


def load_mtvqa_dataset(dataset_dir: str) -> Dict[str, Any]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Missing MTVQA directory: {root}")

    splits: Dict[str, List[Dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        split_path = root / f"{split}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing MTVQA split file: {split_path}")
        splits[split] = read_jsonl(split_path)

    return {
        "manifest": _resolve_manifest(root, splits),
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
    }


def _pairs_filename(split: str, suffix: str) -> str:
    normalized = suffix.strip()
    if normalized and not normalized.startswith("_"):
        normalized = f"_{normalized}"
    return f"pairs_{split}{normalized}.jsonl"


def load_mtvqa_pair_dataset(dataset_dir: str, pair_suffix: str = "_merged") -> Dict[str, Any]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Missing MTVQA directory: {root}")

    splits: Dict[str, List[Dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        pair_path = root / _pairs_filename(split, pair_suffix)
        if not pair_path.exists() and pair_suffix:
            fallback = root / _pairs_filename(split, "")
            if fallback.exists():
                pair_path = fallback
        if not pair_path.exists():
            raise FileNotFoundError(
                f"Missing MTVQA pair split file for split={split}: {pair_path}"
            )
        splits[split] = read_jsonl(pair_path)

    return {
        "manifest": {
            "dataset_name": "mtvqa_qa_pairs",
            "pair_suffix": pair_suffix,
            "counts": {split: len(rows) for split, rows in splits.items()},
            "source": str(root),
        },
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
    }


def filter_mtvqa_samples(
    samples: List[Dict[str, Any]],
    languages: Optional[Iterable[str]] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    filtered = samples
    if languages is not None:
        language_set = {str(lang).strip().lower() for lang in languages if str(lang).strip()}
        filtered = [
            sample for sample in filtered if str(sample.get("language", "")).strip().lower() in language_set
        ]
    if max_samples is not None and max_samples >= 0:
        filtered = filtered[:max_samples]
    return filtered


def filter_mtvqa_pairs(
    pairs: List[Dict[str, Any]],
    categories: Optional[Iterable[str]] = None,
    max_pairs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    filtered = pairs
    if categories is not None:
        category_set = {str(cat).strip() for cat in categories if str(cat).strip()}
        filtered = [pair for pair in filtered if str(pair.get("category", "")).strip() in category_set]
    if max_pairs is not None and max_pairs >= 0:
        filtered = filtered[:max_pairs]
    return filtered


def build_mtvqa_single_view_samples(
    paired_records: List[Dict[str, Any]],
    view: str = "source",
) -> List[Dict[str, Any]]:
    if view not in {"source", "pivot"}:
        raise ValueError("view must be one of {'source', 'pivot'}")

    view_key = "source_view" if view == "source" else "pivot_view"
    flattened: List[Dict[str, Any]] = []
    for pair in paired_records:
        payload = pair.get(view_key) or {}
        flattened.append(
            {
                "sample_id": str(payload.get("sample_id", pair.get("pair_id", ""))),
                "document_id": str(payload.get("image_uid", pair.get("pair_id", ""))),
                "language": str(payload.get("language", "")).lower(),
                "image_path": str(payload.get("image_path", "")),
                "image_rel_path": str(payload.get("image_rel_path", "")),
                "question": str(payload.get("question", "")),
                "answer": str(payload.get("answer", "")),
                "split": str(pair.get("split", "")),
                "category": str(pair.get("category", "")),
                "pair_id": str(pair.get("pair_id", "")),
                "view": view,
            }
        )
    return flattened
