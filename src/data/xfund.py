"""XFUND QA-conversion loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

XFUND_LANGUAGES = ("de", "es", "fr", "it", "ja", "pt", "zh")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_xfund_qa_dataset(dataset_dir: str) -> Dict[str, Any]:
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing XFUND QA manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits: Dict[str, List[Dict[str, Any]]] = {}
    for split_name in ("train", "val", "test"):
        split_path = root / f"{split_name}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing XFUND QA split file: {split_path}")
        splits[split_name] = read_jsonl(split_path)

    return {
        "manifest": manifest,
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
    }


def load_xfund_qa_paired_dataset(dataset_dir: str) -> Dict[str, Any]:
    """Load paired XFUND QA dataset with source + English-pivot question views."""
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing paired XFUND manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits: Dict[str, List[Dict[str, Any]]] = {}
    for split_name in ("train", "val", "test"):
        split_path = root / f"{split_name}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing paired XFUND split file: {split_path}")
        splits[split_name] = read_jsonl(split_path)

    return {
        "manifest": manifest,
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
    }


def filter_xfund_samples(
    samples: List[Dict[str, Any]],
    languages: Optional[Iterable[str]] = None,
    max_samples: Optional[int] = None,
    stratified: bool = False,
) -> List[Dict[str, Any]]:
    if languages is not None:
        language_set = {str(lang).strip() for lang in languages if str(lang).strip()}
        samples = [sample for sample in samples if str(sample.get("language")) in language_set]
    if max_samples is not None and max_samples >= 0:
        if stratified:
            return _balance_by_language(samples, max_samples, language_key="language")
        return samples[:max_samples]
    return samples


def _balance_by_language(
    records: List[Dict[str, Any]],
    max_total: int,
    language_key: str = "language",
) -> List[Dict[str, Any]]:
    """Round-robin select up to max_total records, evenly across observed languages.

    Preserves original order within each language. If a language has fewer
    records than its share, the surplus is distributed to remaining languages.
    """
    if max_total <= 0 or not records:
        return []
    by_lang: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for record in records:
        lang = str(record.get(language_key, ""))
        if lang not in by_lang:
            by_lang[lang] = []
            order.append(lang)
        by_lang[lang].append(record)
    selected: List[Dict[str, Any]] = []
    cursors = {lang: 0 for lang in order}
    while len(selected) < max_total:
        progressed = False
        for lang in order:
            if len(selected) >= max_total:
                break
            cursor = cursors[lang]
            bucket = by_lang[lang]
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                cursors[lang] = cursor + 1
                progressed = True
        if not progressed:
            break
    return selected


def filter_xfund_pairs(
    pairs: List[Dict[str, Any]],
    languages: Optional[Iterable[str]] = None,
    max_pairs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Filter paired XFUND records by source language and max pair count."""
    if languages is not None:
        language_set = {str(lang).strip() for lang in languages if str(lang).strip()}
        pairs = [pair for pair in pairs if str(pair.get("language")) in language_set]
    if max_pairs is not None and max_pairs >= 0:
        return pairs[:max_pairs]
    return pairs


def build_xfund_single_view_samples(
    paired_records: List[Dict[str, Any]],
    view: str = "source",
) -> List[Dict[str, Any]]:
    """
    Convert paired records into single-view samples for fair single-split evaluation.

    Args:
        paired_records: Paired XFUND records with source and pivot views.
        view: "source" or "pivot" question view.
    """
    if view not in {"source", "pivot"}:
        raise ValueError("view must be one of {'source', 'pivot'}")

    view_key = "source_view" if view == "source" else "pivot_en_view"
    flattened: List[Dict[str, Any]] = []
    for record in paired_records:
        view_payload = record.get(view_key) or {}
        question_text = str(view_payload.get("question") or "").strip()
        if not question_text:
            raise ValueError(
                f"paired record missing {view_key}.question for sample_id={record.get('sample_id')}"
            )

        flattened.append(
            {
                "sample_id": str(record["sample_id"]),
                "document_id": str(record["document_id"]),
                "language": str(record["language"]),
                "official_split": str(record.get("official_split", "")),
                "image_path": str(record["image_path"]),
                "image_rel_path": str(record.get("image_rel_path", "")),
                "question": question_text,
                "answer": str(record["answer"]),
                "evidence_items": record.get("evidence_items") or [],
                "pair_id": str(record.get("pair_id") or record["sample_id"]),
                "view": view,
            }
        )
    return flattened
