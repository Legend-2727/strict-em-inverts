"""OOD split and artifact helpers for paired MMCricBench evaluation."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .paired import _pair_identifier

_ORDINAL_WORDS = (
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
)


def normalize_question_template(question: str) -> str:
    """
    Heuristically normalize a question into a coarse template.

    The goal is to reduce leakage from repeated near-identical question strings,
    not to build a complete semantic parser.
    """
    original = " ".join(str(question).strip().split())
    working = original

    working = re.sub(r"'[^']+'|\"[^\"]+\"", "<ENT>", working)
    working = re.sub(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})'s\b",
        "<ENT>'s",
        working,
    )
    working = re.sub(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b(?=\s+than\b)",
        "<ENT>",
        working,
    )
    working = re.sub(
        r"(?<=than\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b",
        "<ENT>",
        working,
    )

    working = working.lower()
    working = re.sub(r"\bteam\s+\d+\b", "<TEAM>", working)
    working = re.sub(r"\b(?:%s)\b" % "|".join(_ORDINAL_WORDS), "<ORD>", working)
    working = re.sub(r"\b\d+(?:st|nd|rd|th)\b", "<ORD>", working)
    working = re.sub(r"\b\d+(?:\.\d+)?\b", "<NUM>", working)
    working = re.sub(r"\b\d+\s*-\s*fer\b", "<NUM>-fer", working)
    working = re.sub(r"\s+", " ", working).strip(" ?")
    return working


def _pair_to_record(pair: Dict[str, Any], template: str) -> Dict[str, Any]:
    en_sample = pair["en_sample"]
    hi_sample = pair["hi_sample"]
    return {
        "pair_id": _pair_identifier(pair),
        "english_id": str(en_sample["id"]),
        "hindi_id": str(hi_sample["id"]),
        "english_question": str(en_sample["question"]),
        "hindi_question": str(hi_sample["question"]),
        "template": template,
    }


def build_template_ood_split_payload(
    pairs: Sequence[Dict[str, Any]],
    split_name: str,
    seed: int,
    holdout_ratio: float = 0.2,
) -> Dict[str, Any]:
    """Group pairs by normalized template and hold out full template groups."""
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be between 0 and 1")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    template_examples: Dict[str, List[str]] = defaultdict(list)
    for pair in pairs:
        en_question = str(pair["en_sample"]["question"])
        hi_question = str(pair["hi_sample"]["question"])
        template = normalize_question_template(en_question or hi_question)
        grouped[template].append(pair)
        if en_question not in template_examples[template]:
            template_examples[template].append(en_question)
        if hi_question not in template_examples[template]:
            template_examples[template].append(hi_question)

    if len(grouped) < 2:
        raise ValueError("template-OOD split requires at least two template groups")

    templates = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(templates)

    target_eval_pairs = max(1, round(len(pairs) * holdout_ratio))
    eval_templates: List[str] = []
    eval_pairs: List[Dict[str, Any]] = []

    for template in templates:
        eval_templates.append(template)
        eval_pairs.extend(grouped[template])
        if len(eval_pairs) >= target_eval_pairs and len(eval_templates) < len(templates):
            break

    eval_pair_ids = sorted(_pair_identifier(pair) for pair in eval_pairs)
    eval_pair_id_set = set(eval_pair_ids)
    train_pairs = [pair for pair in pairs if _pair_identifier(pair) not in eval_pair_id_set]

    if not train_pairs:
        raise ValueError("template-OOD split left no training pairs; reduce holdout_ratio")

    held_out_templates = [
        {
            "template": template,
            "pair_count": len(grouped[template]),
            "example_questions": template_examples[template][:4],
        }
        for template in sorted(eval_templates)
    ]

    return {
        "split_name": split_name,
        "regime": "s1_template_ood",
        "source_split": "test_single",
        "selection_seed": seed,
        "holdout_ratio": holdout_ratio,
        "template_count_total": len(grouped),
        "held_out_template_count": len(eval_templates),
        "train_pair_count": len(train_pairs),
        "eval_pair_count": len(eval_pair_ids),
        "pair_ids": eval_pair_ids,
        "train_pair_ids": sorted(_pair_identifier(pair) for pair in train_pairs),
        "held_out_templates": held_out_templates,
        "pair_records": [
            _pair_to_record(pair, normalize_question_template(str(pair["en_sample"]["question"])))
            for pair in sorted(eval_pairs, key=_pair_identifier)
        ],
        "normalization_notes": [
            "lowercase and whitespace normalization",
            "replace numbers with <NUM>",
            "replace ordinal words and ordinal numerals with <ORD>",
            "replace Team <num> with <TEAM>",
            "replace repeated proper-name spans with <ENT> when practical",
        ],
    }


def save_split_payload(payload: Dict[str, Any], output_path: str) -> Path:
    """Write a JSON split artifact to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_summary_payload(
    results: Dict[str, Any],
    split_metadata: Dict[str, Any],
    checkpoint_tag: str,
    predictions_path: str,
    summary_path: str,
) -> Dict[str, Any]:
    """Build a paper-facing summary JSON payload from paired eval results."""
    payload = {
        "split_name": results["split_name"],
        "checkpoint_tag": checkpoint_tag,
        "strict_metrics": results["strict_metrics"],
        "normalized_metrics": results["normalized_metrics"],
        "counts": results["counts"],
        "generation_config": results.get("generation_config", {}),
        "split_metadata": split_metadata,
        "predictions_path": predictions_path,
        "summary_path": summary_path,
    }
    if results.get("split_file") is not None:
        payload["split_file"] = results["split_file"]
    return payload


def write_eval_artifacts(
    predictions: List[Dict[str, Any]],
    summary_payload: Dict[str, Any],
    output_dir: str,
    checkpoint_tag: str,
) -> Dict[str, str]:
    """Write explicit predictions/summary artifact names for a regime run."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / f"predictions_{checkpoint_tag}.json"
    summary_path = out_dir / f"summary_{checkpoint_tag}.json"

    with predictions_path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2, ensure_ascii=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

    return {
        "predictions_path": str(predictions_path),
        "summary_path": str(summary_path),
    }
