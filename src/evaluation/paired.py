"""Shared paired EN/HI evaluation and fixed split helpers."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm

from src.data import language_from_id
from src.utils.evidence import parse_evidence_items
from src.utils.helpers import extract_answer_qwen, compute_exact_match

logger = logging.getLogger(__name__)


def create_paired_dataset(
    dataset_split,
    max_pairs: Optional[int] = None,
    pair_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Create English/Hindi paired samples from an MMCricBench split."""
    en_map: Dict[str, int] = {}
    hi_map: Dict[str, int] = {}

    for idx, sid in enumerate(dataset_split["id"]):
        sid = str(sid)
        lang = language_from_id(sid)
        core_id = sid.replace(f"{lang}-", "", 1)
        if lang == "english":
            en_map[core_id] = idx
        elif lang == "hindi":
            hi_map[core_id] = idx

    if pair_ids is not None:
        ordered_core_ids = [str(pid) for pid in pair_ids]
    else:
        ordered_core_ids = sorted(en_map)

    pairs: List[Dict[str, Any]] = []
    for core_id in ordered_core_ids:
        if core_id not in hi_map:
            continue
        if core_id not in en_map:
            continue
        pairs.append(
            {
                "pair_id": core_id,
                "core_id": core_id,
                "en_sample": dataset_split[en_map[core_id]],
                "hi_sample": dataset_split[hi_map[core_id]],
            }
        )
        if max_pairs and len(pairs) >= max_pairs:
            break
    return pairs


def strict_exact_match(prediction: str, ground_truth: str) -> bool:
    """Match the legacy script behavior: lowercase + strip only."""
    return prediction.strip().lower() == ground_truth.strip().lower()


def _pair_identifier(pair: Dict[str, Any]) -> str:
    return str(pair.get("pair_id") or pair.get("core_id"))


def save_fixed_eval_split(
    pairs: Sequence[Dict[str, Any]],
    split_path: str,
    split_name: str,
    eval_size: int,
    seed: int,
) -> Path:
    """Create and save a deterministic fixed paired eval split."""
    if eval_size <= 0:
        raise ValueError("fixed eval size must be positive")
    if eval_size > len(pairs):
        raise ValueError(
            f"requested fixed eval size {eval_size} exceeds available pairs {len(pairs)}"
        )

    rng = random.Random(seed)
    selected_pairs = rng.sample(list(pairs), eval_size)
    selected_pairs = sorted(selected_pairs, key=_pair_identifier)

    payload = {
        "split_name": split_name,
        "selection_seed": seed,
        "num_pairs": len(selected_pairs),
        "pair_ids": [_pair_identifier(pair) for pair in selected_pairs],
        "sample_ids": [
            {
                "pair_id": _pair_identifier(pair),
                "english_id": str(pair["en_sample"]["id"]),
                "hindi_id": str(pair["hi_sample"]["id"]),
            }
            for pair in selected_pairs
        ],
    }

    output_path = Path(split_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Saved fixed paired eval split to %s", output_path)
    return output_path


def load_fixed_eval_split(split_path: str) -> Dict[str, Any]:
    with open(split_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "pair_ids" not in payload or not isinstance(payload["pair_ids"], list):
        raise ValueError(f"invalid fixed eval split file: {split_path}")
    return payload


def select_pairs_from_fixed_split(
    pairs: Sequence[Dict[str, Any]],
    split_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Select paired samples using a saved split definition."""
    pair_map = {_pair_identifier(pair): pair for pair in pairs}
    selected_pairs: List[Dict[str, Any]] = []

    missing: List[str] = []
    for pair_id in split_payload["pair_ids"]:
        pair = pair_map.get(str(pair_id))
        if pair is None:
            missing.append(str(pair_id))
        else:
            selected_pairs.append(pair)

    if missing:
        raise ValueError(
            f"fixed eval split references {len(missing)} missing pair ids: {missing[:5]}"
        )
    return selected_pairs


def exclude_pairs(
    pairs: Sequence[Dict[str, Any]],
    excluded_pairs: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    excluded_ids = {_pair_identifier(pair) for pair in excluded_pairs}
    return [pair for pair in pairs if _pair_identifier(pair) not in excluded_ids]


def prepare_fixed_paired_eval_split(
    all_pairs: Sequence[Dict[str, Any]],
    split_path: Optional[str],
    create_split: bool,
    split_name: str,
    eval_size: int,
    seed: int,
) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Create or load a fixed split and return selected eval pairs."""
    if create_split and not split_path:
        raise ValueError("--create_fixed_eval_split requires --fixed_eval_split")

    resolved_split_path: Optional[str] = split_path
    if create_split:
        resolved_split_path = str(
            save_fixed_eval_split(all_pairs, split_path, split_name, eval_size, seed)
        )

    if not resolved_split_path:
        return None, None

    split_payload = load_fixed_eval_split(resolved_split_path)
    selected_pairs = select_pairs_from_fixed_split(all_pairs, split_payload)
    return selected_pairs, resolved_split_path


def _compute_group_metrics(predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(predictions)
    strict_correct = sum(1 for pred in predictions if pred["strict_em"])
    normalized_correct = sum(1 for pred in predictions if pred["normalized_em"])
    return {
        "total": total,
        "strict_correct": strict_correct,
        "strict_accuracy": (strict_correct / total * 100.0) if total else 0.0,
        "normalized_correct": normalized_correct,
        "normalized_accuracy": (normalized_correct / total * 100.0) if total else 0.0,
    }


def summarize_paired_predictions(predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    english = [pred for pred in predictions if pred["language"] == "english"]
    hindi = [pred for pred in predictions if pred["language"] == "hindi"]
    overall = _compute_group_metrics(predictions)
    english_metrics = _compute_group_metrics(english)
    hindi_metrics = _compute_group_metrics(hindi)

    return {
        "strict_metrics": {
            "overall_accuracy": overall["strict_accuracy"],
            "english_accuracy": english_metrics["strict_accuracy"],
            "hindi_accuracy": hindi_metrics["strict_accuracy"],
            "gap_pp": english_metrics["strict_accuracy"] - hindi_metrics["strict_accuracy"],
        },
        "normalized_metrics": {
            "overall_accuracy": overall["normalized_accuracy"],
            "english_accuracy": english_metrics["normalized_accuracy"],
            "hindi_accuracy": hindi_metrics["normalized_accuracy"],
            "gap_pp": english_metrics["normalized_accuracy"] - hindi_metrics["normalized_accuracy"],
        },
        "counts": {
            "overall": overall,
            "english": english_metrics,
            "hindi": hindi_metrics,
        },
    }


def build_judge_ready_predictions(predictions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Save a lightweight judge-ready view from the same prediction records."""
    return [
        {
            "pair_id": pred.get("pair_id"),
            "sample_id": pred["sample_id"],
            "id": pred["id"],
            "language": pred["language"],
            "question": pred["question"],
            "ground_truth": pred["ground_truth"],
            "model_prediction": pred["raw_response"],
            "extracted_prediction": pred["extracted_prediction"],
            "strict_em": pred["strict_em"],
            "normalized_em": pred["normalized_em"],
            "split_name": pred["split_name"],
            "checkpoint_tag": pred["checkpoint_tag"],
        }
        for pred in predictions
    ]


@torch.no_grad()
def evaluate_paired_model(
    model,
    processor,
    pairs: Sequence[Dict[str, Any]],
    device: str,
    build_chat_input_fn: Callable[[Any, Dict[str, Any], str, str], Dict[str, Any]],
    max_new_tokens: int = 128,
    split_name: str = "eval",
    checkpoint_tag: str = "base",
    save_predictions_dir: Optional[str] = None,
    split_file: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    save_judge_ready: bool = False,
    split_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run paired evaluation with raw/extracted predictions and both EM metrics."""
    model.eval()
    predictions: List[Dict[str, Any]] = []

    for pair in tqdm(pairs, desc=f"Evaluating {checkpoint_tag}"):
        pair_id = _pair_identifier(pair)
        for language, sample_key in [("english", "en_sample"), ("hindi", "hi_sample")]:
            sample = pair[sample_key]
            question = str(sample["question"])
            ground_truth = str(sample["answer"])

            inputs = build_chat_input_fn(processor, sample, question, device)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            gen_ids = generated_ids[0, inputs["input_ids"].shape[1]:]
            raw_response = processor.decode(gen_ids, skip_special_tokens=True).strip()
            evidence_payload = parse_evidence_items(raw_response)
            extracted_prediction = evidence_payload.get("answer_text") or extract_answer_qwen(raw_response)

            predictions.append(
                {
                    "pair_id": pair_id,
                    "sample_id": str(sample["id"]),
                    "id": str(sample["id"]),
                    "language": language,
                    "question": question,
                    "ground_truth": ground_truth,
                    "raw_response": raw_response,
                    "extracted_prediction": extracted_prediction,
                    "strict_em": strict_exact_match(extracted_prediction, ground_truth),
                    "normalized_em": compute_exact_match(extracted_prediction, ground_truth),
                    "split_name": split_name,
                    "checkpoint_tag": checkpoint_tag,
                    "evidence_text": evidence_payload.get("evidence_text"),
                    "parsed_evidence": evidence_payload.get("parsed_evidence"),
                    "evidence_parse_ok": evidence_payload.get("evidence_parse_ok", False),
                }
            )

    metrics = summarize_paired_predictions(predictions)
    summary = {
        "split_name": split_name,
        "checkpoint_tag": checkpoint_tag,
        "split_file": split_file,
        "strict_metrics": metrics["strict_metrics"],
        "normalized_metrics": metrics["normalized_metrics"],
        "counts": metrics["counts"],
        "generation_config": generation_config or {},
    }
    if split_metadata is not None:
        summary["split_metadata"] = split_metadata

    if save_predictions_dir:
        out_dir = Path(save_predictions_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{split_name}_{checkpoint_tag}"
        predictions_path = out_dir / f"{stem}_predictions.json"
        summary_path = out_dir / f"{stem}_summary.json"
        summary["predictions_path"] = str(predictions_path)
        summary["summary_path"] = str(summary_path)
        judge_path = None
        if save_judge_ready:
            judge_path = out_dir / f"{stem}_judge_ready.json"
            summary["judge_ready_path"] = str(judge_path)

        with predictions_path.open("w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        if judge_path is not None:
            with judge_path.open("w", encoding="utf-8") as f:
                json.dump(build_judge_ready_predictions(predictions), f, indent=2, ensure_ascii=False)

    summary["predictions"] = predictions
    return summary
