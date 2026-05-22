"""Single-sample evaluation helpers for external QA-style datasets."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm

from src.utils.evidence import parse_evidence_items
from src.utils.helpers import compute_exact_match, extract_answer_qwen


def strict_exact_match(prediction: str, ground_truth: str) -> bool:
    return prediction.strip().lower() == ground_truth.strip().lower()


def _write_progress_state(progress_path: Path, evaluated: int, total: int, checkpoint_tag: str) -> None:
    payload = {
        "checkpoint_tag": checkpoint_tag,
        "evaluated": int(evaluated),
        "total": int(total),
        "percent": (float(evaluated) / float(total) * 100.0) if total else 0.0,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    progress_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def summarize_single_predictions(predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    languages = sorted({str(pred["language"]) for pred in predictions})
    per_language = {
        language: _compute_group_metrics(
            [pred for pred in predictions if str(pred["language"]) == language]
        )
        for language in languages
    }
    overall = _compute_group_metrics(predictions)
    macro_strict = (
        sum(item["strict_accuracy"] for item in per_language.values()) / len(per_language)
        if per_language
        else 0.0
    )
    macro_normalized = (
        sum(item["normalized_accuracy"] for item in per_language.values()) / len(per_language)
        if per_language
        else 0.0
    )

    return {
        "strict_metrics": {
            "overall_accuracy": overall["strict_accuracy"],
            "macro_accuracy": macro_strict,
            "per_language_accuracy": {
                language: metrics["strict_accuracy"]
                for language, metrics in per_language.items()
            },
        },
        "normalized_metrics": {
            "overall_accuracy": overall["normalized_accuracy"],
            "macro_accuracy": macro_normalized,
            "per_language_accuracy": {
                language: metrics["normalized_accuracy"]
                for language, metrics in per_language.items()
            },
        },
        "counts": {
            "overall": overall,
            "per_language": per_language,
        },
    }


def build_judge_ready_predictions(predictions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "sample_id": pred["sample_id"],
            "document_id": pred["document_id"],
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
def evaluate_single_model(
    model,
    processor,
    samples: Sequence[Dict[str, Any]],
    device: str,
    build_chat_input_fn: Callable[[Any, Dict[str, Any], str, str], Dict[str, Any]],
    max_new_tokens: int = 128,
    split_name: str = "eval",
    checkpoint_tag: str = "base",
    save_predictions_dir: Optional[str] = None,
    split_file: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    split_metadata: Optional[Dict[str, Any]] = None,
    save_judge_ready: bool = False,
    show_progress: bool = True,
    progress_every: int = 0,
    resume: bool = False,
    checkpoint_every: int = 100,
) -> Dict[str, Any]:
    model.eval()
    predictions: List[Dict[str, Any]] = []

    if resume and not save_predictions_dir:
        raise ValueError("resume=True requires save_predictions_dir")

    predictions_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    checkpoint_jsonl_path: Optional[Path] = None
    progress_path: Optional[Path] = None
    judge_path: Optional[Path] = None

    if save_predictions_dir:
        out_dir = Path(save_predictions_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = out_dir / f"predictions_{checkpoint_tag}.json"
        summary_path = out_dir / f"summary_{checkpoint_tag}.json"
        checkpoint_jsonl_path = out_dir / f"predictions_{checkpoint_tag}.jsonl"
        progress_path = out_dir / f"progress_{checkpoint_tag}.json"
        if save_judge_ready:
            judge_path = out_dir / f"judge_ready_{checkpoint_tag}.json"

        if not resume:
            for stale in [predictions_path, summary_path, checkpoint_jsonl_path, progress_path, judge_path]:
                if stale is not None and stale.exists():
                    stale.unlink()
        else:
            loaded_from = None
            if checkpoint_jsonl_path.exists():
                loaded_from = checkpoint_jsonl_path
                with checkpoint_jsonl_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            predictions.append(json.loads(line))
                        except json.JSONDecodeError:
                            # Ignore a partial final line from abrupt shutdowns.
                            continue
            elif predictions_path.exists():
                loaded_from = predictions_path
                with predictions_path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                    if isinstance(loaded, list):
                        predictions = list(loaded)
                if checkpoint_jsonl_path is not None:
                    with checkpoint_jsonl_path.open("w", encoding="utf-8") as ckpt_handle:
                        for pred in predictions:
                            ckpt_handle.write(json.dumps(pred, ensure_ascii=False) + "\n")

            if loaded_from is not None:
                print(
                    f"[resume] loaded={len(predictions)} predictions from {loaded_from}",
                    flush=True,
                )

    eos_token_id: Optional[int] = None
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    total = len(samples)
    completed_ids = {str(pred.get("sample_id", "")) for pred in predictions}
    remaining_samples = (
        [sample for sample in samples if str(sample["sample_id"]) not in completed_ids]
        if resume
        else list(samples)
    )
    resumed_predictions = len(predictions)

    if show_progress:
        iterable = tqdm(
            remaining_samples,
            desc=f"Evaluating {checkpoint_tag}",
            total=total,
            initial=resumed_predictions,
        )
    else:
        iterable = remaining_samples

    append_handle = (
        checkpoint_jsonl_path.open("a", encoding="utf-8") if checkpoint_jsonl_path is not None else None
    )

    if progress_path is not None:
        _write_progress_state(progress_path, resumed_predictions, total, checkpoint_tag)

    try:
        for idx, sample in enumerate(iterable, start=1):
            question = str(sample["question"])
            ground_truth = str(sample["answer"])

            inputs = build_chat_input_fn(processor, sample, question, device)
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            }
            if eos_token_id is not None:
                generate_kwargs["pad_token_id"] = eos_token_id
            generated_ids = model.generate(**inputs, **generate_kwargs)
            gen_ids = generated_ids[0, inputs["input_ids"].shape[1]:]
            raw_response = processor.decode(gen_ids, skip_special_tokens=True).strip()
            evidence_payload = parse_evidence_items(raw_response)
            extracted_prediction = evidence_payload.get("answer_text") or extract_answer_qwen(raw_response)

            prediction_row = {
                "sample_id": str(sample["sample_id"]),
                "document_id": str(sample["document_id"]),
                "language": str(sample["language"]),
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
            predictions.append(prediction_row)

            if append_handle is not None:
                append_handle.write(json.dumps(prediction_row, ensure_ascii=False) + "\n")
                append_handle.flush()

            evaluated_count = resumed_predictions + idx
            if progress_path is not None and (
                checkpoint_every <= 0
                or evaluated_count == total
                or evaluated_count % checkpoint_every == 0
            ):
                _write_progress_state(progress_path, evaluated_count, total, checkpoint_tag)

            if not show_progress and progress_every > 0 and evaluated_count % progress_every == 0:
                print(
                    f"[progress] evaluated={evaluated_count}/{total} checkpoint={checkpoint_tag}",
                    flush=True,
                )
    finally:
        if append_handle is not None:
            append_handle.close()

    metrics = summarize_single_predictions(predictions)
    summary = {
        "split_name": split_name,
        "checkpoint_tag": checkpoint_tag,
        "split_file": split_file,
        "resume_enabled": resume,
        "resumed_predictions": resumed_predictions,
        "total_samples": total,
        "evaluated_samples": len(predictions),
        "strict_metrics": metrics["strict_metrics"],
        "normalized_metrics": metrics["normalized_metrics"],
        "counts": metrics["counts"],
        "generation_config": generation_config or {},
    }
    if split_metadata is not None:
        summary["split_metadata"] = split_metadata

    if save_predictions_dir and predictions_path is not None and summary_path is not None:
        summary["predictions_path"] = str(predictions_path)
        summary["summary_path"] = str(summary_path)
        if checkpoint_jsonl_path is not None:
            summary["checkpoint_jsonl_path"] = str(checkpoint_jsonl_path)
        if progress_path is not None:
            summary["progress_path"] = str(progress_path)

        with predictions_path.open("w", encoding="utf-8") as handle:
            json.dump(predictions, handle, indent=2, ensure_ascii=False)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

        if save_judge_ready and judge_path is not None:
            with judge_path.open("w", encoding="utf-8") as handle:
                json.dump(build_judge_ready_predictions(predictions), handle, indent=2, ensure_ascii=False)
            summary["judge_ready_path"] = str(judge_path)

        if progress_path is not None:
            _write_progress_state(progress_path, len(predictions), total, checkpoint_tag)

    summary["predictions"] = predictions
    return summary
