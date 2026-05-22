"""
Evaluation utilities for MMCricBench experiments.

Provides shared checkpoint management, metric computation, judge evaluation,
summary printing, and post-hoc analysis helpers used across experiment scripts.
"""

import os
import re
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Callable

import torch
from tqdm import tqdm

from src.utils.helpers import extract_answer_qwen, compute_exact_match, create_judge_prompt, create_judge_prompt_components
from src.data import language_from_id, split_from_id
from .paired import (
    create_paired_dataset,
    evaluate_paired_model,
    prepare_fixed_paired_eval_split,
    exclude_pairs,
    load_fixed_eval_split,
    select_pairs_from_fixed_split,
    save_fixed_eval_split,
)
from .ood import (
    normalize_question_template,
    build_template_ood_split_payload,
    save_split_payload,
    load_json,
    build_summary_payload,
    write_eval_artifacts,
)
from .single import evaluate_single_model, summarize_single_predictions

logger = logging.getLogger(__name__)


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def load_checkpoint(checkpoint_file: str) -> Optional[Dict]:
    """
    Load a previously saved checkpoint, returning *None* when the file does
    not exist or is unreadable.
    """
    path = Path(checkpoint_file)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Could not load checkpoint {checkpoint_file}: {exc}")
        return None


def save_checkpoint(checkpoint_file: str, results: Dict, category_results: Dict):
    """
    Save an inference checkpoint (single-response mode).

    Works with the results dict produced by :func:`run_inference_loop`.
    """
    checkpoint_results = results.copy()
    n_preds = len(results["predictions"])
    checkpoint_results["accuracy"] = (
        results["correct"] / n_preds if n_preds > 0 else 0
    )
    checkpoint_results["category_accuracy"] = {}
    for cat, cat_res in category_results.items():
        checkpoint_results["category_accuracy"][cat] = {
            "accuracy": cat_res["correct"] / cat_res["total"] if cat_res["total"] > 0 else 0,
            "correct": cat_res["correct"],
            "total": cat_res["total"],
        }
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_results, f, indent=2)
    logger.info(f"Checkpoint saved: {n_preds} samples processed")


def save_steered_checkpoint(
    checkpoint_file: str, results: Dict, category_results: Dict
):
    """
    Save an evaluation checkpoint (steered + baseline dual-response mode).

    Works with the results dict produced by :func:`run_steered_evaluation`.
    """
    n_processed = len(results["predictions"])
    checkpoint_results = results.copy()
    checkpoint_results["accuracy_steered"] = (
        results["correct_steered"] / n_processed if n_processed > 0 else 0
    )
    checkpoint_results["accuracy_baseline"] = (
        results["correct_baseline"] / n_processed if n_processed > 0 else 0
    )
    checkpoint_results["category_accuracy"] = {}
    for mode in ["steered", "baseline"]:
        checkpoint_results["category_accuracy"][mode] = {}
        for cat, cat_res in category_results.get(mode, {}).items():
            checkpoint_results["category_accuracy"][mode][cat] = {
                "accuracy": cat_res["correct"] / cat_res["total"] if cat_res["total"] > 0 else 0,
                "correct": cat_res["correct"],
                "total": cat_res["total"],
            }
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_results, f, indent=2)
    logger.info(f"Checkpoint saved: {n_processed} samples processed")


def save_judge_checkpoint(checkpoint_file: str, judge_results: Dict):
    """Save judge evaluation checkpoint."""
    with open(checkpoint_file, "w") as f:
        json.dump(judge_results, f, indent=2)
    logger.info(
        f"Judge checkpoint saved: {len(judge_results['evaluations'])} samples evaluated"
    )


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_metrics(results: Dict, category_results: Dict):
    """Compute accuracy metrics (single-response mode) in-place."""
    results["accuracy"] = (
        results["correct"] / results["total_samples"]
        if results["total_samples"] > 0
        else 0
    )
    results["category_accuracy"] = {}
    for cat, cat_res in category_results.items():
        results["category_accuracy"][cat] = {
            "accuracy": cat_res["correct"] / cat_res["total"] if cat_res["total"] > 0 else 0,
            "correct": cat_res["correct"],
            "total": cat_res["total"],
        }


def compute_steered_metrics(results: Dict, category_results: Dict):
    """Compute accuracy metrics (steered + baseline mode) in-place."""
    n = results["total_samples"]
    results["accuracy_steered"] = results["correct_steered"] / n if n > 0 else 0
    results["accuracy_baseline"] = results["correct_baseline"] / \
        n if n > 0 else 0
    results["category_accuracy"] = {}
    for mode in ["steered", "baseline"]:
        results["category_accuracy"][mode] = {}
        for cat, cat_res in category_results.get(mode, {}).items():
            results["category_accuracy"][mode][cat] = {
                "accuracy": cat_res["correct"] / cat_res["total"] if cat_res["total"] > 0 else 0,
                "correct": cat_res["correct"],
                "total": cat_res["total"],
            }


# =============================================================================
# SAMPLE FROM DATASET
# =============================================================================

def get_filtered_samples(
    dataset_split,
    max_samples: Optional[int] = None,
):
    """
    Filters a dataset split, returning up to `max_samples` per category and language.

    Args:
        dataset_split: HuggingFace dataset split.
        max_samples: Max samples *per category per language* (None -> all).

    Returns:
        A HuggingFace Dataset containing the filtered samples.
    """
    # Group sample indices by category and language
    category_language_samples: Dict = defaultdict(
        lambda: {"english": [], "hindi": []}
    )
    
    ids = dataset_split["id"]
    categories = dataset_split["category"]
    
    for idx, (sample_id, category) in enumerate(zip(ids, categories)):
        language = language_from_id(sample_id)
        category_language_samples[category][language].append(idx)

    # Select samples based on the threshold
    samples_to_evaluate: List[int] = []
    
    if max_samples is not None:
        logger.info(f"Selecting up to {max_samples} samples per category per language")
        for category, lang_dict in category_language_samples.items():
            for lang in ["english", "hindi"]:
                indices = lang_dict[lang]
                selected = indices[:max_samples]
                samples_to_evaluate.extend(selected)
                logger.info(f"  {category} ({lang}): {len(selected)} samples selected")
    else:
        logger.info("No max_samples specified; selecting all samples.")
        samples_to_evaluate = list(range(len(dataset_split)))

    logger.info(f"Total samples after filtering: {len(samples_to_evaluate)}")

    return dataset_split.select(samples_to_evaluate)


def get_selected_samples_by_id(
    dataset_split,
    selected_ids: List[str],
):
    """
    Select samples from a dataset split based on a list of sample IDs.

    Args:
        dataset_split: HuggingFace dataset split.
        selected_ids: List of sample IDs to select.
    Returns:
        A HuggingFace Dataset containing the selected samples.
    """
    id_set = set(selected_ids)
    all_ids = dataset_split["id"]
    indices = [idx for idx, sample_id in enumerate(all_ids) if sample_id in id_set]

    missing = id_set - set(all_ids[i] for i in indices)
    if missing:
        logger.warning(f"IDs not found in dataset: {missing}")

    logger.info(f"Selected {len(indices)} samples by ID (requested {len(selected_ids)})")
    return dataset_split.select(indices)


# =============================================================================
# JUDGE RESPONSE PARSING
# =============================================================================

def parse_judge_response(response: str) -> Dict:
    """
    Parse JSON response from judge model.

    Args:
        response: Raw response from judge

    Returns:
        Dictionary with 'reasoning' and 'is_correct' keys
    """
    try:
        json_match = re.search(
            r'\{[^}]*"reasoning"[^}]*"is_correct"[^}]*\}', response, re.DOTALL
        )
        if json_match:
            return json.loads(json_match.group())
        return json.loads(response)
    except json.JSONDecodeError:
        logger.warning(
            f"Failed to parse judge response as JSON: {response[:100]}..."
        )
        response_lower = response.lower()
        is_correct = (
            "correct" in response_lower
            and "not correct" not in response_lower
            and "incorrect" not in response_lower
        )
        return {"reasoning": response, "is_correct": is_correct}


# =============================================================================
# INFERENCE LOOP (single-response, no steering)
# =============================================================================

def run_inference_loop(
    model,
    dataset_split,
    split_name: str,
    prompt_template: str,
    max_samples: Optional[int] = None,
    output_dir: str = "results",
    checkpoint_interval: int = 50,
    checkpoint_file: Optional[str] = None,
    max_new_tokens: int = 512,
) -> Dict:
    """
    Run VLM inference on a dataset split (single response per sample).

    This is the shared inference loop used by the pipeline script.  It generates
    one response per sample, computes exact-match accuracy, and supports
    checkpoint/resume.

    Args:
        model: A model wrapper with a ``.chat(user_message, images, max_new_tokens)`` method.
        dataset_split: HuggingFace dataset split.
        split_name: Name of the split (e.g. ``"test_single"``).
        prompt_template: Prompt template with ``{question}`` placeholder.
        max_samples: Maximum number of samples to evaluate (``None`` → all).
        output_dir: Directory for saving checkpoint files.
        checkpoint_interval: Save checkpoint every *K* samples.
        checkpoint_file: Explicit checkpoint path (auto-generated if ``None``).
        max_new_tokens: Maximum tokens to generate per sample.

    Returns:
        Dictionary with evaluation metrics and predictions.
    """
    logger.info(f"Running inference on {split_name}...")

    n_samples = len(dataset_split)
    if max_samples is not None:
        n_samples = min(n_samples, max_samples)

    if checkpoint_file is None:
        checkpoint_file = os.path.join(
            output_dir, f"checkpoint_{split_name}.json")

    # Try to load existing checkpoint for resume
    processed_ids: set = set()
    results: Dict = {
        "split": split_name,
        "total_samples": n_samples,
        "correct": 0,
        "predictions": [],
    }
    category_results: Dict = {}

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
            results = checkpoint_data
            for pred in results["predictions"]:
                processed_ids.add(pred["id"])
                cat = pred["category"]
                if cat not in category_results:
                    category_results[cat] = {"correct": 0, "total": 0}
                category_results[cat]["total"] += 1
                if pred.get("is_correct", False):
                    category_results[cat]["correct"] += 1
            logger.info(
                f"Resuming from checkpoint: {len(processed_ids)} samples already processed"
            )
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")

    samples_since_checkpoint = 0

    for idx in tqdm(range(n_samples), desc=f"Processing {split_name}"):
        sample = dataset_split[idx]
        sample_id = sample["id"]

        if sample_id in processed_ids:
            continue

        question = sample["question"]
        ground_truth = sample["answer"]
        category = sample["category"]
        images = sample["images"]

        prompt = prompt_template.format(question=question)

        try:
            response = model.chat(
                user_message=prompt,
                images=images,
                max_new_tokens=max_new_tokens,
            )
            prediction = extract_answer_qwen(response)
            is_correct = compute_exact_match(prediction, ground_truth)

            if is_correct:
                results["correct"] += 1

            if category not in category_results:
                category_results[category] = {"correct": 0, "total": 0}
            category_results[category]["total"] += 1
            if is_correct:
                category_results[category]["correct"] += 1

            results["predictions"].append(
                {
                    "id": sample_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "full_response": response,
                    "is_correct": is_correct,
                    "category": category,
                }
            )

        except Exception as e:
            logger.error(f"Error processing sample {sample_id}: {e}")
            results["predictions"].append(
                {
                    "id": sample_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": None,
                    "full_response": str(e),
                    "is_correct": False,
                    "category": category,
                    "error": True,
                }
            )

        finally:
            torch.cuda.empty_cache()
            samples_since_checkpoint += 1
            if samples_since_checkpoint >= checkpoint_interval:
                save_checkpoint(checkpoint_file, results, category_results)
                samples_since_checkpoint = 0

    compute_metrics(results, category_results)
    save_checkpoint(checkpoint_file, results, category_results)
    return results


# =============================================================================
# STEERED EVALUATION LOOP
# =============================================================================

def run_steered_evaluation(
    model,
    processor,
    dataset_split,
    split_name: str,
    generate_fn: Callable,
    steering_vectors: Dict,
    key_scale: float = 0.0,
    value_scale: float = 1.0,
    steer_layers: Optional[List[int]] = None,
    max_samples: Optional[int] = None,
    output_dir: str = "results",
    checkpoint_interval: int = 50,
    device: str = "cuda",
    prompt_template: str = "Answer precisely in 1-2 words, answer in digits when required.\n\nQuestion: {question}",
    max_new_tokens: int = 512,
) -> Dict:
    """
    Run evaluation with KV cache steering (baseline + steered per sample).

    Args:
        model: The raw VLM model (not a wrapper).
        processor: The model processor / tokenizer.
        dataset_split: HuggingFace dataset split.
        split_name: Name of the split.
        generate_fn: Callable with signature
            ``(model, processor, messages, key_steer_vectors, value_steer_vectors,
              key_scale, value_scale, steer_layers, max_new_tokens, device) -> str``.
        steering_vectors: Dict with ``"keys"`` and ``"values"`` entries.
        key_scale, value_scale: Steering strength.
        steer_layers: Which layers to steer (``None`` → all).
        max_samples: Max samples *per category per language* (``None`` → all).
        output_dir: Directory for checkpoint files.
        checkpoint_interval: Checkpoint every *K* samples.
        device: Target device.
        prompt_template: Prompt template with ``{question}`` placeholder.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Dictionary with evaluation results.
    """
    logger.info(f"Running steered evaluation on {split_name}...")
    logger.info(f"  Key scale: {key_scale}, Value scale: {value_scale}")

    key_steer_vectors = steering_vectors.get("keys")
    value_steer_vectors = steering_vectors.get("values")

    # Organise samples by category and language
    category_language_samples: Dict = defaultdict(
        lambda: {"english": [], "hindi": []})
    ids = dataset_split["id"]
    categories = dataset_split["category"]
    for idx, (sample_id, category) in enumerate(zip(ids, categories)):
        language = language_from_id(sample_id)
        category_language_samples[category][language].append(idx)

    # Select samples to evaluate
    samples_to_evaluate: List[int] = []
    if max_samples is not None:
        logger.info(
            f"  Selecting up to {max_samples} samples per category per language")
        for category, lang_dict in category_language_samples.items():
            for lang in ["english", "hindi"]:
                indices = lang_dict[lang]
                selected = indices[:max_samples]
                samples_to_evaluate.extend(selected)
                logger.info(
                    f"    {category} ({lang}): {len(selected)} samples")
    else:
        samples_to_evaluate = list(range(len(dataset_split)))

    n_samples = len(samples_to_evaluate)
    logger.info(f"  Total samples to evaluate: {n_samples}")

    checkpoint_file = os.path.join(
        output_dir, f"steered_checkpoint_{split_name}.json")

    processed_ids: set = set()
    results: Dict = {
        "split": split_name,
        "total_samples": n_samples,
        "correct_steered": 0,
        "correct_baseline": 0,
        "predictions": [],
    }
    category_results: Dict = {"steered": {}, "baseline": {}}

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
            results = checkpoint_data
            for pred in results["predictions"]:
                processed_ids.add(pred["id"])
            logger.info(
                f"Resuming from checkpoint: {len(processed_ids)} samples already processed"
            )
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")

    samples_since_checkpoint = 0

    for idx in tqdm(samples_to_evaluate, desc=f"Evaluating {split_name}"):
        sample = dataset_split[idx]
        sample_id = sample["id"]

        if sample_id in processed_ids:
            continue

        question = sample["question"]
        ground_truth = sample["answer"]
        category = sample["category"]
        images = sample["images"]

        prompt = prompt_template.format(question=question)

        content: list = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        try:
            # Baseline generation (no steering)
            baseline_response = generate_fn(
                model=model,
                processor=processor,
                messages=messages,
                key_steer_vectors=None,
                value_steer_vectors=None,
                max_new_tokens=max_new_tokens,
                device=device,
            )

            # Steered generation
            steered_response = generate_fn(
                model=model,
                processor=processor,
                messages=messages,
                key_steer_vectors=key_steer_vectors,
                value_steer_vectors=value_steer_vectors,
                key_scale=key_scale,
                value_scale=value_scale,
                steer_layers=steer_layers,
                max_new_tokens=max_new_tokens,
                device=device,
            )

            baseline_pred = extract_answer_qwen(baseline_response)
            steered_pred = extract_answer_qwen(steered_response)

            baseline_correct = compute_exact_match(baseline_pred, ground_truth)
            steered_correct = compute_exact_match(steered_pred, ground_truth)

            if baseline_correct:
                results["correct_baseline"] += 1
            if steered_correct:
                results["correct_steered"] += 1

            for mode, is_correct in [
                ("steered", steered_correct),
                ("baseline", baseline_correct),
            ]:
                if category not in category_results[mode]:
                    category_results[mode][category] = {
                        "correct": 0, "total": 0}
                category_results[mode][category]["total"] += 1
                if is_correct:
                    category_results[mode][category]["correct"] += 1

            results["predictions"].append(
                {
                    "id": sample_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "category": category,
                    "baseline_response": baseline_response,
                    "baseline_prediction": baseline_pred,
                    "baseline_correct": baseline_correct,
                    "steered_response": steered_response,
                    "steered_prediction": steered_pred,
                    "steered_correct": steered_correct,
                }
            )

        except Exception as e:
            logger.error(f"Error processing sample {sample_id}: {e}")
            results["predictions"].append(
                {
                    "id": sample_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "category": category,
                    "error": str(e),
                }
            )

        finally:
            torch.cuda.empty_cache()
            samples_since_checkpoint += 1
            if samples_since_checkpoint >= checkpoint_interval:
                save_steered_checkpoint(
                    checkpoint_file, results, category_results)
                samples_since_checkpoint = 0

    compute_steered_metrics(results, category_results)
    save_steered_checkpoint(checkpoint_file, results, category_results)
    return results


# =============================================================================
# JUDGE EVALUATION
# =============================================================================

def _batch_judge_generate(judge_model, tasks, max_new_tokens=1024):
    """
    Generate judge responses for a batch of tasks.

    Tries ``judge_model.batch_chat()`` when available and the batch has more
    than one item.  Falls back to sequential ``judge_model.chat()`` calls
    otherwise (or on error).

    Args:
        judge_model: Model wrapper with ``chat()`` and optionally ``batch_chat()``.
        tasks: List of task dicts, each with ``system_prompt`` and ``judge_prompt`` keys.
        max_new_tokens: Maximum tokens to generate per response.

    Returns:
        List of response strings (``None`` for items that failed).
    """
    if not tasks:
        return []

    # Try true batch generation when the model supports it
    if len(tasks) > 1 and hasattr(judge_model, "batch_chat"):
        messages_list = []
        for task in tasks:
            messages = []
            if task.get("system_prompt"):
                messages.append({"role": "system", "content": task["system_prompt"]})
            messages.append({"role": "user", "content": task["judge_prompt"]})
            messages_list.append(messages)

        try:
            return judge_model.batch_chat(messages_list, max_new_tokens=max_new_tokens)
        except Exception as e:
            logger.warning(f"Batch generation failed, falling back to sequential: {e}")

    # Sequential fallback
    responses = []
    for task in tasks:
        try:
            response = judge_model.chat(
                user_message=task["judge_prompt"],
                system_prompt=task.get("system_prompt"),
                max_new_tokens=max_new_tokens,
            )
            responses.append(response)
        except Exception as e:
            logger.error(f"Error generating judge response: {e}")
            responses.append(None)
    return responses


def run_judge_evaluation(
    judge_model,
    predictions_file: str,
    output_dir: str,
    checkpoint_interval: int = 50,
    data_key_path: Optional[List[str]] = None,
    judge_false_only: bool = False,
    judge_response_type: str = "single",
    batch_size: int = 1,
) -> Dict:
    """
    Run LLM-as-Judge evaluation on existing predictions.

    Supports both **single-response** predictions (from the pipeline script) and
    **dual-response** predictions with steered/baseline columns (from the KV
    steering script).  Set ``judge_response_type`` accordingly:

    * ``"single"`` - expects ``full_response`` / ``prediction`` columns.
    * ``"steered"`` / ``"baseline"`` / ``"both"`` - expects
      ``steered_response`` and/or ``baseline_response`` columns.

    Args:
        judge_model: Model wrapper with a ``.chat()`` method.
        predictions_file: Path to JSON file with predictions.
        output_dir: Directory for saving results.
        checkpoint_interval: Save checkpoint every *K* samples.
        data_key_path: Key path to navigate to predictions list in JSON.
        judge_false_only: Only evaluate samples where exact match was incorrect.
        judge_response_type: ``"single"``, ``"steered"``, ``"baseline"``, or ``"both"``.
        batch_size: Number of predictions to judge per batch (default: 1).

    Returns:
        Dictionary with judge evaluation results.
    """
    logger.info(f"Running LLM-as-Judge evaluation on {predictions_file}...")
    logger.info(f"Judge response type: {judge_response_type}")
    logger.info(f"Judge false only: {judge_false_only}")

    if data_key_path is None:
        data_key_path = ["predictions"]

    with open(predictions_file, "r") as f:
        data = json.load(f)

    predictions = data
    for key in data_key_path:
        predictions = predictions.get(key, {})

    if not isinstance(predictions, list):
        raise ValueError(
            f"Could not find predictions list at path: {data_key_path}")

    logger.info(f"Found {len(predictions)} predictions to evaluate")

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = os.path.join(
        output_dir, "judge_evaluation_checkpoint.json")

    processed_ids: set = set()

    # Build initial results dict depending on mode
    if judge_response_type == "single":
        judge_results: Dict = {
            "total_samples": len(predictions),
            "judge_correct": 0,
            "exact_match_correct": 0,
            "evaluations": [],
        }
    else:
        judge_results = {
            "total_samples": len(predictions),
            "judge_correct_steered": 0,
            "judge_correct_baseline": 0,
            "exact_match_correct_steered": 0,
            "exact_match_correct_baseline": 0,
            "evaluations": [],
        }

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                judge_results = json.load(f)
            for eval_item in judge_results["evaluations"]:
                processed_ids.add(eval_item["id"])
            logger.info(
                f"Resuming from checkpoint: {len(processed_ids)} samples already processed"
            )
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")

    samples_since_checkpoint = 0

    # Filter to unprocessed predictions
    unprocessed_preds = [p for p in predictions if p.get("id") not in processed_ids]
    logger.info(
        f"Prepared {len(unprocessed_preds)} predictions to evaluate (batch_size={batch_size})"
    )

    n_batches = (
        (len(unprocessed_preds) + batch_size - 1) // batch_size
        if unprocessed_preds
        else 0
    )

    for batch_idx in tqdm(range(n_batches), desc="Judge evaluation (batches)"):
        batch_start = batch_idx * batch_size
        batch_preds = unprocessed_preds[batch_start : batch_start + batch_size]

        # ----- SINGLE response mode -----
        if judge_response_type == "single":
            # Prepare judge tasks for this batch
            batch_tasks = []
            for pred in batch_preds:
                model_full_response = pred.get("full_response")
                model_prediction = pred.get("prediction")
                if model_full_response is None:
                    continue

                exact_match_correct = pred.get("is_correct", False)
                if judge_false_only and exact_match_correct:
                    continue

                system_prompt, judge_prompt = create_judge_prompt_components(
                    pred.get("question"), pred.get("ground_truth"), model_full_response
                )

                batch_tasks.append(
                    {
                        "pred": pred,
                        "system_prompt": system_prompt,
                        "judge_prompt": judge_prompt,
                        "exact_match_correct": exact_match_correct,
                    }
                )

            if not batch_tasks:
                continue

            # Batch generate
            judge_responses = _batch_judge_generate(
                judge_model, batch_tasks, max_new_tokens=1024
            )

            # Process results
            for task, judge_response in zip(batch_tasks, judge_responses):
                pred = task["pred"]
                sample_id = pred.get("id")
                category = pred.get("category")
                question = pred.get("question")
                ground_truth = pred.get("ground_truth")
                model_prediction = pred.get("prediction")
                model_full_response = pred.get("full_response")
                exact_match_correct = task["exact_match_correct"]

                if judge_response is not None:
                    judge_verdict = parse_judge_response(judge_response)

                    if exact_match_correct:
                        judge_results["exact_match_correct"] += 1
                    if judge_verdict.get("is_correct", False):
                        judge_results["judge_correct"] += 1

                    judge_results["evaluations"].append(
                        {
                            "id": sample_id,
                            "category": category,
                            "question": question,
                            "ground_truth": ground_truth,
                            "model_prediction": model_prediction,
                            "model_full_response": model_full_response,
                            "exact_match_correct": exact_match_correct,
                            "judge_correct": judge_verdict.get("is_correct", False),
                            "judge_reasoning": judge_verdict.get("reasoning", ""),
                            "judge_raw_response": judge_response,
                        }
                    )
                else:
                    logger.error(
                        f"Error processing sample {sample_id} with judge"
                    )
                    judge_results["evaluations"].append(
                        {
                            "id": sample_id,
                            "category": category,
                            "question": question,
                            "ground_truth": ground_truth,
                            "model_prediction": model_prediction,
                            "model_full_response": model_full_response,
                            "exact_match_correct": exact_match_correct,
                            "judge_correct": False,
                            "judge_reasoning": "Error: generation failed",
                            "judge_raw_response": "",
                            "error": True,
                        }
                    )

        # ----- DUAL response mode (steered / baseline / both) -----
        else:
            # Prepare all judge tasks for this batch of predictions
            batch_tasks = []  # flat list of tasks needing a model call
            batch_eval_entries = []  # one eval entry per prediction

            for pred in batch_preds:
                sample_id = pred.get("id")
                question = pred.get("question")
                ground_truth = pred.get("ground_truth")
                category = pred.get("category")

                responses_to_judge: list = []
                if judge_response_type in ["steered", "both"]:
                    sr = pred.get("steered_response")
                    sp = pred.get("steered_prediction")
                    sc = pred.get("steered_correct", False)
                    if sr is not None:
                        responses_to_judge.append(
                            {"type": "steered", "full_response": sr,
                                "prediction": sp, "exact_match_correct": sc}
                        )
                if judge_response_type in ["baseline", "both"]:
                    br = pred.get("baseline_response")
                    bp = pred.get("baseline_prediction")
                    bc = pred.get("baseline_correct", False)
                    if br is not None:
                        responses_to_judge.append(
                            {"type": "baseline", "full_response": br,
                                "prediction": bp, "exact_match_correct": bc}
                        )

                if not responses_to_judge:
                    continue

                eval_entry: Dict = {
                    "id": sample_id,
                    "category": category,
                    "question": question,
                    "ground_truth": ground_truth,
                }

                for resp_data in responses_to_judge:
                    resp_type = resp_data["type"]
                    full_response = resp_data["full_response"]
                    prediction = resp_data["prediction"]
                    exact_match_correct = resp_data["exact_match_correct"]

                    if judge_false_only and exact_match_correct:
                        eval_entry[f"{resp_type}_prediction"] = prediction
                        eval_entry[f"{resp_type}_full_response"] = full_response
                        eval_entry[f"{resp_type}_exact_match_correct"] = exact_match_correct
                        eval_entry[f"{resp_type}_judge_correct"] = None
                        eval_entry[f"{resp_type}_judge_reasoning"] = "Skipped (exact match was correct)"
                        eval_entry[f"{resp_type}_judge_raw_response"] = None
                        continue

                    if resp_type == "steered":
                        judge_results["exact_match_correct_steered"] += int(
                            exact_match_correct)
                    else:
                        judge_results["exact_match_correct_baseline"] += int(
                            exact_match_correct)

                    system_prompt, judge_prompt = create_judge_prompt_components(
                        question, ground_truth, full_response)

                    batch_tasks.append(
                        {
                            "eval_entry": eval_entry,
                            "resp_type": resp_type,
                            "prediction": prediction,
                            "full_response": full_response,
                            "exact_match_correct": exact_match_correct,
                            "system_prompt": system_prompt,
                            "judge_prompt": judge_prompt,
                        }
                    )

                batch_eval_entries.append(eval_entry)

            # Batch generate all judge responses for dual-mode tasks
            if batch_tasks:
                judge_responses = _batch_judge_generate(
                    judge_model, batch_tasks, max_new_tokens=1024
                )

                for task, judge_response in zip(batch_tasks, judge_responses):
                    eval_entry = task["eval_entry"]
                    resp_type = task["resp_type"]
                    prediction = task["prediction"]
                    full_response = task["full_response"]
                    exact_match_correct = task["exact_match_correct"]

                    if judge_response is not None:
                        judge_verdict = parse_judge_response(judge_response)

                        if judge_verdict.get("is_correct", False):
                            if resp_type == "steered":
                                judge_results["judge_correct_steered"] += 1
                            else:
                                judge_results["judge_correct_baseline"] += 1

                        eval_entry[f"{resp_type}_prediction"] = prediction
                        eval_entry[f"{resp_type}_full_response"] = full_response
                        eval_entry[f"{resp_type}_exact_match_correct"] = exact_match_correct
                        eval_entry[f"{resp_type}_judge_correct"] = judge_verdict.get(
                            "is_correct", False)
                        eval_entry[f"{resp_type}_judge_reasoning"] = judge_verdict.get(
                            "reasoning", "")
                        eval_entry[f"{resp_type}_judge_raw_response"] = judge_response
                    else:
                        sample_id = eval_entry["id"]
                        logger.error(
                            f"Error processing sample {sample_id} ({resp_type}) with judge"
                        )
                        eval_entry[f"{resp_type}_prediction"] = prediction
                        eval_entry[f"{resp_type}_full_response"] = full_response
                        eval_entry[f"{resp_type}_exact_match_correct"] = exact_match_correct
                        eval_entry[f"{resp_type}_judge_correct"] = False
                        eval_entry[f"{resp_type}_judge_reasoning"] = "Error: generation failed"
                        eval_entry[f"{resp_type}_judge_raw_response"] = ""
                        eval_entry["error"] = True

            # Append completed eval entries for this batch
            for eval_entry in batch_eval_entries:
                judge_results["evaluations"].append(eval_entry)

        torch.cuda.empty_cache()
        samples_since_checkpoint += len(batch_preds)
        if samples_since_checkpoint >= checkpoint_interval:
            save_judge_checkpoint(checkpoint_file, judge_results)
            samples_since_checkpoint = 0

    # ---- Compute final metrics ----
    if judge_response_type == "single":
        total_evaluated = len(
            [e for e in judge_results["evaluations"]
                if not e.get("error", False)]
        )
        judge_results["judge_accuracy"] = (
            judge_results["judge_correct"] /
            total_evaluated if total_evaluated > 0 else 0
        )
        judge_results["exact_match_accuracy"] = (
            judge_results["exact_match_correct"] /
            total_evaluated if total_evaluated > 0 else 0
        )
    else:
        total_steered = sum(
            1
            for e in judge_results["evaluations"]
            if "steered_judge_correct" in e and e.get("steered_judge_correct") is not None
        )
        total_baseline = sum(
            1
            for e in judge_results["evaluations"]
            if "baseline_judge_correct" in e and e.get("baseline_judge_correct") is not None
        )
        if total_steered > 0:
            judge_results["judge_accuracy_steered"] = judge_results["judge_correct_steered"] / total_steered
            judge_results["exact_match_accuracy_steered"] = judge_results["exact_match_correct_steered"] / total_steered
        else:
            judge_results["judge_accuracy_steered"] = 0
            judge_results["exact_match_accuracy_steered"] = 0
        if total_baseline > 0:
            judge_results["judge_accuracy_baseline"] = judge_results["judge_correct_baseline"] / total_baseline
            judge_results["exact_match_accuracy_baseline"] = judge_results["exact_match_correct_baseline"] / total_baseline
        else:
            judge_results["judge_accuracy_baseline"] = 0
            judge_results["exact_match_accuracy_baseline"] = 0

    save_judge_checkpoint(checkpoint_file, judge_results)
    return judge_results


# =============================================================================
# SUMMARY PRINTING
# =============================================================================

def print_inference_summary(results: Dict, model_info: Optional[Dict] = None):
    """Print evaluation summary for single-response inference."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY - INFERENCE MODE")
    print("=" * 80)

    if model_info:
        print(f"Model: {model_info.get('model_name', 'Unknown')}")
        print(
            f"4-bit Quantization: {model_info.get('quantization_4bit', False)}")
        print(
            f"Flash Attention 2: {model_info.get('use_flash_attention', False)}")
        print("-" * 80)

    print(
        f"\nOverall Accuracy: {results.get('accuracy', 0):.2%} "
        f"({results.get('correct', 0)}/{results.get('total_samples', 0)})"
    )

    if "category_accuracy" in results:
        print("\nCategory-wise breakdown:")
        for cat, cat_acc in results["category_accuracy"].items():
            print(
                f"  - {cat}: {cat_acc['accuracy']:.2%} ({cat_acc['correct']}/{cat_acc['total']})")
    print("=" * 80)


def print_steered_summary(results: Dict, config: Dict):
    """Print evaluation summary for steered + baseline results."""
    print("\n" + "=" * 80)
    print("KV CACHE STEERING EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Model: {config.get('model_name', 'Unknown')}")
    print(f"Key Scale: {config.get('key_scale', 0.0)}")
    print(f"Value Scale: {config.get('value_scale', 1.0)}")
    print(f"Num Contrastive Pairs: {config.get('num_contrastive_pairs', 0)}")
    print(
        f"Positive Prompt Type: {config.get('positive_prompt_type', 'Unknown')}")
    print("-" * 80)

    print("\nOverall Accuracy:")
    print(
        f"  Baseline:  {results.get('accuracy_baseline', 0):.2%} "
        f"({results.get('correct_baseline', 0)}/{results.get('total_samples', 0)})"
    )
    print(
        f"  Steered:   {results.get('accuracy_steered', 0):.2%} "
        f"({results.get('correct_steered', 0)}/{results.get('total_samples', 0)})"
    )
    improvement = results.get("accuracy_steered", 0) - \
        results.get("accuracy_baseline", 0)
    print(f"  Improvement: {improvement:+.2%}")

    if "category_accuracy" in results:
        print("\nCategory-wise breakdown:")
        categories: set = set()
        for mode in ["baseline", "steered"]:
            if mode in results["category_accuracy"]:
                categories.update(results["category_accuracy"][mode].keys())
        for cat in sorted(categories):
            baseline_acc = results["category_accuracy"].get(
                "baseline", {}).get(cat, {}).get("accuracy", 0)
            steered_acc = results["category_accuracy"].get(
                "steered", {}).get(cat, {}).get("accuracy", 0)
            diff = steered_acc - baseline_acc
            print(
                f"  {cat}: Baseline {baseline_acc:.2%} -> Steered {steered_acc:.2%} ({diff:+.2%})")
    print("=" * 80)


def print_judge_summary(results: Dict, judge_response_type: str = "single"):
    """Print summary for judge evaluation results."""
    print("\n" + "=" * 80)
    print("LLM-AS-JUDGE EVALUATION SUMMARY")
    print("=" * 80)

    if judge_response_type == "single":
        total = len([e for e in results["evaluations"]
                    if not e.get("error", False)])
        print(
            f"\nJudge Accuracy: {results.get('judge_accuracy', 0):.2%} "
            f"({results.get('judge_correct', 0)}/{total})"
        )
        print(
            f"Exact Match Accuracy: {results.get('exact_match_accuracy', 0):.2%} "
            f"({results.get('exact_match_correct', 0)}/{total})"
        )
        agreements = sum(
            1
            for e in results["evaluations"]
            if e.get("exact_match_correct") == e.get("judge_correct") and not e.get("error", False)
        )
        print(
            f"\nAgreement Rate: {agreements / total:.2%} ({agreements}/{total})")
    else:
        if "judge_accuracy_steered" in results:
            total_steered = sum(
                1
                for e in results["evaluations"]
                if "steered_judge_correct" in e and e.get("steered_judge_correct") is not None
            )
            print(f"\nSteered Responses:")
            print(
                f"  Judge Accuracy: {results.get('judge_accuracy_steered', 0):.2%} "
                f"({results.get('judge_correct_steered', 0)}/{total_steered})"
            )
            print(
                f"  Exact Match Accuracy: {results.get('exact_match_accuracy_steered', 0):.2%} "
                f"({results.get('exact_match_correct_steered', 0)}/{total_steered})"
            )
        if "judge_accuracy_baseline" in results:
            total_baseline = sum(
                1
                for e in results["evaluations"]
                if "baseline_judge_correct" in e and e.get("baseline_judge_correct") is not None
            )
            print(f"\nBaseline Responses:")
            print(
                f"  Judge Accuracy: {results.get('judge_accuracy_baseline', 0):.2%} "
                f"({results.get('judge_correct_baseline', 0)}/{total_baseline})"
            )
            print(
                f"  Exact Match Accuracy: {results.get('exact_match_accuracy_baseline', 0):.2%} "
                f"({results.get('exact_match_correct_baseline', 0)}/{total_baseline})"
            )
        if "judge_accuracy_steered" in results and "judge_accuracy_baseline" in results:
            improvement = results.get(
                "judge_accuracy_steered", 0) - results.get("judge_accuracy_baseline", 0)
            print(f"\nJudge Accuracy Improvement: {improvement:+.2%}")
    print("=" * 80)


# =============================================================================
# POST-HOC ANALYSIS HELPERS
# =============================================================================

def english_hindi_accuracy(predictions: List[Dict], correct_key: str = "is_correct") -> Dict:
    """
    Compute per-language accuracy from a list of predictions.

    Args:
        predictions: List of prediction dicts with ``id`` and a correctness key.
        correct_key: Key in each dict indicating correctness.

    Returns:
        Dict with ``hindi_accuracy``, ``english_accuracy``, ``gap``, and counts.
    """
    hindi = [p for p in predictions if language_from_id(p["id"]) == "hindi"]
    english = [p for p in predictions if language_from_id(
        p["id"]) == "english"]

    hindi_correct = sum(1 for p in hindi if p.get(correct_key, False))
    english_correct = sum(1 for p in english if p.get(correct_key, False))

    hindi_acc = hindi_correct / len(hindi) if hindi else 0
    english_acc = english_correct / len(english) if english else 0

    return {
        "hindi_accuracy": hindi_acc,
        "hindi_correct": hindi_correct,
        "hindi_total": len(hindi),
        "english_accuracy": english_acc,
        "english_correct": english_correct,
        "english_total": len(english),
        "gap": english_acc - hindi_acc,
    }


def class_accuracy(
    predictions: List[Dict],
    baseline_key: str = "baseline_correct",
    steered_key: str = "steered_correct",
    null_replacement: bool = True
) -> Dict[str, Dict]:
    """
    Compute per (language × category) accuracy from steered-evaluation predictions.

    Args:
        predictions: Prediction dicts with ``id``, ``category``, and correctness keys.
        baseline_key: Key for baseline correctness.
        steered_key: Key for steered correctness.
        null_replacement: Replace null with this, True by default.

    Returns:
        Nested dict: ``{class_name: {total, baseline_correct, steered_correct, ...}}``.
    """
    class_counts: Dict = {}

    for item in predictions:
        cat = item["category"]
        lang = language_from_id(item["id"])
        split = split_from_id(item["id"])
        cls = f"{lang}-{cat}"

        if cls not in class_counts:
            class_counts[cls] = {
                "total": 0,
                "baseline_correct": 0,
                "steered_correct": 0,
                "split": split,
            }

        class_counts[cls]["total"] += 1

        # 1. Get value. Default to False (0) if key is ABSENT.
        b_val = item.get(baseline_key, False)
        s_val = item.get(steered_key, False)

        # 2. If key was PRESENT but value is None (JSON null), apply replacement.
        if b_val is None:
            b_val = null_replacement
        if s_val is None:
            s_val = null_replacement

        # 3. Increment if Truthy
        if b_val:
            class_counts[cls]["baseline_correct"] += 1
        if s_val:
            class_counts[cls]["steered_correct"] += 1

    # Add accuracy fields
    for cls, counts in class_counts.items():
        t = counts["total"]
        counts["baseline_accuracy"] = counts["baseline_correct"] / t if t else 0
        counts["steered_accuracy"] = counts["steered_correct"] / t if t else 0

    return class_counts


def class_accuracy_single_key(
    predictions: List[Dict],
    key: str = "judge_correct",
    null_replacement: bool = True
) -> Dict[str, Dict]:
    """
    Compute per (language × category) accuracy from predictions using a single correctness key.

    Args:
        predictions: Prediction dicts with ``id``, ``category``, and correctness key.
        key: Key for correctness field to analyze.
        null_replacement: Replace null with this, True by default.

    Returns:
        Nested dict: ``{class_name: {total, correct, accuracy, split}}``.
    """
    class_counts: Dict = {}

    for item in predictions:
        cat = item["category"]
        lang = language_from_id(item["id"])
        split = split_from_id(item["id"])
        cls = f"{lang}-{cat}"

        if cls not in class_counts:
            class_counts[cls] = {
                "total": 0,
                "correct": 0,
                "split": split,
            }

        class_counts[cls]["total"] += 1

        # 1. Get value. Default to False (0) if key is ABSENT.
        val = item.get(key, False)

        # 2. If key was PRESENT but value is None (JSON null), apply replacement.
        if val is None:
            val = null_replacement

        # 3. Increment if Truthy
        if val:
            class_counts[cls]["correct"] += 1

    # Add accuracy fields
    for cls, counts in class_counts.items():
        t = counts["total"]
        counts["accuracy"] = counts["correct"] / t if t else 0

    return class_counts
