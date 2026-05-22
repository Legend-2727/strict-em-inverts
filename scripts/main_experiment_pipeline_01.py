"""
Main Experiment Pipeline for MMCricBench Evaluation.

This script provides a unified pipeline for:
1) Running inference using VLM (Qwen-2.5 VL 3B/7B Instruct) on MMCricBench
2) Using LLM-as-Judge (Qwen3-4B) to evaluate the outputs

Usage Examples:
--------------
# Run inference with 3B model
python scripts/main_experiment_pipeline_01.py --mode inference --model-size 3b

# Run inference with 7B model, no quantization, low-res images
python scripts/main_experiment_pipeline_01.py --mode inference --model-size 7b --no-quantization --lowresolution

# Run LLM-as-Judge evaluation on existing predictions
python scripts/main_experiment_pipeline_01.py --mode judge --predictions-file results/checkpoint_test_single.json

# Run both inference and judge in one go
python scripts/main_experiment_pipeline_01.py --mode both --model-size 3b
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Optional

import torch
from dotenv import load_dotenv

from src.utils.logging import get_logger, setup_logging
from src.models.qwen25_wrapper_2 import Qwen25Wrapper2
from src.models.qwen3_4b import Qwen3_4BWrapper
from src.data import load_mmcricbench
from src.data.prompts import NEGATIVE_PROMPT_TEMPLATE, POSITIVE_PROMPTS
from src.evaluation import (
    run_inference_loop,
    run_judge_evaluation,
    print_inference_summary,
    print_judge_summary,
)

# Configure logging
setup_logging(level="INFO")
logger = get_logger(__name__)

load_dotenv()


def main():
    """Main experiment pipeline function."""
    parser = argparse.ArgumentParser(
        description="Main Experiment Pipeline for MMCricBench Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection
    parser.add_argument("--mode", type=str, choices=["inference", "judge", "both"], required=True,
                        help="Pipeline mode: 'inference', 'judge', or 'both'")

    # Model configuration
    parser.add_argument("--model-size", type=str, choices=["3b", "7b"], default="3b",
                        help="VLM model size (default: 3b)")
    parser.add_argument("--no-quantization", action="store_true",
                        help="Disable 4-bit quantization")
    parser.add_argument("--no-flash-attention", action="store_true",
                        help="Disable flash attention 2")
    parser.add_argument("--lowresolution", action="store_true",
                        help="Use low resolution images (512*28*28 max_pixels)")

    # Dataset configuration
    parser.add_argument("--split", type=str, choices=["single", "multi", "both"], default="single",
                        help="Which split(s) to evaluate (default: single)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples per split (default: all)")
    
    # prompt configuration
    parser.add_argument("--prompt-template", type=str, default="negative",
                        help="Prompt template to use: 'negative' or 'positive~{key}' (e.g., 'positive~translate-first')")

    # I/O configuration
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--predictions-file", type=str, default=None,
                        help="Path to predictions file (required for judge mode)")
    parser.add_argument("--checkpoint-interval", type=int, default=50,
                        help="Save checkpoint every K samples (default: 50)")

    # Judge configuration
    parser.add_argument("--judge-false-only", action="store_true",
                        help="Only evaluate samples where exact match was incorrect")
    parser.add_argument("--judge-batch-size", type=int, default=1,
                        help="Number of predictions to judge per batch (default: 1)")

    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"experiment_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Validate arguments
    if args.mode in ["judge"] and args.predictions_file is None:
        parser.error("--predictions-file is required for judge mode")
    
    # Determine prompt template
    if args.prompt_template == "negative":
        prompt_template = NEGATIVE_PROMPT_TEMPLATE
    elif args.prompt_template.startswith("positive"):
        key = args.prompt_template.split("~", 1)[-1]
        prompt_template = POSITIVE_PROMPTS.get(key)
        if prompt_template is None:
            parser.error(f"Invalid positive prompt key: {key}")
    else:
        parser.error(f"Invalid prompt template: {args.prompt_template}")
    
    print(f"Using prompt template: {args.prompt_template}")

    # === INFERENCE MODE ===
    if args.mode in ["inference", "both"]:
        logger.info("=" * 80)
        logger.info("STARTING INFERENCE MODE")
        logger.info("=" * 80)

        dataset = load_mmcricbench()

        model_name = (
            "Qwen/Qwen2.5-VL-7B-Instruct" if args.model_size == "7b"
            else "Qwen/Qwen2.5-VL-3B-Instruct"
        )
        logger.info(f"Initializing VLM model: {model_name}...")

        model_kwargs: dict = {
            "model_name": model_name,
            "quantization_4bit": not args.no_quantization,
            "use_flash_attention": not args.no_flash_attention,
        }
        if args.lowresolution:
            model_kwargs["max_pixels"] = 512 * 28 * 28

        vlm_model = Qwen25Wrapper2(**model_kwargs)
        model_info = vlm_model.get_model_info()
        logger.info(f"Model info: {model_info}")

        all_results: Dict = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "quantization_4bit": not args.no_quantization,
            "use_flash_attention": not args.no_flash_attention,
            "lowresolution": args.lowresolution,
            "splits": {},
        }

        for split_tag in ["single", "multi"]:
            if args.split not in [split_tag, "both"]:
                continue
            split_name = f"test_{split_tag}"
            results = run_inference_loop(
                model=vlm_model,
                dataset_split=dataset[split_name],
                split_name=split_name,
                prompt_template=prompt_template,
                max_samples=args.max_samples,
                output_dir=output_dir,
                checkpoint_interval=args.checkpoint_interval,
            )
            all_results["splits"][split_name] = results
            print_inference_summary(results, model_info)

        inference_output_file = os.path.join(
            output_dir, "inference_results.json")
        with open(inference_output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Inference results saved to {inference_output_file}")

        if args.mode == "both":
            del vlm_model
            torch.cuda.empty_cache()
            logger.info("VLM model unloaded to free memory for judge")

    # === JUDGE MODE ===
    if args.mode in ["judge", "both"]:
        logger.info("=" * 80)
        logger.info("STARTING JUDGE MODE")
        logger.info("=" * 80)

        if args.mode == "both":
            predictions_file = os.path.join(
                output_dir, "checkpoint_test_single.json")
        else:
            predictions_file = args.predictions_file

        if not os.path.exists(predictions_file):
            logger.error(f"Predictions file not found: {predictions_file}")
            sys.exit(1)

        logger.info("Initializing LLM Judge (Qwen3-4B-Instruct-2507)...")
        judge_model = Qwen3_4BWrapper(
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            quantization_4bit=not args.no_quantization,
            use_flash_attention=not args.no_flash_attention,
        )
        logger.info(f"Judge model info: {judge_model.get_model_info()}")

        # Detect split paths
        try:
            with open(predictions_file, "r") as f:
                pred_data = json.load(f)
            splits = pred_data.get("splits", {}).keys()
        except Exception as e:
            logger.error(f"Failed to load predictions file: {e}")
            splits = []

        paths = [["splits", split, "predictions"] for split in splits]

        judge_results: Dict = {}
        for path in paths:
            logger.info(f"Running judge evaluation on path: {path}...")
            jr = run_judge_evaluation(
                judge_model=judge_model,
                predictions_file=predictions_file,
                output_dir=output_dir,
                checkpoint_interval=args.checkpoint_interval,
                data_key_path=path,
                judge_false_only=args.judge_false_only,
                judge_response_type="single",
                batch_size=args.judge_batch_size,
            )
            judge_results.update(jr)

        print_judge_summary(judge_results, judge_response_type="single")

        judge_output_file = os.path.join(
            output_dir, "judge_evaluation_results.json")
        with open(judge_output_file, "w") as f:
            json.dump(judge_results, f, indent=2)
        logger.info(f"Judge evaluation results saved to {judge_output_file}")

    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"All results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
