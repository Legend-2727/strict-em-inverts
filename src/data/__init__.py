"""
Data loading utilities for the dynamix project.

Provides shared dataset loading functions used across experiment scripts:
- MMCricBench dataset loading
- XFUND QA-conversion loading
- GSM8K sample loading
- Dataset entry selection and language utilities
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency in some environments
    def load_dotenv(*_args, **_kwargs):
        return False
from .xfund import (
    XFUND_LANGUAGES,
    build_xfund_single_view_samples,
    filter_xfund_pairs,
    filter_xfund_samples,
    load_xfund_qa_dataset,
    load_xfund_qa_paired_dataset,
)
from .mtvqa import (
    build_mtvqa_single_view_samples,
    filter_mtvqa_pairs,
    filter_mtvqa_samples,
    load_mtvqa_dataset,
    load_mtvqa_pair_dataset,
)

logger = logging.getLogger(__name__)


def load_mmcricbench(cache_dir: Optional[str] = None) -> Dict:
    """
    Load the MMCricBench dataset.

    Args:
        cache_dir: Optional cache directory for the dataset.
                   Falls back to MMCRICBENCH_CACHE env variable.

    Returns:
        Dataset dictionary with 'test_single' and 'test_multi' splits
    """
    logger.info("Loading MMCricBench dataset...")

    if cache_dir is None:
        load_dotenv()
        cache_dir = os.environ.get("MMCRICBENCH_CACHE")

    from datasets import load_dataset

    dataset = load_dataset("DIALab/MMCricBench", cache_dir=cache_dir)

    logger.info(f"Dataset loaded successfully:")
    logger.info(f"  - test_single: {len(dataset['test_single'])} samples")
    logger.info(f"  - test_multi: {len(dataset['test_multi'])} samples")

    return dataset


def load_gsm8k_from_directory(data_dir: str = "data/gsm8k_oai") -> List[Dict]:
    """
    Load GSM8K data from JSON files in a directory.

    Args:
        data_dir: Directory containing GSM8K JSON files

    Returns:
        List of GSM8K samples with 'query', 'steps', and 'solution' fields
    """
    logger.info(f"Loading GSM8K data from: {data_dir}")

    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"GSM8K data directory not found: {data_dir}")

    samples = []
    json_files = sorted(data_path.glob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                sample = json.load(f)
                samples.append(sample)
        except Exception as e:
            logger.warning(f"Error loading {json_file}: {e}")
            continue

    logger.info(f"Loaded {len(samples)} GSM8K samples")
    return samples


def select_entry_by_id(
    dataset: Dict,
    split: str,
    entry_id: str,
    id_colname: str = "id",
) -> Optional[Dict]:
    """
    Select a specific entry from the dataset by its ID.

    Args:
        dataset: The dataset dictionary
        split: The split name ('test_single' or 'test_multi')
        entry_id: The ID of the entry to select
        id_colname: Column name for the ID field

    Returns:
        The selected dataset entry, or None if not found
    """
    dataset_split = dataset[split]
    id_column = dataset_split[id_colname]

    for idx, current_id in enumerate(id_column):
        if current_id == entry_id:
            return dataset_split[idx]

    logger.warning(f"Entry ID {entry_id} not found in split {split}.")
    return None


def language_from_id(sample_id: str) -> str:
    """
    Extract the language from a sample ID.

    MMCricBench IDs follow the pattern: ``<language>-<split>-<number>``
    (e.g. ``"hindi-single-25"``).  This function returns the language prefix.

    Args:
        sample_id: The sample ID string

    Returns:
        Language string (e.g. "hindi", "english")
    """
    return sample_id.split("-")[0]


def split_from_id(sample_id: str) -> str:
    """
    Extract the split from a sample ID.

    MMCricBench IDs follow the pattern: ``<language>-<split>-<number>``
    (e.g. ``"hindi-single-25"``).  This function returns the split component.

    Args:
        sample_id: The sample ID string

    Returns:
        Split string (e.g. "single", "multi")
    """
    return sample_id.split("-")[1]


def split_merge(results_dict: Dict) -> List[Dict]:
    """
    Merge test_single and test_multi predictions from a results dictionary.

    Args:
        results_dict: Results dict with ``splits.test_single.predictions``
                      and ``splits.test_multi.predictions`` keys.

    Returns:
        Combined list of prediction dicts from both splits.
    """
    predictions = list(
        results_dict.get("splits", {}).get(
            "test_single", {}).get("predictions", [])
    )
    multi_predictions = (
        results_dict.get("splits", {}).get(
            "test_multi", {}).get("predictions", [])
    )
    predictions.extend(multi_predictions)
    return predictions


def split_merge_simple(results_dict: Dict) -> List[Dict]:
    """
    Merge test_single and test_multi predictions from a results dictionary.

    Args:
        results_dict: Results dict with ``test_single`` and ``test_multi`` keys.
    Returns:
        Combined list of prediction dicts from both splits.
    """
    evaluations = []
    for key, value in results_dict.items():
        evaluations.extend(value.get("evaluations", []))
    return evaluations


def split_merge_legacy(results_dict: Dict) -> List[Dict]:
    """
    Same output as split_merge, but the input does not have "test_single" and "test_multi" keys.
    Input only has "evaluations" key
    Args:
        results_dict: Results dict with ``evaluations`` key containing list of dicts.
    Returns:
        Combined list of prediction dicts from the evaluations list.
    """
    return results_dict.get("evaluations", [])
