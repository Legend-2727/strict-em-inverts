"""
Post-hoc analysis of MMCricBench experiment results.

Provides CLI-driven analysis helpers:
  - Hindi vs English accuracy from inference or judge results
  - Category × language breakdowns for steered/baseline experiments
  - Disagreement analysis between exact-match and judge evaluations

Usage Examples:
--------------
# Hindi vs English accuracy from inference results
python scripts/main_experiment_extra_analysis.py lang-accuracy results/inference_results.json

# Hindi vs English accuracy from judge evaluation results
python scripts/main_experiment_extra_analysis.py lang-accuracy results/judge_results.json --judge

# Category × language breakdown for steered experiments
python scripts/main_experiment_extra_analysis.py class-accuracy results/evaluation_results.json

# Disagreement analysis between judge and exact-match
python scripts/main_experiment_extra_analysis.py disagree results/evaluation_results.json
"""

import json
import os
import argparse
from typing import Dict, List, Optional

from src.data import language_from_id, split_merge, split_merge_simple, split_merge_legacy
from src.evaluation import english_hindi_accuracy, class_accuracy, class_accuracy_single_key


def load_json(filepath: str) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)


def parse_class_breakdown(class_breakdown: Dict[str, Dict]) -> None:
    languages = set()
    categories = set()

    # Structure: matrix[(category, language)] = {stats}
    matrix_data = {}

    # Aggregators for totals
    lang_totals = {}
    cat_totals = {}
    grand_total = {'baseline_correct': 0, 'steered_correct': 0, 'total': 0}

    for key, stats in class_breakdown.items():
        # Parse "language-category"
        try:
            lang, cat = key.split('-', 1)
        except ValueError:
            print(f"Skipping malformed key: {key}")
            continue

        languages.add(lang)
        categories.add(cat)

        matrix_data[(cat, lang)] = stats

        # Extract counts directly using the keys produced by class_accuracy
        n = stats.get('total', 0)
        b_correct = stats.get('baseline_correct', 0)
        s_correct = stats.get('steered_correct', 0)

        # Update Language Totals
        if lang not in lang_totals:
            lang_totals[lang] = {'b': 0, 's': 0, 't': 0}
        lang_totals[lang]['b'] += b_correct
        lang_totals[lang]['s'] += s_correct
        lang_totals[lang]['t'] += n

        # Update Category Totals
        if cat not in cat_totals:
            cat_totals[cat] = {'b': 0, 's': 0, 't': 0}
        cat_totals[cat]['b'] += b_correct
        cat_totals[cat]['s'] += s_correct
        cat_totals[cat]['t'] += n

        # Update Grand Total
        grand_total['baseline_correct'] += b_correct
        grand_total['steered_correct'] += s_correct
        grand_total['total'] += n

    # Sort for consistent display
    sorted_langs = sorted(list(languages))
    sorted_cats = sorted(list(categories))

    # Helper to format cell: "Base | Steer"
    def format_cell(b_correct, s_correct, total):
        if total == 0:
            return "  N/A  "
        # b_acc = b_correct / total
        # s_acc = s_correct / total
        # return f"{b_acc:.2f} | {s_acc:.2f}"
        return f"{b_correct}|{s_correct} /{total}"

    # --- Display Matrix ---

    col_width = 18
    first_col_width = 25

    # 1. Header
    header = f"{'Category':<{first_col_width}}"
    for lang in sorted_langs:
        header += f"{lang:^{col_width}}"
    header += f"{'TOTAL':^{col_width}}"

    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # 2. Rows (Categories)
    for cat in sorted_cats:

        prunedcat = cat if len(
            cat) <= first_col_width else cat[:first_col_width-3] + "..."
        row_str = f"{prunedcat:<{first_col_width}}"

        for lang in sorted_langs:
            if (cat, lang) in matrix_data:
                stats = matrix_data[(cat, lang)]
                n = stats.get('total', 0)
                b = stats.get('baseline_correct', 0)
                s = stats.get('steered_correct', 0)
                row_str += f"{format_cell(b, s, n):^{col_width}}"
            else:
                row_str += f"{'-':^{col_width}}"

        # Right edge: Category Total
        ct = cat_totals.get(cat, {'b': 0, 's': 0, 't': 0})
        row_str += f"{format_cell(ct['b'], ct['s'], ct['t']):^{col_width}}"

        print(row_str)

    print("-" * len(header))

    # 3. Bottom (Language Totals)
    footer = f"{'TOTAL':<{first_col_width}}"
    for lang in sorted_langs:
        lt = lang_totals.get(lang, {'b': 0, 's': 0, 't': 0})
        footer += f"{format_cell(lt['b'], lt['s'], lt['t']):^{col_width}}"

    # Grand Total (Bottom Right)
    gt_b = grand_total['baseline_correct']
    gt_s = grand_total['steered_correct']
    gt_t = grand_total['total']
    footer += f"{format_cell(gt_b, gt_s, gt_t):^{col_width}}"

    print(footer)
    print("-" * len(header))


# =============================================================================
# JSON SUMMARY HELPERS
# =============================================================================

def compute_breakdown_summary(class_breakdown: Dict[str, Dict], keys: List[str]) -> Dict:
    """
    Compute language totals, category totals, and grand total from a class breakdown dict.

    Parameters
    ----------
    class_breakdown : dict
        Keys like "english-BasicArithmetic", values have 'total' and one or more count keys.
    keys : list[str]
        The count keys to aggregate, e.g. ["correct"] or ["baseline_correct", "steered_correct"].

    Returns
    -------
    dict with 'per_class', 'per_language', 'per_category', 'grand_total'.
    """
    per_class = {}
    lang_totals: Dict[str, Dict] = {}
    cat_totals: Dict[str, Dict] = {}
    grand_total = {k: 0 for k in keys}
    grand_total["total"] = 0

    for cls_key, stats in class_breakdown.items():
        try:
            lang, cat = cls_key.split('-', 1)
        except ValueError:
            continue

        n = stats.get("total", 0)
        entry = {"total": n}
        for k in keys:
            entry[k] = stats.get(k, 0)
            entry[k.replace("correct", "accuracy")] = (
                stats.get(k, 0) / n if n else 0
            )
        entry["split"] = stats.get("split", None)
        per_class[cls_key] = entry

        # Accumulate language totals
        if lang not in lang_totals:
            lang_totals[lang] = {k: 0 for k in keys}
            lang_totals[lang]["total"] = 0
        lang_totals[lang]["total"] += n
        for k in keys:
            lang_totals[lang][k] += stats.get(k, 0)

        # Accumulate category totals
        if cat not in cat_totals:
            cat_totals[cat] = {k: 0 for k in keys}
            cat_totals[cat]["total"] = 0
        cat_totals[cat]["total"] += n
        for k in keys:
            cat_totals[cat][k] += stats.get(k, 0)

        # Accumulate grand total
        grand_total["total"] += n
        for k in keys:
            grand_total[k] += stats.get(k, 0)

    # Compute accuracies for aggregated totals
    for totals_dict in [lang_totals, cat_totals]:
        for group, vals in totals_dict.items():
            t = vals["total"]
            for k in keys:
                vals[k.replace("correct", "accuracy")] = vals[k] / t if t else 0

    t = grand_total["total"]
    for k in keys:
        grand_total[k.replace("correct", "accuracy")] = grand_total[k] / t if t else 0

    return {
        "per_class": per_class,
        "per_language": lang_totals,
        "per_category": cat_totals,
        "grand_total": grand_total,
    }


def save_json_output(data: Dict, output_path: str) -> None:
    """Save dict as JSON to the given path."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nJSON output saved to: {output_path}")


# =============================================================================
# ANALYSIS COMMANDS
# =============================================================================

def cmd_lang_accuracy(args):
    """Print per-language accuracy."""
    data = load_json(args.file)

    if args.judge:
        predictions = data.get("evaluations", [])
        correct_key = "judge_correct"
    else:
        predictions = split_merge(data)
        correct_key = "is_correct"

    stats = english_hindi_accuracy(predictions, correct_key=correct_key)

    label = "[Judge Eval]" if args.judge else "[Inference]"
    print(f"{label} Hindi Accuracy:   {stats['hindi_accuracy']:.4f} "
          f"({stats['hindi_correct']}/{stats['hindi_total']})")
    print(f"{label} English Accuracy: {stats['english_accuracy']:.4f} "
          f"({stats['english_correct']}/{stats['english_total']})")
    print(f"Gap (English - Hindi): {stats['gap']:.4f}")


def cmd_class_accuracy(args):
    """Print per (language × category) accuracy for steered experiments."""
    data = load_json(args.file)
    predictions = split_merge(data)

    breakdown = class_accuracy(predictions)

    print("Class-wise Accuracy:")
    for cls in sorted(breakdown):
        c = breakdown[cls]
        print(
            f"  {cls}: Total={c['total']}, "
            f"Baseline={c['baseline_correct']} ({c['baseline_accuracy']:.4f}), "
            f"Steered={c['steered_correct']} ({c['steered_accuracy']:.4f})"
        )

    # Language-wise summary
    print("\nLanguage-wise Accuracy:")
    for lang in ["hindi", "english"]:
        total = sum(c["total"]
                    for cls, c in breakdown.items() if cls.startswith(lang))
        bl = sum(c["baseline_correct"]
                 for cls, c in breakdown.items() if cls.startswith(lang))
        st = sum(c["steered_correct"]
                 for cls, c in breakdown.items() if cls.startswith(lang))
        bl_acc = bl / total if total else 0
        st_acc = st / total if total else 0
        print(
            f"  {lang}: Total={total}, "
            f"Baseline={bl} ({bl_acc:.4f}), "
            f"Steered={st} ({st_acc:.4f})"
        )


def cmd_class_accuracy_matrix(args):
    """Display class accuracy in matrix form."""
    data = load_json(args.file)
    predictions = split_merge_legacy(data)

    # Use correct_key to determine which field to analyze
    # Default to "is_correct" for inference results
    # correct_key = getattr(args, 'correct_key', 'is_correct')
    correct_key = "judge_correct"
    
    breakdown = class_accuracy_single_key(predictions, key=correct_key)

    # Prepare matrix data
    languages = set()
    categories = set()
    matrix_data = {}
    lang_totals = {}
    cat_totals = {}
    grand_total = {'correct': 0, 'total': 0}

    for key, stats in breakdown.items():
        # Parse "language-category"
        try:
            lang, cat = key.split('-', 1)
        except ValueError:
            print(f"Skipping malformed key: {key}")
            continue

        languages.add(lang)
        categories.add(cat)
        matrix_data[(cat, lang)] = stats

        n = stats.get('total', 0)
        correct = stats.get('correct', 0)

        # Update Language Totals
        if lang not in lang_totals:
            lang_totals[lang] = {'correct': 0, 'total': 0}
        lang_totals[lang]['correct'] += correct
        lang_totals[lang]['total'] += n

        # Update Category Totals
        if cat not in cat_totals:
            cat_totals[cat] = {'correct': 0, 'total': 0}
        cat_totals[cat]['correct'] += correct
        cat_totals[cat]['total'] += n

        # Update Grand Total
        grand_total['correct'] += correct
        grand_total['total'] += n

    # Sort for consistent display
    sorted_langs = sorted(list(languages))
    sorted_cats = sorted(list(categories))

    # Helper to format cell
    def format_cell(correct, total):
        if total == 0:
            return "  N/A  "
        return f"{correct}/{total}"

    # Display Matrix
    col_width = 18
    first_col_width = 25

    # 1. Header
    header = f"{'Category':<{first_col_width}}"
    for lang in sorted_langs:
        header += f"{lang:^{col_width}}"
    header += f"{'TOTAL':^{col_width}}"

    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # 2. Rows (Categories)
    for cat in sorted_cats:
        prunedcat = cat if len(cat) <= first_col_width else cat[:first_col_width-3] + "..."
        row_str = f"{prunedcat:<{first_col_width}}"

        for lang in sorted_langs:
            if (cat, lang) in matrix_data:
                stats = matrix_data[(cat, lang)]
                n = stats.get('total', 0)
                correct = stats.get('correct', 0)
                row_str += f"{format_cell(correct, n):^{col_width}}"
            else:
                row_str += f"{'-':^{col_width}}"

        # Right edge: Category Total
        ct = cat_totals.get(cat, {'correct': 0, 'total': 0})
        row_str += f"{format_cell(ct['correct'], ct['total']):^{col_width}}"

        print(row_str)

    print("-" * len(header))

    # 3. Bottom (Language Totals)
    footer = f"{'TOTAL':<{first_col_width}}"
    for lang in sorted_langs:
        lt = lang_totals.get(lang, {'correct': 0, 'total': 0})
        footer += f"{format_cell(lt['correct'], lt['total']):^{col_width}}"

    # Grand Total (Bottom Right)
    footer += f"{format_cell(grand_total['correct'], grand_total['total']):^{col_width}}"

    print(footer)
    print("-" * len(header))

    # --- JSON output ---
    json_output_path = getattr(args, "json_output", None)
    if json_output_path is None:
        json_output_path = os.path.join(
            os.path.dirname(args.file), "summary_class_accuracy_matrix.json"
        )

    json_data = {
        "analysis_type": "class-accuracy-matrix",
        "description": "Vanilla CoT accuracy per (language x category)",
        "source_file": os.path.abspath(args.file),
        "method_mapping": {
            "correct": "vanilla_cot",
        },
        "breakdown": compute_breakdown_summary(breakdown, ["correct"]),
    }
    save_json_output(json_data, json_output_path)


def cmd_disagree(args):
    """Show cases where steered and baseline disagree."""
    data = load_json(args.file)
    predictions = split_merge(data)

    disagree = [p for p in predictions if p.get(
        "steered_correct") != p.get("baseline_correct")]
    print(f"Total Disagree Cases: {len(disagree)}")

    for item in disagree:
        print(f"ID: {item['id']}")
        print(f"  Steered Correct:  {item.get('steered_correct')}")
        print(f"  Baseline Correct: {item.get('baseline_correct')}")
        print(f"  Question: {item['question']}")
        print(f"  Ground Truth: {item['ground_truth']}")
        print("  --------------------------------")


def cmd_judge_diff(args):
    """Show cases where judge and exact-match disagree."""
    data = load_json(args.file)
    evaluations = data.get("evaluations", [])

    filtered = [
        e for e in evaluations
        if e.get("exact_match_correct") != e.get("judge_correct")
    ]

    print(f"Total differing cases: {len(filtered)}")
    em_only = sum(1 for e in filtered if e.get("exact_match_correct"))
    judge_only = sum(1 for e in filtered if e.get("judge_correct"))
    print(f"  Exact-match correct only: {em_only}")
    print(f"  Judge correct only:       {judge_only}")

    if args.verbose:
        for item in filtered:
            print(f"\nID: {item['id']}")
            print(f"  EM Correct:    {item.get('exact_match_correct')}")
            print(f"  Judge Correct: {item.get('judge_correct')}")
            print(f"  Question:      {item.get('question')}")
            print(f"  Ground Truth:  {item.get('ground_truth')}")


def cmd_steer_evaluation(args):
    """Evaluate steered model predictions.

    This analyses results that have *two* prediction columns: a steered
    prediction (CoT steering) and a baseline prediction (simple prompt).
    """

    data = load_json(args.file)
    evaluations = split_merge_simple(data)

    # classwise accuracy
    # NOTE: If null is present in json, it means that judge was not needed (already correct from exact matching).
    # That is why null_replacement is set to True
    class_breakdown_judge = class_accuracy(
        evaluations,
        baseline_key="baseline_judge_correct",
        steered_key="steered_judge_correct",
        null_replacement=True
    )

    single_breakdown = {k: v for k, v in class_breakdown_judge.items() if v.get("split") == "single"}
    multi_breakdown = {k: v for k, v in class_breakdown_judge.items() if v.get("split") == "multi"}

    # --- Text output (original) ---
    print("single split breakdown:")
    parse_class_breakdown(single_breakdown)
    print("\n" + "=" * 80 + "\n")
    print("multi split breakdown:")
    parse_class_breakdown(multi_breakdown)
    print("\n" + "=" * 80 + "\n")
    print("combined breakdown:")
    parse_class_breakdown(class_breakdown_judge)

    # --- JSON output ---
    json_output_path = getattr(args, "json_output", None)
    if json_output_path is None:
        # Default: save alongside the input file
        json_output_path = os.path.join(
            os.path.dirname(args.file), "summary_steer_evaluation.json"
        )

    count_keys = ["baseline_correct", "steered_correct"]
    json_data = {
        "analysis_type": "steer-evaluation",
        "description": "Steered (CoT steering) vs Baseline (simple prompt) accuracy",
        "source_file": os.path.abspath(args.file),
        "method_mapping": {
            "baseline_correct": "simple_prompt",
            "steered_correct": "cot_steering",
        },
        "breakdowns": {
            "single": compute_breakdown_summary(single_breakdown, count_keys),
            "multi": compute_breakdown_summary(multi_breakdown, count_keys),
            "combined": compute_breakdown_summary(class_breakdown_judge, count_keys),
        },
    }
    save_json_output(json_data, json_output_path)


# =============================================================================
# COMBINE RESULTS
# =============================================================================

def _merge_per_class(steer_combined: Dict, vanilla_breakdown: Dict) -> Dict:
    """
    Merge per-class results from steer-evaluation (CoT steering + simple prompt)
    and class-accuracy-matrix (vanilla CoT) into a unified dict.

    Returns
    -------
    dict keyed by class name, each value:
        {total, vanilla_cot: {correct, accuracy},
                cot_steering: {correct, accuracy},
                simple_prompt: {correct, accuracy}}
    """
    steer_classes = steer_combined.get("per_class", {})
    vanilla_classes = vanilla_breakdown.get("per_class", {})

    all_keys = sorted(set(steer_classes.keys()) | set(vanilla_classes.keys()))
    merged = {}

    for cls in all_keys:
        s = steer_classes.get(cls, {})
        v = vanilla_classes.get(cls, {})
        total = s.get("total", v.get("total", 0))

        merged[cls] = {
            "total": total,
            "split": s.get("split", v.get("split", None)),
            "vanilla_cot": {
                "correct": v.get("correct", None),
                "accuracy": v.get("accuracy", None),
            },
            "cot_steering": {
                "correct": s.get("steered_correct", None),
                "accuracy": s.get("steered_accuracy", None),
            },
            "simple_prompt": {
                "correct": s.get("baseline_correct", None),
                "accuracy": s.get("baseline_accuracy", None),
            },
        }
    return merged


def _merge_aggregates(steer_agg: Dict, vanilla_agg: Dict) -> Dict:
    """Merge language-level, category-level, or grand total dicts."""
    all_keys = sorted(set(steer_agg.keys()) | set(vanilla_agg.keys()))
    merged = {}

    for key in all_keys:
        s = steer_agg.get(key, {})
        v = vanilla_agg.get(key, {})
        total = s.get("total", v.get("total", 0))

        merged[key] = {
            "total": total,
            "vanilla_cot": {
                "correct": v.get("correct", None),
                "accuracy": v.get("accuracy", None),
            },
            "cot_steering": {
                "correct": s.get("steered_correct", None),
                "accuracy": s.get("steered_accuracy", None),
            },
            "simple_prompt": {
                "correct": s.get("baseline_correct", None),
                "accuracy": s.get("baseline_accuracy", None),
            },
        }
    return merged


def _merge_grand_total(steer_gt: Dict, vanilla_gt: Dict) -> Dict:
    """Merge grand total dicts."""
    total = steer_gt.get("total", vanilla_gt.get("total", 0))
    return {
        "total": total,
        "vanilla_cot": {
            "correct": vanilla_gt.get("correct", None),
            "accuracy": vanilla_gt.get("accuracy", None),
        },
        "cot_steering": {
            "correct": steer_gt.get("steered_correct", None),
            "accuracy": steer_gt.get("steered_accuracy", None),
        },
        "simple_prompt": {
            "correct": steer_gt.get("baseline_correct", None),
            "accuracy": steer_gt.get("baseline_accuracy", None),
        },
    }


def cmd_combine_results(args):
    """
    Merge JSON outputs from steer-evaluation and class-accuracy-matrix
    into a single combined JSON with three methods: vanilla_cot, cot_steering,
    simple_prompt.

    steer-evaluation JSON has breakdowns.combined (baseline=simple_prompt,
    steered=cot_steering).
    class-accuracy-matrix JSON has breakdown (correct=vanilla_cot).
    """
    steer_data = load_json(args.steer_json)
    vanilla_data = load_json(args.vanilla_json)

    # Use the 'combined' breakdown from steer-evaluation (single+multi)
    steer_combined = steer_data.get("breakdowns", {}).get("combined", {})
    vanilla_breakdown = vanilla_data.get("breakdown", {})

    combined = {
        "description": (
            "Merged analysis: vanilla CoT vs CoT steering vs simple prompt. "
            "Per-class, per-language, per-category, and grand totals."
        ),
        "source_files": {
            "steer_evaluation": steer_data.get("source_file"),
            "class_accuracy_matrix": vanilla_data.get("source_file"),
        },
        "methods": ["vanilla_cot", "cot_steering", "simple_prompt"],
        "per_class": _merge_per_class(steer_combined, vanilla_breakdown),
        "per_language": _merge_aggregates(
            steer_combined.get("per_language", {}),
            vanilla_breakdown.get("per_language", {}),
        ),
        "per_category": _merge_aggregates(
            steer_combined.get("per_category", {}),
            vanilla_breakdown.get("per_category", {}),
        ),
        "grand_total": _merge_grand_total(
            steer_combined.get("grand_total", {}),
            vanilla_breakdown.get("grand_total", {}),
        ),
    }

    # Also include per-split breakdowns from steer-evaluation
    for split_name in ("single", "multi"):
        split_steer = steer_data.get("breakdowns", {}).get(split_name, {})
        # Filter vanilla classes to matching split
        vanilla_per_class = vanilla_breakdown.get("per_class", {})
        vanilla_split_classes = {
            k: v for k, v in vanilla_per_class.items()
            if v.get("split") == split_name
        }
        vanilla_split = compute_breakdown_summary(
            # Reconstruct a mini breakdown dict
            {k: {"total": v["total"], "correct": v["correct"],
                  "split": v.get("split")}
             for k, v in vanilla_split_classes.items()},
            ["correct"],
        ) if vanilla_split_classes else {}

        combined[f"per_split_{split_name}"] = {
            "per_class": _merge_per_class(split_steer, vanilla_split),
            "per_language": _merge_aggregates(
                split_steer.get("per_language", {}),
                vanilla_split.get("per_language", {}),
            ),
            "per_category": _merge_aggregates(
                split_steer.get("per_category", {}),
                vanilla_split.get("per_category", {}),
            ),
            "grand_total": _merge_grand_total(
                split_steer.get("grand_total", {}),
                vanilla_split.get("grand_total", {}),
            ),
        }

    save_json_output(combined, args.output)
    print(f"\nCombined results for {len(combined['per_class'])} classes across 3 methods.")


def cmd_combined_summary(args):
    """Display a combined summary table from a combined analysis JSON.

    Each cell shows results for all three methods (simple prompt, vanilla CoT,
    CoT steering) plus the number of examples (n).

    Cell format: SP|CoT|CoTS /n
    """
    data = load_json(args.file)

    methods = data.get("methods", ["vanilla_cot", "cot_steering", "simple_prompt"])
    per_class = data.get("per_class", {})
    per_language = data.get("per_language", {})
    per_category = data.get("per_category", {})
    grand_total = data.get("grand_total", {})

    # Collect languages and categories from per_class keys
    languages = set()
    categories = set()
    matrix_data = {}

    for key, stats in per_class.items():
        try:
            lang, cat = key.split('-', 1)
        except ValueError:
            print(f"Skipping malformed key: {key}", flush=True)
            continue
        languages.add(lang)
        categories.add(cat)
        matrix_data[(cat, lang)] = stats

    sorted_langs = sorted(languages)
    sorted_cats = sorted(categories)

    # --- Helpers ---
    method_labels = {
        "simple_prompt": "SP",
        "vanilla_cot": "CoT",
        "cot_steering": "CoTS",
    }

    def _get_correct(entry: Dict, method: str) -> Optional[int]:
        """Extract correct count for a method from an entry."""
        m = entry.get(method, {})
        if isinstance(m, dict):
            return m.get("correct")
        return None

    def format_cell(entry: Dict) -> str:
        """Format: SP|CoT|CoTS /n"""
        total = entry.get("total", 0)
        if total == 0:
            return "N/A"
        sp = _get_correct(entry, "simple_prompt")
        cot = _get_correct(entry, "vanilla_cot")
        cots = _get_correct(entry, "cot_steering")
        parts = [
            str(sp) if sp is not None else "?",
            str(cot) if cot is not None else "?",
            str(cots) if cots is not None else "?",
        ]
        return f"{parts[0]}|{parts[1]}|{parts[2]} /{total}"

    # --- Display ---
    col_width = 26
    first_col_width = 25

    legend = f"Cell format: {method_labels['simple_prompt']}|{method_labels['vanilla_cot']}|{method_labels['cot_steering']} /n"
    print(legend)
    print(f"  SP   = Simple Prompt")
    print(f"  CoT  = Chain of Thought (vanilla)")
    print(f"  CoTS = Chain of Thought Steering")
    print()

    # Header
    header = f"{'Category':<{first_col_width}}"
    for lang in sorted_langs:
        header += f"{lang:^{col_width}}"
    header += f"{'TOTAL':^{col_width}}"

    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    # Rows
    for cat in sorted_cats:
        prunedcat = cat if len(cat) <= first_col_width else cat[:first_col_width - 3] + "..."
        row_str = f"{prunedcat:<{first_col_width}}"

        for lang in sorted_langs:
            if (cat, lang) in matrix_data:
                cell = format_cell(matrix_data[(cat, lang)])
            else:
                cell = "-"
            row_str += f"{cell:^{col_width}}"

        # Category total
        ct = per_category.get(cat, {})
        row_str += f"{format_cell(ct):^{col_width}}"
        print(row_str)

    print(sep)

    # Footer (language totals)
    footer = f"{'TOTAL':<{first_col_width}}"
    for lang in sorted_langs:
        lt = per_language.get(lang, {})
        footer += f"{format_cell(lt):^{col_width}}"

    # Grand total
    footer += f"{format_cell(grand_total):^{col_width}}"
    print(footer)
    print(sep)


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc analysis of MMCricBench experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # lang-accuracy
    p_lang = subparsers.add_parser(
        "lang-accuracy", help="Per-language accuracy")
    p_lang.add_argument("file", help="Path to results JSON")
    p_lang.add_argument("--judge", action="store_true",
                        help="Results are judge evaluations")
    p_lang.set_defaults(func=cmd_lang_accuracy)

    # class-accuracy
    p_cls = subparsers.add_parser(
        "class-accuracy", help="Per (language × category) accuracy")
    p_cls.add_argument("file", help="Path to steered evaluation results JSON")
    p_cls.set_defaults(func=cmd_class_accuracy)

    # language-class accuracy matrix with totals on right and bottom
    p_cls_matrix = subparsers.add_parser(
        "class-accuracy-matrix", help="Display class accuracy in matrix form")
    p_cls_matrix.add_argument("file", help="Path to steered evaluation results JSON")
    p_cls_matrix.set_defaults(func=cmd_class_accuracy_matrix)

    # disagree
    p_dis = subparsers.add_parser(
        "disagree", help="Steered vs baseline disagreements")
    p_dis.add_argument("file", help="Path to steered evaluation results JSON")
    p_dis.set_defaults(func=cmd_disagree)

    # judge-diff
    p_jd = subparsers.add_parser(
        "judge-diff", help="Judge vs exact-match disagreements")
    p_jd.add_argument("file", help="Path to judge evaluation results JSON")
    p_jd.add_argument("--verbose", action="store_true",
                      help="Print per-sample details")
    p_jd.set_defaults(func=cmd_judge_diff)

    # steer-evaluation
    p_steer = subparsers.add_parser(
        "steer-evaluation", help="Evaluate steered model predictions")
    p_steer.add_argument("file", help="Path to steered model predictions JSON")
    p_steer.add_argument("--json-output", dest="json_output", default=None,
                         help="Path to save structured JSON output (default: alongside input)")
    p_steer.set_defaults(func=cmd_steer_evaluation)

    # class-accuracy-matrix: add --json-output
    p_cls_matrix.add_argument("--json-output", dest="json_output", default=None,
                              help="Path to save structured JSON output (default: alongside input)")

    # combine-results
    p_combine = subparsers.add_parser(
        "combine-results",
        help="Merge JSON outputs from steer-evaluation and class-accuracy-matrix",
    )
    p_combine.add_argument(
        "steer_json",
        help="Path to steer-evaluation JSON (summary_steer_evaluation.json)",
    )
    p_combine.add_argument(
        "vanilla_json",
        help="Path to class-accuracy-matrix JSON (summary_class_accuracy_matrix.json)",
    )
    p_combine.add_argument(
        "-o", "--output",
        default="results/combined_analysis.json",
        help="Path to save the combined JSON (default: results/combined_analysis.json)",
    )
    p_combine.set_defaults(func=cmd_combine_results)

    # combined-summary
    p_csummary = subparsers.add_parser(
        "combined-summary",
        help="Display a summary table from a combined analysis JSON (all 3 methods per cell)",
    )
    p_csummary.add_argument(
        "file",
        help="Path to a combined analysis JSON (e.g. results/km1/combined_analysis.json)",
    )
    p_csummary.set_defaults(func=cmd_combined_summary)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
