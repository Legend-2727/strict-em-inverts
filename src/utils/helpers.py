"""Helper utilities for model inspection and manipulation."""

from typing import List, Dict, Any, Optional
import logging
from typing import List, Tuple
import torch
from torch import nn

import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_named_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Retrieve a list of named modules from a given model.

    Args:
        model: The PyTorch model from which to retrieve named modules.

    Returns:
        A list of tuples containing module names and their corresponding modules.

    Example:
        >>> model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        >>> modules = get_named_modules(model)
        >>> print(modules)
        [('', Sequential(...)), ('0', Linear(...)), ('1', ReLU(...))]
    """
    return list(model.named_modules())


def get_names_of_modules(model: nn.Module) -> List[str]:
    """
    Retrieve a list of module names from a given model.

    Args:
        model: The PyTorch model from which to retrieve module names.

    Returns:
        A list of module names as strings.

    Example:
        >>> model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        >>> names = get_names_of_modules(model)
        >>> print(names)
        ['', '0', '1']
    """
    return [name for name, _ in model.named_modules()]


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer string
    """
    # Convert to lowercase and strip whitespace
    answer = answer.lower().strip()

    # Remove common punctuation
    answer = answer.rstrip('.').rstrip(',')

    return answer


def extract_answer_qwen(response: str) -> str:
    """
    Extract the actual answer from Qwen model response.

    The model might provide explanations along with the answer.
    This function attempts to extract just the answer portion.

    Args:
        response: Full model response

    Returns:
        Extracted answer
    """
    response = str(response or "").strip()
    if not response:
        return ""

    tagged_match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, flags=re.IGNORECASE | re.DOTALL)
    if tagged_match:
        tagged_answer = tagged_match.group(1).strip()
        if tagged_answer:
            return tagged_answer.rstrip('.').strip()

    # Remove trailing punctuation but preserve the content
    response = response.rstrip('.').strip()
    response_lower = response.lower()

    # Look for common answer patterns
    patterns = [
        "final answer:",
        "final answer is",
        "the answer is",
        "answer:",
        "answer is",
        "the answer:",
        "الإجابة:",
        "الجواب:",
    ]

    for pattern in patterns:
        if pattern in response_lower:
            idx = response_lower.find(pattern) + len(pattern)
            answer = response[idx:].strip()
            answer = re.split(r"[\n\r]", answer, maxsplit=1)[0].strip()
            if answer:
                return answer.rstrip('.').strip()

    # Fallback: pick the last non-empty, non-instruction-like line.
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return ""

    skip_prefixes = (
        "the user wants",
        "the question asks",
        "let's",
        "1.",
        "2.",
        "3.",
    )
    for line in reversed(lines):
        candidate = line.lstrip("-* ").strip()
        lowered = candidate.lower()
        if not candidate:
            continue
        if lowered.startswith(skip_prefixes):
            continue
        if "analyze the image" in lowered or "scan the text" in lowered:
            continue
        return candidate.rstrip('.').strip()

    # Last resort: preserve previous behavior.
    return lines[0].rstrip('.').strip()


def extract_answer_smolvlm(response: str) -> str:
    """
    Extract the actual answer from SmolVLM model response.

    The model might provide explanations along with the answer.
    This function attempts to extract just the answer portion.

    Args:
        response: Full model response

    Returns:
        Extracted answer
    """
    # First, extract content after "Assistant:" if present (SmolVLM chat format)
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()

    # Remove trailing punctuation but preserve the content
    response = response.rstrip('.')

    response_lower = response.lower()

    # Look for common answer patterns
    patterns = [
        "the answer is",
        "answer:",
        "answer is",
        "the answer:",
    ]

    for pattern in patterns:
        if pattern in response_lower:
            idx = response_lower.find(pattern) + len(pattern)
            answer = response[idx:].strip()
            # Take just the first line/sentence
            answer = answer.split('\n')[0].split('.')[0].strip()
            return answer

    # If no pattern found, return the whole response (first line if multiline)
    first_line = response.strip().split('\n')[0]
    return first_line


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    """
    Check if prediction exactly matches ground truth after normalization.

    Args:
        prediction: Model's prediction
        ground_truth: Ground truth answer

    Returns:
        True if exact match, False otherwise
    """
    pred_normalized = normalize_answer(prediction)
    gt_normalized = normalize_answer(ground_truth)

    return pred_normalized == gt_normalized


def create_judge_prompt(question: str, ground_truth: str, full_response: str) -> str:
    """
    Create a prompt for the LLM judge to evaluate model predictions.

    Args:
        question: The question asked
        ground_truth: The correct answer
        full_response: The model's predicted answer
    Returns:
        A formatted prompt string for the judge
    """
    return (
        f"You are an impartial judge evaluating the performance of a VLM on cricket scorecard questions.\n\n"
        f"Question: {question}\n"
        f"Ground Truth: {ground_truth}\n"
        f"Model Prediction: {full_response}\n\n"
        f"Task: Compare the Model Prediction to the Ground Truth. "
        f"Determine if the prediction is semantically correct, even if the formatting differs. "
        f"Do not mark incomplete answers as correct. "
        f"Provide a clear reasoning for your judgment. "
        f"Respond with a JSON object containing two fields: 'reasoning' (string) and 'is_correct' (boolean)."
    )


# def create_judge_prompt(question: str, ground_truth: str, full_response: str) -> str:
#     """
#     Create a prompt for the LLM judge to evaluate model predictions.
#     """
#     return (
#         f"You are an expert cricket analyst acting as an impartial judge.\n"
#         f"Your task is to evaluate a VLM's extraction of data from cricket scorecards.\n\n"
        
#         f"### DATA \n"
#         f"Question: {question}\n"
#         f"Ground Truth: {ground_truth}\n"
#         f"Model Prediction: {full_response}\n\n"
        
#         f"### EVALUATION CRITERIA\n"
#         f"1. **Numerical Accuracy:** Numbers (runs, wickets, overs, strike rates) must be exact. 45 is NOT the same as 46.\n"
#         f"2. **Name Variations:** Accept standard cricket abbreviations (e.g., 'V Kohli' == 'Virat Kohli', 'MS Dhoni' == 'Dhoni').\n"
#         f"3. **Formatting:** Ignore stylistic differences (e.g., '10-2-45-1' vs '10-2-45-1' with different spacing) provided the data is correct.\n"
#         f"4. **Completeness:** If the question asks for two items (e.g., 'Bowler and Runs'), the model must provide both.\n\n"
        
#         f"### OUTPUT FORMAT\n"
#         f"Provide your response in raw JSON format only. Do not output Markdown code blocks (```json). "
#         f"The JSON must contain:\n"
#         f"- \"step_by_step_reasoning\": A string explaining the comparison.\n"
#         f"- \"is_correct\": A boolean (true/false).\n\n"
        
#         f"JSON Response:"
#     )


# def create_judge_prompt_components(question: str, ground_truth: str, full_response: str) -> Tuple[str, str]:
#     """
#     Returns (system_prompt, user_message) for the judge.
#     """
#     # 1. The Rules (System Prompt)
#     system_prompt = (
#         "You are an expert cricket analyst acting as an impartial judge. "
#         "Your task is to evaluate the performance of a VLM on cricket scorecard questions.\n\n"
#         "### EVALUATION RULES\n"
#         "1. Numerical Accuracy: Numbers (runs, wickets, overs) must be exact.\n"
#         "2. Name Variations: Accept standard abbreviations (e.g., 'V Kohli' == 'Virat Kohli').\n"
#         "3. Completeness: The model must provide all requested data.\n\n"
#         "### OUTPUT FORMAT\n"
#         "Respond with a valid JSON object ONLY. Do not use Markdown blocks.\n"
#         "{\n"
#         '  "step_by_step_reasoning": "string",\n'
#         '  "is_correct": boolean\n'
#         "}"
#     )

#     # 2. The Data (User Message)
#     user_message = (
#         f"Evaluate this prediction:\n\n"
#         f"Question: {question}\n"
#         f"Ground Truth: {ground_truth}\n"
#         f"Model Prediction: {full_response}"
#     )
    
#     return system_prompt, user_message


# def create_judge_prompt_components(question: str, ground_truth: str, full_response: str) -> Tuple[str, str]:
#     """
#     Returns (system_prompt, user_message) for the judge.
#     """
#     # # 1. The Rules (System Prompt) - Significantly expanded for robustness
#     # system_prompt = (
#     #     "You are an expert cricket analyst and an AI evaluator. "
#     #     "Your task is to determine if a Large-Language Model's (LLM) response aligns with the Ground Truth.\n\n"
        
#     #     "### CONTEXT\n"
#     #     "The 'Ground Truth' is a concise answer (a single number, name, or yes/no).\n"
#     #     "The 'Model Prediction' may be a long, step-by-step Chain-of-Thought explanation.\n\n"
        
#     #     "### EVALUATION CRITERIA\n"
#     #     "1. **Extraction**: You must read the entire Model Prediction to find the **final conclusion**. Ignore the intermediate steps unless they contradict the final answer.\n"
#     #     "2. **Equivalence**: Treat the following as correct matches:\n"
#     #     "   - **Numbers**: '8', '8.0', 'eight', and '8 runs' are equivalent.\n"
#     #     "   - **Booleans**: 'Yes', 'True', 'Correct', and 'It did' are equivalent. 'No', 'False', 'Incorrect' are equivalent.\n"
#     #     "   - **Names**: Standard abbreviations are accepted (e.g., 'V Kohli' == 'Virat Kohli').\n"
#     #     "3. **Golden Rule**: If the logic in the Model Prediction leads to the correct answer, but the formatting is different (e.g., specific sentence structure), it is CORRECT.\n"
#     #     "4. **Fail Conditions**: If the model is ambiguous, hedges (e.g., 'It might be 8'), or arrives at a different final number, it is INCORRECT.\n\n"
        
#     #     "### OUTPUT FORMAT\n"
#     #     "Respond with a valid JSON object ONLY. Do not use Markdown blocks.\n"
#     #     "{\n"
#     #     '  "reasoning": "Briefly explain why the extracted answer matches or misses the ground truth.",\n'
#     #     '  "is_correct": boolean\n'
#     #     "}"
#     # )

#     system_prompt = (
#         "You are an expert cricket analyst and an AI evaluator. "
#         "Your task is to determine if a Large-Language Model's (LLM) response aligns with the Ground Truth.\n\n"
        
#         "### CONTEXT\n"
#         "The 'Ground Truth' is a concise answer derived from cricket scorecards (e.g., a strike rate, player name, or innings number).\n"
#         "The 'Model Prediction' may be a long, step-by-step Chain-of-Thought explanation.\n\n"
        
#         "### EVALUATION CRITERIA\n"
#         "1. **Extraction**: You must read the entire Model Prediction to find the **final conclusion**.\n"
#         "2. **Equivalence Rules** (Apply these strictly):\n"
#         "   - **Decimal Tolerance**: For Strike Rates and Economy Rates, allow a margin of error of +/- 0.1. (e.g., If GT is 142.86, accept 142.8, 142.9, or 142.85).\n" # Added for C2/C3 questions
#         "   - **Innings/Dates**: Treat '1', '1st', 'First', and 'First Innings' as identical.\n" # Added for Categorical questions 
#         "   - **Names**: Accept standard initials and partial matches if unambiguous (e.g., 'S Smith' == 'Steve Smith').\n"
#         "   - **Booleans**: 'Yes/True' and 'No/False' are equivalent.\n"
#         "3. **Golden Rule**: If the logic in the Model Prediction leads to the correct answer, but the formatting is different, it is CORRECT.\n"
#         "4. **Fail Conditions**: If the model is ambiguous, hedges, or arrives at a number outside the tolerance range, it is INCORRECT.\n\n"
        
#         "### OUTPUT FORMAT\n"
#         "Respond with a valid JSON object ONLY.\n"
#         "{\n"
#         '  "reasoning": "Brief explanation of the match/mismatch.",\n'
#         '  "is_correct": boolean\n'
#         "}"
#     )

#     # 2. The Data (User Message) - Structured to separate logic from data
#     user_message = (
#         f"### INPUT DATA\n\n"
#         f"**Question:** {question}\n"
#         f"**Ground Truth:** {ground_truth}\n\n"
#         f"**Model Prediction:**\n{full_response}\n\n"
#         f"--- \n"
#         f"Based on the text above, does the Model Prediction eventually provide the answer '{ground_truth}'?"
#     )
    
#     return system_prompt, user_message


def create_judge_prompt_components(question: str, ground_truth: str, full_response: str) -> Tuple[str, str]:
    """
    Returns (system_prompt, user_message) for the judge, optimized for MMCRICBENCH-3K.
    """
    # system_prompt = (
    #     "You are an expert cricket analyst and an AI evaluator. "
    #     "Your task is to determine if a Large-Language Model's (LLM) response aligns with the Ground Truth based on cricket scorecard data.\n\n"
        
    #     "### CONTEXT\n"
    #     "The 'Ground Truth' is a precise answer derived directly from a database (e.g., '142.86', 'Virat Kohli', '1').\n"
    #     "The 'Model Prediction' may be a verbose Chain-of-Thought explanation.\n\n"
        
    #     "### EVALUATION CRITERIA\n"
    #     "1. **Extraction**: You must read the entire Model Prediction to find the **final conclusion**. Ignore intermediate steps unless they contradict the final result.\n"
    #     "   - **Truncation Policy**: If the response is cut off (ends mid-sentence), check if the **final answer** was clearly stated *before* the cut-off. \n"
    #     "       - *Valid*: 'The answer is 5 because... [cut off]'. (Mark as CORRECT if 5 matches GT).\n"
    #     "       - *Invalid*: 'Calculating the runs, we get... [cut off]'. (Mark as INCORRECT).\n"
    #     "       - *Ambiguous*: 'The strike rate is 14... [cut off]'. (Mark as INCORRECT if GT is 142.8).\n\n"
        
    #     "2. **Strict Completeness (Pass/Fail)**:\n"
    #     "   - **Math Must Be Solved**: If the model sets up a calculation (e.g., '50/40') but fails to output the final number (e.g., '1.25'), it is INCORRECT.\n"
    #     "   - **Lists**: If the Ground Truth contains multiple items (e.g., 'Player A, Player B'), the model MUST list ALL of them. Missing even one name makes the answer INCORRECT. Do not award partial credit.\n"
        
    #     "3. **Equivalence Rules (Allowances)**:\n"
    #     "   - **Floating Point Numbers**: For Strike Rates and Economy Rates, allow a margin of error of +/- 0.2 to account for rounding differences (e.g., If GT is '142.86', accept '142.8', '142.9', or '142.85').\n"
    #     "   - **Innings/Ordinals**: Treat '1', '1st', 'First', and 'First Innings' as identical.\n"
    #     "   - **Cricket Terminology**: Recognize that 'Duck' is equivalent to '0 runs' or '0'.\n"
    #     "   - **Names**: Accept standard initials and partial matches if unambiguous (e.g., 'S Smith' == 'Steve Smith').\n"
    #     "   - **Booleans**: 'Yes/True' and 'No/False' are equivalent.\n"
        
    #     "4. **Fail Conditions**: If the model is ambiguous, hedges (e.g., 'It might be...'), provides a range when a specific number is required, or arrives at a number outside the tolerance range, it is INCORRECT.\n\n"
        
    #     "### OUTPUT FORMAT\n"
    #     "Respond with a valid JSON object ONLY. Do not use Markdown blocks.\n"
    #     "{\n"
    #     '  "reasoning": "Briefly explain why the extracted answer matches or misses the ground truth, specifically noting if items were missing from a list or math was unfinished.",\n'
    #     '  "is_correct": boolean\n'
    #     "}"
    # )

    system_prompt = (
        "You are an expert cricket analyst and an AI evaluator evaluating the 'MMCRICBENCH-3K' dataset. "
        "Your task is to determine if a Large-Language Model's (LLM) response aligns with the Ground Truth based on cricket scorecard data.\n\n"
        
        "### CONTEXT\n"
        "The 'Ground Truth' is a precise answer derived directly from a database via SQL (e.g., '142.86', 'Virat Kohli', '1').\n"
        "The 'Model Prediction' may be a verbose Chain-of-Thought explanation or a direct answer.\n\n"
        
        "### EVALUATION CRITERIA\n"
        "1. **Extraction**: You must read the entire Model Prediction to find the **final conclusion**. Ignore intermediate steps unless they contradict the final result.\n"
        "   - **Truncation Policy**: If the response is cut off, check if the **final answer** was clearly stated *before* the cut-off.\n"
        
        "2. **Strict Completeness**:\n"
        "   - **Math Must Be Solved**: Evaluation of formulas (e.g., '50/40') without a final value is INCORRECT.\n"
        "   - **Lists**: If the Ground Truth is a list of names/items (comma-separated), the model MUST include ALL items. Missing distinct items makes the answer INCORRECT.\n"
        
        "3. **Equivalence Rules (Allowances)**:\n"
        "   - **Numerical Tolerance**: For metrics like Strike Rate, Economy, or Run Rate, allow a margin of error of +/- 0.2 (e.g., GT '142.86' matches '142.9').\n"
        "   - **Formatting**: Treat '4' and '4.0' as identical. Treat '1', '1st', 'First', and 'First Innings' as identical.\n"
        "   - **Cricket Terminology**: 'Duck' == '0 runs'. 'Golden Duck' implies 0 runs off 1 ball (ensure the entity matches the GT).\n"
        "   - **Names**: Accept standard initials, partial matches (Last Name only), and minor transliteration differences (e.g., 'Jadeja' vs 'R. Jadeja') if unambiguous.\n"
        
        "### OUTPUT FORMAT\n"
        "Respond with a valid JSON object ONLY.\n"
        "{\n"
        '  "reasoning": "Brief evaluation of the match, noting any missing list items or math errors.",\n'
        '  "is_correct": boolean\n'
        "}"
    )

    user_message = (
        f"### INPUT DATA\n\n"
        f"**Question:** {question}\n"
        f"**Ground Truth:** {ground_truth}\n\n"
        f"**Model Prediction:**\n{full_response}\n\n"
        f"--- \n"
        f"Based on the text above, does the Model Prediction provide the answer '{ground_truth}'?"
    )
    
    return system_prompt, user_message


def transform_predictions_to_judge_format(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Pure function to transform a list of prediction dictionaries into judge input format.
    Isolating this logic makes it easy to unit test without files.

    Args:
        predictions: List of dictionaries with model predictions
    Returns:
        List of dictionaries formatted for LLM judge input
    """
    judge_dataset = []

    for item in predictions:
        # validation: Ensure required keys exist
        if not all(k in item for k in ('question', 'ground_truth', 'prediction')):
            logging.warning(
                f"Skipping item {item.get('id', 'unknown')}: Missing keys.")
            continue

        judge_prompt = create_judge_prompt(
            question=item['question'],
            ground_truth=item['ground_truth'],
            model_prediction=item['prediction']
        )

        entry = {
            "id": item.get('id'),
            "category": item.get('category'),
            "input_data": {
                "question": item['question'],
                "ground_truth": item['ground_truth'],
                "model_prediction": item['prediction']
            },
            "judge_prompt": judge_prompt
        }
        judge_dataset.append(entry)

    return judge_dataset


def generate_judge_input_file(
    input_filename: str,
    output_filename: str,
    data_key_path: Optional[List[str]] = None
) -> None:
    """
    Orchestrator function: Handles I/O and calls the transformation logic.

    Args:
        input_filename: Path to input JSON.
        output_filename: Path to save the result.
        data_key_path: Optional list of keys to navigate to the predictions list 
                       (e.g., ['splits', 'test_single', 'predictions'] -> data['splits']['test_single']['predictions']).
    """
    if data_key_path is None:
        data_key_path = ['splits', 'test_single', 'predictions']

    try:
        with open(input_filename, 'r') as f:
            data = json.load(f)

        # Navigate to the target list using the key path
        predictions = data
        for key in data_key_path:
            predictions = predictions.get(key, {})

        if not isinstance(predictions, list) or not predictions:
            raise ValueError(
                f"Could not find a list of predictions at path: {data_key_path}")

        # Transform data
        judge_dataset = transform_predictions_to_judge_format(predictions)

        # Write output
        with open(output_filename, 'w') as f:
            json.dump(judge_dataset, f, indent=2)

        logging.info(
            f"Successfully converted {len(judge_dataset)} entries to {output_filename}")

    except FileNotFoundError:
        logging.error(f"Input file not found: {input_filename}")
    except json.JSONDecodeError:
        logging.error(f"Input file is not valid JSON: {input_filename}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
