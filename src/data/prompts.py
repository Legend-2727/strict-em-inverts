"""
Centralized prompt templates shared across datasets and experiments.

All positive (CoT / reasoning) and negative (direct-answer) templates live here
to avoid prompt drift across scripts.
"""

# =============================================================================
# POSITIVE (REASONING / COT) PROMPT TEMPLATES
# =============================================================================

POSITIVE_PROMPTS = {
    "translate-first": (
        "Think step by step. First, identify the relevant Hindi text in the image "
        "and translate it into English to understand the column headers and row entities. "
        "Then, extract the specific numerical values associated with these translated entities. "
        "Finally, perform the necessary calculations to answer the question.\n\n"
        "Question: {question}"
    ),
    "structural-anchor": (
        "Let's analyze the scorecard structure step by step. Locate the row corresponding "
        "to the player or team mentioned in the question. Map the Hindi column headers to "
        "their English statistical equivalents (e.g., 'Overs', 'Runs', 'Wickets'). "
        "Extract the data points at the intersection of these rows and columns, and use "
        "them to derive the final answer.\n\n"
        "Question: {question}"
    ),
    "verification": (
        "Think step by step to ensure accuracy. "
        "1. Transcribe the relevant numbers and Hindi text from the image. "
        "2. Translate the Hindi terms to English to confirm the context (batting vs. bowling). "
        "3. Set up the arithmetic equation required by the question. "
        "4. Calculate the result and output the final answer.\n\n"
        "Question: {question}"
    ),
    "simple-cot": (
        "Think step by step. If the table is in Hindi, translate to English, "
        "then extract the relevant data and find the answer. If necessary, compute step by step.\n\n"
        "Question: {question}"
    ),
    "simple-cot-2": (
        "Question: {question}\n\n"
        "Think step by step."
    ),
    "simple-cot-3": (
        "Think step by step.\n\n"
        "Question: {question}"
    ),
    "translate-cot-1": (
        "Think step by step. If necessary, extract the data from the context, translate the key terms and numbers into English, and then answer."
        "Question: {question}\n\n"
    ),
    "translate-cot-2": (
        "Think step by step. Extract the data from the context, translate the key terms and numbers into English, and then answer."
        "Question: {question}\n\n"
    ),
    "translate-cot-3": (
        "Think step by step. Extract the data from the Hindi question, translate the key terms and numbers into English, and then answer."
        "Question: {question}\n\n"
    ),
    "translate-cot-4": (
        "Think step by step. Extract the data from the context, translate the key terms and numbers into English if applicable, and then answer.\n\n"
        "Question: {question}"
    ),
    "gemini-001": (
        "**Role:**\n\n"
        "You are an expert Cricket Statistician and Data Analyst. You are provided with cricket scorecard images (in English or Hindi) and must answer questions requiring retrieval, calculation, and multi-step reasoning.\n\n"
        "**Critical Domain Context & Formulas:**\n\n"
        "Use these definitions strictly for your calculations:\n\n"
        "- **Strike Rate (Batting):** (Runs Scored / Balls Faced) * 100\n"
        "- **Economy Rate (Bowling):** Runs Conceded / Overs Bowled\n"
        "- **Duck:** A batsman dismissed for exactly 0 runs.\n"
        "- **Golden Duck:** A batsman dismissed for 0 runs having faced exactly 1 ball.\n"
        "- **Century:** A batsman scoring 100 or more runs.\n"
        "- **n-fer (e.g., 4-fer):** A bowler taking n or more wickets in a single innings.\n"
        "- **Maiden:** An over where 0 runs are conceded.\n\n"
        "**Language Translation Guide (Hindi to English):**\n\n"
        "If the table is in Hindi, map the headers as follows before extracting data:\n\n"
        "- रन -> Runs\n"
        "- गेंदें / बॉल -> Balls\n"
        "- विकेट -> Wickets\n"
        "- ओवर -> Overs\n"
        "- इकोनॉमी -> Economy Rate\n"
        "- स्ट्राइक रेट -> Strike Rate\n"
        "- शून्य / 0 -> Duck\n\n"
        "**Reasoning Protocol:**\n\n"
        "You must output your thought process in the following step-by-step format before the final answer:\n\n"
        "1. **Language & Structure Scan:**\n"
        "   - Is the scorecard in English or Hindi?\n"
        "   - Identify the table headers. (If Hindi, explicitly map them to English keywords provided above).\n"
        "   - Self-Correction: Watch out for visually similar digits or text artifacts.\n\n"
        "2. **Target Retrieval:**\n"
        "   - Locate the specific row for the player/team mentioned in the query.\n"
        "   - Extract the raw values needed (e.g., \"Virat's Runs = 94, Balls = 50\").\n"
        "   - Constraint Check: Are we looking at the 1st Innings or 2nd Innings? (Check image titles).\n\n"
        "3. **Calculation & Logic:**\n"
        "   - State the formula you are using (e.g., \"Calculating Strike Rate\").\n"
        "   - Plug in the extracted numbers.\n"
        "   - Perform the arithmetic step-by-step.\n"
        "   - Rounding: Round to 2 decimal places unless specified otherwise.\n\n"
        "4. **Final Verification:**\n"
        "   - Does the answer match the question type? (e.g., If asking \"Who\", provide a name. If \"How many\", provide a count).\n\n"
        "**Final Output Format:**\n\n"
        "Provide the final answer precisely in 1-2 words or digits. Do not include verbose sentences in the final line.\n\n"
        "**Example Output:**\n\n"
        "Reasoning:\n"
        "1. Image is Hindi. Mapped \"स्ट्राइक रेट\" to Strike Rate.\n"
        "2. Located row for \"Rohit Sharma\". Extracted Runs: 45, Balls: 30.\n"
        "3. Formula: (45 / 30) * 100 = 150.00.\n"
        "4. Verified against question \"What is Rohit's strike rate?\".\n\n"
        "Final Answer: 150.00\n\n"
        "Question: {question}"
    ),
}

# =============================================================================
# NEGATIVE / DIRECT-ANSWER PROMPT TEMPLATE
# =============================================================================

NEGATIVE_PROMPT_TEMPLATE = (
    "Answer precisely in 1-2 words, answer in digits when required.\n\n"
    "Question: {question}"
)

# Alias used by the evaluation pipeline (identical content)
DIRECT_ANSWER_PROMPT = NEGATIVE_PROMPT_TEMPLATE


# =============================================================================
# MTVQA DIRECT-ANSWER PROTOCOL PROMPT
# =============================================================================

MTVQA_DIRECT_PROMPT = (
    "Answer using only information visible in the image.\n"
    "Return exactly one line in the language of the question:\n"
    "<answer>YOUR_ANSWER</answer>\n"
    "Do not include analysis, reasoning, bullets, or any extra text.\n\n"
    "Question: {question}"
)


EVIDENCE_SCHEMA_DESCRIPTION = (
    "Return compact machine-readable evidence for the semi-structured document. "
    "Use a JSON list inside <evidence> tags where each item has exactly these "
    "string fields: entity, field, value. Keep evidence short and grounded in "
    "the document. Then return the final answer inside <answer> tags."
)


EVIDENCE_PROMPT_TEMPLATE = (
    f"{EVIDENCE_SCHEMA_DESCRIPTION}\n\n"
    "Output exactly in this format:\n"
    "<evidence>[{{\"entity\":\"...\",\"field\":\"...\",\"value\":\"...\"}}]</evidence>\n"
    "<answer>...</answer>\n\n"
    "Question: {question}"
)


def build_direct_answer_prompt(question: str) -> str:
    """Format the shared direct-answer prompt."""
    return DIRECT_ANSWER_PROMPT.format(question=question)


def build_mtvqa_direct_prompt(question: str) -> str:
    """Format the MTVQA protocol-locked direct-answer prompt."""
    return MTVQA_DIRECT_PROMPT.format(question=question)


def build_evidence_prompt(question: str) -> str:
    """Format the shared evidence-grounding prompt."""
    return EVIDENCE_PROMPT_TEMPLATE.format(question=question)
