"""Utilities for compact evidence-grounded outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

EVIDENCE_OPEN_TAG = "<evidence>"
EVIDENCE_CLOSE_TAG = "</evidence>"
ANSWER_OPEN_TAG = "<answer>"
ANSWER_CLOSE_TAG = "</answer>"

_FIELD_PATTERNS = [
    "strike rate",
    "economy rate",
    "boundary percentage",
    "run rate",
    "golden duck",
    "duck",
    "century",
    "wickets",
    "maidens",
    "extras",
    "leg byes",
    "byes",
    "no-balls",
    "no balls",
    "wides",
    "boundaries",
    "sixes",
    "fours",
    "catches",
]


def _extract_tagged_span(text: str, open_tag: str, close_tag: str) -> Optional[str]:
    if not text:
        return None
    pattern = re.escape(open_tag) + r"(.*?)" + re.escape(close_tag)
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def extract_evidence_text(text: str) -> Optional[str]:
    """Return the raw evidence payload between evidence tags, if present."""
    return _extract_tagged_span(text, EVIDENCE_OPEN_TAG, EVIDENCE_CLOSE_TAG)


def extract_answer_text(text: str) -> Optional[str]:
    """Return the raw answer payload between answer tags, if present."""
    return _extract_tagged_span(text, ANSWER_OPEN_TAG, ANSWER_CLOSE_TAG)


def parse_evidence_items(text: str) -> Dict[str, Any]:
    """
    Parse compact evidence items from a model response.

    Returns a dict with:
    - evidence_text
    - parsed_evidence
    - evidence_parse_ok
    - answer_text
    """
    evidence_text = extract_evidence_text(text)
    answer_text = extract_answer_text(text)

    if evidence_text is None:
        return {
            "evidence_text": None,
            "parsed_evidence": None,
            "evidence_parse_ok": False,
            "answer_text": answer_text,
        }

    try:
        parsed = json.loads(evidence_text)
    except json.JSONDecodeError:
        return {
            "evidence_text": evidence_text,
            "parsed_evidence": None,
            "evidence_parse_ok": False,
            "answer_text": answer_text,
        }

    if not isinstance(parsed, list):
        return {
            "evidence_text": evidence_text,
            "parsed_evidence": None,
            "evidence_parse_ok": False,
            "answer_text": answer_text,
        }

    normalized: List[Dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            return {
                "evidence_text": evidence_text,
                "parsed_evidence": None,
                "evidence_parse_ok": False,
                "answer_text": answer_text,
            }
        normalized.append(
            {
                "entity": str(item.get("entity", "")).strip(),
                "field": str(item.get("field", "")).strip(),
                "value": str(item.get("value", "")).strip(),
            }
        )

    return {
        "evidence_text": evidence_text,
        "parsed_evidence": normalized,
        "evidence_parse_ok": True,
        "answer_text": answer_text,
    }


def format_evidence_answer(evidence_items: List[Dict[str, Any]], answer: str) -> str:
    """Format a compact evidence + answer target for PEG training or decoding."""
    safe_items = [
        {
            "entity": str(item.get("entity", "")).strip(),
            "field": str(item.get("field", "")).strip(),
            "value": str(item.get("value", "")).strip(),
        }
        for item in evidence_items
    ]
    evidence_json = json.dumps(safe_items, ensure_ascii=False, separators=(",", ":"))
    return (
        f"{EVIDENCE_OPEN_TAG}{evidence_json}{EVIDENCE_CLOSE_TAG}\n"
        f"{ANSWER_OPEN_TAG}{str(answer).strip()}{ANSWER_CLOSE_TAG}"
    )


def infer_field_from_question(question: str) -> str:
    """Heuristic field extraction for pseudo-evidence supervision."""
    lowered = question.lower()
    for field in _FIELD_PATTERNS:
        if field in lowered:
            return field
    if "difference" in lowered:
        return "difference"
    if "sum" in lowered or "total" in lowered:
        return "total"
    if "count" in lowered or "how many" in lowered:
        return "count"
    if lowered.startswith("who"):
        return "entity"
    if lowered.startswith("which"):
        return "selection"
    if lowered.startswith("did") or lowered.startswith("has") or lowered.startswith("does"):
        return "boolean"
    return "value"


def infer_evidence_items(question: str, answer: str, max_items: int = 2) -> List[Dict[str, str]]:
    """
    Build compact pseudo-evidence targets from question + answer.

    This is intentionally heuristic and generic. It creates structured,
    question-conditioned supervision without hardcoding cricket formulas.
    """
    question = str(question).strip()
    answer = str(answer).strip()
    field = infer_field_from_question(question)

    possessive_match = re.search(r"what is\s+(.+?)'s\s+([a-zA-Z -]+)\??$", question, re.IGNORECASE)
    if possessive_match:
        entity = possessive_match.group(1).strip()
        return [{"entity": entity, "field": possessive_match.group(2).strip(), "value": answer}]

    compare_match = re.search(
        r"^(?:has|did|does)\s+(.+?)\s+(?:hit|taken|take|score|scored|concede|conceded|bowl|bowled|have)\s+.+?\s+than\s+(.+?)\?$",
        question,
        re.IGNORECASE,
    )
    if compare_match:
        return [
            {"entity": compare_match.group(1).strip(), "field": field, "value": answer},
            {"entity": compare_match.group(2).strip(), "field": field, "value": answer},
        ][:max_items]

    who_match = re.search(r"who\s+(.+?)\?$", question, re.IGNORECASE)
    if who_match:
        return [{"entity": "match", "field": who_match.group(1).strip(), "value": answer}]

    which_match = re.search(r"which\s+(.+?)\?$", question, re.IGNORECASE)
    if which_match:
        return [{"entity": "match", "field": which_match.group(1).strip(), "value": answer}]

    return [{"entity": "match", "field": field, "value": answer}]
