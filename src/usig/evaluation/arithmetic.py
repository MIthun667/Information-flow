from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

_NUMBER = re.compile(r"(?<![\w/])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w/])")
_FRACTION = re.compile(r"[-+]?\d+\s*/\s*\d+")
_FINAL_ANSWER = re.compile(
    r"(?:####|final\s+answer\s*(?:is|:)?|answer\s*(?:is|:)?)\s*"
    r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
    re.IGNORECASE,
)
_BOXED_ANSWER = re.compile(
    r"\\boxed\{\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*\}"
)


def extract_integer_answer(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        return {
            "parsed_answer": None,
            "normalized_answer": None,
            "parsing_status": "error",
            "error_reason": "empty_or_nonnumeric_response",
        }
    if _FRACTION.search(text):
        return {
            "parsed_answer": None,
            "normalized_answer": None,
            "parsing_status": "error",
            "error_reason": "fraction_not_allowed",
        }
    explicit_final_answers = _FINAL_ANSWER.findall(text)
    matches = (
        [explicit_final_answers[-1]]
        if explicit_final_answers
        else _NUMBER.findall(text)
    )
    if not matches:
        return {
            "parsed_answer": None,
            "normalized_answer": None,
            "parsing_status": "error",
            "error_reason": "no_numeric_answer",
        }
    if len(matches) != 1:
        return {
            "parsed_answer": None,
            "normalized_answer": None,
            "parsing_status": "error",
            "error_reason": "ambiguous_multiple_numbers",
        }
    parsed = matches[0]
    try:
        value = Decimal(parsed.replace(",", ""))
    except InvalidOperation:
        return {
            "parsed_answer": parsed,
            "normalized_answer": None,
            "parsing_status": "error",
            "error_reason": "malformed_number",
        }
    if not value.is_finite() or value != value.to_integral_value():
        return {
            "parsed_answer": parsed,
            "normalized_answer": None,
            "parsing_status": "error",
            "error_reason": "non_integer_answer",
        }
    normalized = str(int(value))
    return {
        "parsed_answer": parsed,
        "normalized_answer": normalized,
        "parsing_status": "ok",
        "error_reason": None,
    }


def extract_final_integer_answer(text: str) -> dict[str, Any]:
    """Parse the final numeric span, while retaining strict integer validation."""
    if not isinstance(text, str) or not text.strip():
        return extract_integer_answer(text)
    candidates = (
        _BOXED_ANSWER.findall(text)
        or _FINAL_ANSWER.findall(text)
        or _NUMBER.findall(text)
    )
    if not candidates:
        return extract_integer_answer(text)
    return extract_integer_answer(candidates[-1])


def evaluate_arithmetic_answer(
    prediction: str, reference_answer: str, *, final_answer: bool = False
) -> dict[str, Any]:
    extracted = (
        extract_final_integer_answer(prediction)
        if final_answer
        else extract_integer_answer(prediction)
    )
    try:
        reference = Decimal(str(reference_answer).replace(",", "").strip())
    except InvalidOperation as error:
        raise ValueError(f"Invalid integer reference answer: {reference_answer!r}") from error
    if not reference.is_finite() or reference != reference.to_integral_value():
        raise ValueError(f"Invalid integer reference answer: {reference_answer!r}")
    result = {
        **extracted,
        "reference_answer": str(int(reference)),
        "exact_match": False,
    }
    if extracted["parsing_status"] == "ok":
        result["exact_match"] = extracted["normalized_answer"] == result["reference_answer"]
    return result
