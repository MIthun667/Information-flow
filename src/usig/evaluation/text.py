from __future__ import annotations

from collections import Counter
from typing import Any

from usig.data.normalization.text import normalize_answer


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_aliases(prediction: str, aliases: list[str]) -> dict[str, Any]:
    if not aliases:
        raise ValueError("At least one answer alias is required")
    normalized_prediction = normalize_answer(prediction)
    exact = [normalized_prediction == normalize_answer(alias) for alias in aliases]
    return {
        "normalized_prediction": normalized_prediction,
        "exact_match": any(exact),
        "maximum_token_f1": max(token_f1(prediction, alias) for alias in aliases),
        "matched_aliases": [alias for alias, matched in zip(aliases, exact) if matched],
    }


def evaluate_truthfulqa(
    prediction: str, correct_answers: list[str], incorrect_answers: list[str]
) -> dict[str, Any]:
    correct = evaluate_aliases(prediction, correct_answers)
    incorrect = evaluate_aliases(prediction, incorrect_answers)
    if correct["exact_match"] and not incorrect["exact_match"]:
        status = "matched_correct_reference"
    elif incorrect["exact_match"] and not correct["exact_match"]:
        status = "matched_incorrect_reference"
    elif correct["exact_match"] and incorrect["exact_match"]:
        status = "matched_both_reference_sets"
    else:
        status = "unmatched"
    return {
        "status": status,
        "correct_reference_match": correct["exact_match"],
        "incorrect_reference_match": incorrect["exact_match"],
        "correct_reference_token_f1": correct["maximum_token_f1"],
        "incorrect_reference_token_f1": incorrect["maximum_token_f1"],
        "limitation": (
            "Lexical reference matching is diagnostic only and is not a complete "
            "truthfulness or informativeness evaluator."
        ),
    }


def evaluate_ambignq(prediction: str, interpretations: list[dict[str, Any]]) -> dict[str, Any]:
    if not interpretations:
        raise ValueError("AmbigNQ evaluation requires interpretation annotations")
    scores = [
        evaluate_aliases(prediction, interpretation["reference_answers"])
        for interpretation in interpretations
    ]
    matched = sum(score["exact_match"] for score in scores)
    return {
        "any_interpretation_exact_match": matched > 0,
        "interpretation_recall": matched / len(scores),
        "interpretation_aware_token_f1": sum(
            score["maximum_token_f1"] for score in scores
        )
        / len(scores),
        "matched_interpretations": matched,
        "interpretation_count": len(scores),
    }


def evaluate_squad(
    prediction: str, references: list[str], *, answerable: bool
) -> dict[str, Any]:
    normalized = normalize_answer(prediction)
    abstained = normalized in {"", "unanswerable"}
    if not answerable:
        return {
            "answerable": False,
            "no_answer_accuracy": float(abstained),
            "exact_match": float(abstained),
            "token_f1": float(abstained),
            "combined_correct": abstained,
        }
    result = evaluate_aliases(prediction, references)
    return {
        "answerable": True,
        "no_answer_accuracy": None,
        "exact_match": float(result["exact_match"]),
        "token_f1": result["maximum_token_f1"],
        "combined_correct": result["exact_match"],
    }
