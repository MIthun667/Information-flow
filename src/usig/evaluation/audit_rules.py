from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from usig.data.normalization.text import normalize_answer

_PARENTHETICAL = re.compile(r"\s*\([^()]*\)\s*")
_SEGMENT_BREAK = re.compile(r"(?:\n+|;+|\s+[•●▪]\s+)")
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_NUMBERED_LINE = re.compile(r"^\s*(\d+)[.)]\s*(.+?)\s*$", re.MULTILINE)
_ABSTENTION_PATTERNS = (
    re.compile(r"\b(?:cannot|can't|could not)\s+be\s+determined\b"),
    re.compile(r"\b(?:not|isn't|is not)\s+(?:provided|stated|specified|mentioned)\b"),
    re.compile(r"\binsufficient\s+(?:information|context)\b"),
    re.compile(r"\bcontext\s+(?:does not|doesn't)\s+(?:say|state|provide|mention)\b"),
    re.compile(r"\bunknown\s+from\s+(?:the\s+)?passage\b"),
)


def token_f1_normalized(left: str, right: str) -> float:
    left_tokens = normalize_answer(left).split()
    right_tokens = normalize_answer(right).split()
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = sum((Counter(left_tokens) & Counter(right_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall)


def containment_diagnostics(response: str, aliases: list[str]) -> dict[str, Any]:
    normalized_response = normalize_answer(response)
    alias_values = [(alias, normalize_answer(alias)) for alias in aliases]
    answer_in_response = [
        alias
        for alias, normalized in alias_values
        if normalized and normalized in normalized_response
    ]
    response_in_answer = [
        alias
        for alias, normalized in alias_values
        if normalized_response and normalized_response in normalized
    ]
    f1_values = [(alias, token_f1_normalized(response, alias)) for alias in aliases]
    best_alias, maximum_f1 = max(f1_values, key=lambda item: item[1])
    return {
        "normalized_exact_match": any(
            normalized_response == normalized for _, normalized in alias_values
        ),
        "answer_containment": bool(answer_in_response),
        "response_containment": bool(response_in_answer),
        "contained_aliases": answer_in_response,
        "containing_aliases": response_in_answer,
        "maximum_token_f1": maximum_f1,
        "best_overlap_alias": best_alias,
    }


def concise_alias_match(response: str, aliases: list[str]) -> dict[str, Any]:
    diagnostics = containment_diagnostics(response, aliases)
    if diagnostics["normalized_exact_match"]:
        return {**diagnostics, "match": True, "rule": "normalized_exact"}
    response_without_parenthetical = _PARENTHETICAL.sub(" ", response)
    if any(
        normalize_answer(response_without_parenthetical) == normalize_answer(alias)
        for alias in aliases
    ):
        return {**diagnostics, "match": True, "rule": "response_parenthetical"}
    for alias in aliases:
        alias_without_parenthetical = _PARENTHETICAL.sub(" ", alias)
        if normalize_answer(response) == normalize_answer(alias_without_parenthetical):
            return {**diagnostics, "match": True, "rule": "alias_parenthetical"}
    response_tokens = normalize_answer(response).split()
    matched = diagnostics["contained_aliases"]
    if matched:
        suffix_matches = [
            alias
            for alias in matched
            if normalize_answer(response).endswith(normalize_answer(alias))
        ]
        shortest_extra = min(
            (
                len(response_tokens) - len(normalize_answer(alias).split())
                for alias in suffix_matches
            ),
            default=math.inf,
        )
        if shortest_extra <= 8:
            return {**diagnostics, "match": True, "rule": "concise_containment"}
    return {**diagnostics, "match": False, "rule": None}


def segment_response(response: str) -> list[str]:
    segments = []
    for segment in _SEGMENT_BREAK.split(response):
        cleaned = _LIST_PREFIX.sub("", segment).strip(" \t,")
        if cleaned:
            segments.append(cleaned)
    return segments or ([response.strip()] if response.strip() else [])


def evaluate_interpretation_segments(
    response: str, interpretations: list[dict[str, Any]]
) -> dict[str, Any]:
    numbered = [
        (int(index) - 1, text.strip())
        for index, text in _NUMBERED_LINE.findall(response)
    ]
    segments = [text for _, text in numbered] if numbered else segment_response(response)
    details = []
    for interpretation_index, interpretation in enumerate(interpretations):
        candidate_segments = (
            [
                text
                for index, text in numbered
                if index == interpretation_index
            ]
            if numbered
            else segments
        )
        segment_scores = [
            concise_alias_match(segment, interpretation["reference_answers"])
            for segment in candidate_segments
        ]
        whole_score = concise_alias_match(response, interpretation["reference_answers"])
        matched_segments = [
            index
            for index, score in enumerate(segment_scores)
            if score["match"]
        ]
        maximum_f1 = max(
            [whole_score["maximum_token_f1"]]
            + [score["maximum_token_f1"] for score in segment_scores],
            default=0.0,
        )
        details.append(
            {
                "interpretation_id": interpretation.get("interpretation_id"),
                "matched": (
                    bool(matched_segments)
                    if numbered
                    else whole_score["match"] or bool(matched_segments)
                ),
                "matched_segment_indices": matched_segments,
                "maximum_token_f1": maximum_f1,
            }
        )
    covered = sum(item["matched"] for item in details)
    return {
        "segments": segments,
        "interpretations": details,
        "covered_interpretations": covered,
        "interpretation_count": len(interpretations),
        "interpretation_coverage": (
            covered / len(interpretations) if interpretations else None
        ),
        "any_interpretation_match": covered > 0,
        "multiple_interpretations_covered": covered > 1,
        "any_interpretation_token_f1": max(
            (item["maximum_token_f1"] for item in details), default=0.0
        ),
    }


def conservative_abstention(response: str) -> dict[str, Any]:
    normalized = normalize_answer(response)
    if normalized in {"", "unanswerable"}:
        return {"abstained": True, "rule": "canonical"}
    for pattern in _ABSTENTION_PATTERNS:
        if pattern.search(normalized):
            return {"abstained": True, "rule": pattern.pattern}
    return {"abstained": False, "rule": None}
