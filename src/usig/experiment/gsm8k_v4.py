from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from usig.experiment.records import canonical_json
from usig.experiment.repair_v3 import read_jsonl, write_json, write_jsonl
from usig.data.pilot_collection import quartile_labels

VERSION = "gsm8k_calibration_v4"
REPETITION_THRESHOLD = 0.50
MAX_LOOP_RATE = 0.05
FINAL_ANSWER = re.compile(
    r"(?:^|\n)Final answer:\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
    re.IGNORECASE,
)


def prepare_manifest(source: Path, output: Path, count: int = 100) -> dict[str, Any]:
    rows = sorted(
        read_jsonl(source),
        key=lambda item: (item["selection_order"], item["example_id"]),
    )[:count]
    if len(rows) != count:
        raise ValueError(f"Expected {count} deterministic GSM8K rows")
    write_jsonl(output, rows)
    return {
        "version": VERSION,
        "sample_count": len(rows),
        "checksum": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def prepare_full_manifest(
    normalized: Path, gate_path: Path, output: Path
) -> dict[str, Any]:
    gate = require_gate(gate_path)
    requested = gate["projected_required_collection_size"]
    rows = sorted(read_jsonl(normalized), key=lambda item: item["example_id"])
    if requested is None or requested > len(rows):
        raise ValueError(
            f"GSM8K full collection unavailable: projected N={requested}, "
            f"available records={len(rows)}; use a larger model calibration"
        )
    lengths = quartile_labels([len(item["question"].split()) for item in rows])
    magnitudes = quartile_labels(
        [abs(float(item["reference_answers"][0].replace(",", ""))) for item in rows]
    )
    source_checksum = hashlib.sha256(normalized.read_bytes()).hexdigest()
    selected = [
        {
            "canonical_record_checksum": hashlib.sha256(
                (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
            ).hexdigest(),
            "dataset": "gsm8k",
            "example_id": record["example_id"],
            "group_id": record["group_id"],
            "sampling_seed": 2026,
            "sampling_stratum": (
                f"question_length_q{length}|answer_magnitude_q{magnitude}"
            ),
            "selection_order": order,
            "source_checksum": source_checksum,
            "source_split": record["split"],
        }
        for order, (record, length, magnitude) in enumerate(
            zip(rows[:requested], lengths[:requested], magnitudes[:requested])
        )
    ]
    write_jsonl(output, selected)
    return {
        "version": VERSION,
        "sample_count": len(selected),
        "calibration_accuracy": gate["exact_accuracy"],
        "checksum": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def repetition_rate(token_ids: list[int]) -> float:
    if not token_ids:
        return 0.0
    return 1.0 - len(set(token_ids)) / len(token_ids)


def repeated_ngram_rate(text: str, n: int = 4) -> float:
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def repeated_sentence_rate(text: str) -> float:
    sentences = [
        normalize
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if (normalize := " ".join(re.findall(r"\w+", sentence.lower())))
    ]
    return 0.0 if not sentences else 1.0 - len(set(sentences)) / len(sentences)


def required_collection_size(accuracy: float) -> int | None:
    if not 0.0 < accuracy < 1.0:
        return None
    return math.ceil(max(100 / accuracy, 100 / (1 - accuracy)))


def _is_correct(prediction: dict[str, Any]) -> bool:
    if "binary_error" in prediction:
        return not bool(prediction["binary_error"])
    metrics = prediction.get("evaluation_metrics", {})
    return bool(
        metrics.get("exact_match", metrics.get("correct", metrics.get("numeric_exact_match", False)))
    )


def response_diagnostic(prediction: dict[str, Any]) -> dict[str, Any]:
    response = prediction["response"]
    marker = list(FINAL_ANSWER.finditer(response))
    last = marker[-1] if marker else None
    ngram_rate = repeated_ngram_rate(response)
    sentence_rate = repeated_sentence_rate(response)
    trailing = response[last.end() :] if last else ""
    correct = _is_correct(prediction)
    return {
        "example_id": prediction["example_id"],
        "eos_reached": prediction.get("stop_reason") == "eos_token",
        "truncated": bool(prediction["token_limit_reached"]),
        "parse_failure": prediction["evaluation_metrics"]["parsing_status"] != "ok",
        "answer_marker_present": last is not None,
        "answer_marker_character_position": last.start() if last else None,
        "answer_marker_relative_position": (
            last.start() / max(1, len(response)) if last else None
        ),
        "final_answer_stop_detected": prediction.get(
            "final_answer_stop_detected", False
        ),
        "generated_token_count": prediction["generated_token_count"],
        "response_character_count": prediction["response_character_count"],
        "repetition_rate": repetition_rate(prediction["generated_token_ids"]),
        "repeated_ngram_rate": ngram_rate,
        "repeated_sentence_rate": sentence_rate,
        "repetitive_loop": max(ngram_rate, sentence_rate) > REPETITION_THRESHOLD,
        "exact_answer_correct": correct,
        "correct_final_answer_before_continued_generation": bool(
            correct and last and trailing.strip()
        ),
    }


def diagnostics(predictions: Path, output: Path) -> dict[str, Any]:
    rows = read_jsonl(predictions)
    details = [response_diagnostic(item) for item in rows]
    count = len(details)
    truncated = sum(item["truncated"] for item in details)
    parse_failures = sum(item["parse_failure"] for item in details)
    markers = sum(item["answer_marker_present"] for item in details)
    eos = sum(item["eos_reached"] for item in details)
    correct = sum(item["exact_answer_correct"] for item in details)
    loops = sum(item["repetitive_loop"] for item in details)
    accuracy = correct / count if count else 0.0
    projected = required_collection_size(accuracy)
    longest = sorted(
        details,
        key=lambda item: (
            item["generated_token_count"],
            item["response_character_count"],
        ),
        reverse=True,
    )[:10]
    report = {
        "version": VERSION,
        "sample_count": count,
        "eos_count": eos,
        "eos_rate": eos / count,
        "truncation_count": truncated,
        "truncation_rate": truncated / count,
        "parse_failure_count": parse_failures,
        "parse_failure_rate": parse_failures / count,
        "answer_marker_count": markers,
        "answer_marker_rate": markers / count,
        "exact_correct_count": correct,
        "exact_accuracy": accuracy,
        "projected_required_collection_size": projected,
        "projected_minority_class_at_required_size": (
            None if projected is None else min(accuracy, 1 - accuracy) * projected
        ),
        "model_capability_limitation": accuracy < 0.05,
        "mean_repetition_rate": sum(
            item["repetition_rate"] for item in details
        )
        / count,
        "mean_repeated_ngram_rate": sum(
            item["repeated_ngram_rate"] for item in details
        ) / count,
        "mean_repeated_sentence_rate": sum(
            item["repeated_sentence_rate"] for item in details
        ) / count,
        "repetitive_loop_count": loops,
        "repetitive_loop_rate": loops / count,
        "repetition_threshold": REPETITION_THRESHOLD,
        "maximum_allowed_loop_rate": MAX_LOOP_RATE,
        "correct_final_answer_before_continued_generation_count": sum(
            item["correct_final_answer_before_continued_generation"]
            for item in details
        ),
        "stop_reason_counts": dict(
            sorted(Counter(item.get("stop_reason", "unknown") for item in rows).items())
        ),
        "longest_responses": longest,
        "maximum_allowed_truncation_rate": 0.05,
        "maximum_allowed_parse_failure_rate": 0.05,
        "passed": (
            count == 100
            and truncated / count <= 0.05
            and parse_failures / count <= 0.05
            and loops / count <= MAX_LOOP_RATE
            and projected is not None
            and accuracy >= 0.05
        ),
    }
    write_json(output, report)
    return report


def require_gate(gate_path: Path) -> dict[str, Any]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("version") != VERSION or not gate.get("passed"):
        raise PermissionError("GSM8K full collection refused: Version 4 gate did not pass")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--source", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    full = sub.add_parser("full-manifest")
    full.add_argument("--normalized", type=Path, required=True)
    full.add_argument("--gate", type=Path, required=True)
    full.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("diagnostics")
    report.add_argument("--predictions", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    gate = sub.add_parser("require-gate")
    gate.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "manifest":
        result = prepare_manifest(args.source, args.output)
    elif args.action == "full-manifest":
        result = prepare_full_manifest(args.normalized, args.gate, args.output)
    elif args.action == "diagnostics":
        result = diagnostics(args.predictions, args.output)
    else:
        result = require_gate(args.gate)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
