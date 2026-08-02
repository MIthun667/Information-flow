"""Prepare diverse IFI-ARITH candidates for reasoning-intervention screening."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATHS = {
    "larger_integer": (
        PROJECT_ROOT / "data/normalized/ifi_arith/larger_integer.jsonl"
    ),
    "moderate_multiplicative": (
        PROJECT_ROOT
        / "data/normalized/ifi_arith/moderate_multiplicative.jsonl"
    ),
}

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/uncertainty_flow/pilot_v1/curated/reasoning_groups.jsonl"
)

DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "outputs/uncertainty_flow/pilot_v1/audits/"
    "reasoning_candidate_audit.json"
)

# Explicit quotas prevent one source domain from dominating the pilot.
SELECTION_QUOTAS: tuple[tuple[str, str, int], ...] = (
    ("larger_integer", "addition", 2),
    ("larger_integer", "subtraction", 2),
    ("larger_integer", "division", 1),
    ("moderate_multiplicative", "multiplication", 3),
    ("moderate_multiplicative", "division", 2),
)

EXPECTED_TOTAL = sum(count for _, _, count in SELECTION_QUOTAS)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object"
                )

            validate_source_record(payload)
            records.append(payload)

    return records


def validate_source_record(record: Mapping[str, Any]) -> None:
    required = {
        "example_id",
        "domain",
        "question",
        "reference_answers",
        "metadata",
    }
    missing = sorted(required.difference(record))

    if missing:
        raise ValueError(
            "source record is missing fields: " + ", ".join(missing)
        )

    metadata = record["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("source metadata must be a mapping")

    required_metadata = {
        "operand_a",
        "operand_b",
        "operation",
        "expression",
        "expected_answer",
    }
    missing_metadata = sorted(required_metadata.difference(metadata))

    if missing_metadata:
        raise ValueError(
            "source metadata is missing fields: "
            + ", ".join(missing_metadata)
        )

    answers = record["reference_answers"]
    if not isinstance(answers, list) or not answers:
        raise ValueError("source record must provide reference_answers")


def count_digits(value: int) -> int:
    return len(str(abs(value)))


def repeated_digit_ratio(value: int) -> float:
    digits = str(abs(value))
    if not digits:
        return 0.0

    most_common = Counter(digits).most_common(1)[0][1]
    return most_common / len(digits)


def triviality_reasons(record: Mapping[str, Any]) -> list[str]:
    """Identify arithmetic patterns unsuitable for uncertainty screening."""

    metadata = record["metadata"]
    operation = str(metadata["operation"])
    operand_a = abs(int(metadata["operand_a"]))
    operand_b = abs(int(metadata["operand_b"]))
    answer = abs(int(metadata["expected_answer"]))

    reasons: list[str] = []

    convenient_values = {
        0,
        1,
        2,
        5,
        10,
        20,
        25,
        50,
        100,
        200,
        500,
        999,
        1000,
        9999,
        10000,
    }

    if operation == "multiplication":
        if operand_a in convenient_values or operand_b in convenient_values:
            reasons.append("convenient_multiplier")

        if operand_a % 10 == 0 or operand_b % 10 == 0:
            reasons.append("round_multiplier")

    if operation == "subtraction":
        if answer < 100:
            reasons.append("small_difference")

        if operand_a - operand_b in {1, 10, 100, 1000}:
            reasons.append("obvious_difference")

    if operation == "division":
        if operand_b in convenient_values:
            reasons.append("convenient_divisor")

        if operand_a % 1000 == 0:
            reasons.append("round_dividend")

        quotient = operand_a // operand_b if operand_b else 0
        if quotient in convenient_values:
            reasons.append("convenient_quotient")

    if operation == "addition":
        if operand_a % 1000 == 0 or operand_b % 1000 == 0:
            reasons.append("round_addend")

        if answer % 1000 == 0:
            reasons.append("round_sum")

    return sorted(set(reasons))


def difficulty_score(record: Mapping[str, Any]) -> tuple[int, int, int, str]:
    """Rank candidates using digit demand and nontrivial digit structure."""

    metadata = record["metadata"]
    operand_a = abs(int(metadata["operand_a"]))
    operand_b = abs(int(metadata["operand_b"]))
    answer = abs(int(metadata["expected_answer"]))

    total_digits = (
        count_digits(operand_a)
        + count_digits(operand_b)
        + count_digits(answer)
    )

    nonzero_digits = sum(
        digit != "0"
        for digit in f"{operand_a}{operand_b}{answer}"
    )

    digit_diversity = (
        len(set(str(operand_a)))
        + len(set(str(operand_b)))
        + len(set(str(answer)))
    )

    return (
        -total_digits,
        -nonzero_digits,
        -digit_diversity,
        str(record["example_id"]),
    )


def reasoning_scaffold(record: Mapping[str, Any]) -> str:
    """Construct an answer-free, instance-aware reasoning intervention."""

    metadata = record["metadata"]
    operation = str(metadata["operation"])
    operand_a = int(metadata["operand_a"])
    operand_b = int(metadata["operand_b"])

    if operation == "addition":
        return (
            f"Split {operand_a} and {operand_b} into thousands, hundreds, "
            "tens, and ones. Add matching place values from right to left, "
            "record every carry, and then combine the resulting place values."
        )

    if operation == "subtraction":
        return (
            f"Align {operand_a} and {operand_b} by place value. Subtract from "
            "right to left, explicitly borrowing whenever the upper digit is "
            "smaller. Verify the result by adding it to the subtracted value."
        )

    if operation == "multiplication":
        return (
            f"Write {operand_b} as hundreds plus tens plus ones. Multiply "
            f"{operand_a} by each nonzero place-value component separately, "
            "shift each partial product correctly, and add the partial products."
        )

    if operation == "division":
        return (
            f"Estimate how many times {operand_b} fits into {operand_a} using "
            "the leading digits. Multiply the candidate quotient by the divisor "
            "and adjust it until the product exactly matches the dividend."
        )

    raise ValueError(f"unsupported operation: {operation}")


def irrelevant_instruction(operation: str) -> str:
    controls = {
        "addition": (
            "Addition is represented by a plus sign in modern mathematical "
            "notation. Read the full expression before returning an answer."
        ),
        "subtraction": (
            "Subtraction is represented by a minus sign in modern mathematical "
            "notation. Read the full expression before returning an answer."
        ),
        "multiplication": (
            "Multiplication can be represented by a cross, dot, or adjacency. "
            "Read the full expression before returning an answer."
        ),
        "division": (
            "Division can be represented by a slash or division symbol. "
            "Read the full expression before returning an answer."
        ),
    }

    try:
        return controls[operation]
    except KeyError as error:
        raise ValueError(f"unsupported operation: {operation}") from error


def select_candidates(
    records_by_domain: Mapping[str, Iterable[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select quota-constrained candidates with globally unique answers."""

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rejection_counts: Counter[str] = Counter()
    eligible_counts: Counter[str] = Counter()

    for expected_domain, records in records_by_domain.items():
        for record in records:
            domain = str(record["domain"])
            operation = str(record["metadata"]["operation"])

            if domain != expected_domain:
                raise ValueError(
                    f"record domain {domain!r} does not match source "
                    f"{expected_domain!r}"
                )

            reasons = triviality_reasons(record)
            if reasons:
                rejection_counts.update(reasons)
                continue

            buckets[(domain, operation)].append(record)
            eligible_counts[f"{domain}:{operation}"] += 1

    selected: list[dict[str, Any]] = []
    used_answers: set[str] = set()
    duplicate_answer_skips: Counter[str] = Counter()

    for domain, operation, required_count in SELECTION_QUOTAS:
        candidates = sorted(
            buckets[(domain, operation)],
            key=difficulty_score,
        )

        quota_selected: list[dict[str, Any]] = []

        for candidate in candidates:
            answer = str(candidate["metadata"]["expected_answer"])

            if answer in used_answers:
                duplicate_answer_skips[f"{domain}:{operation}"] += 1
                continue

            quota_selected.append(candidate)
            used_answers.add(answer)

            if len(quota_selected) == required_count:
                break

        if len(quota_selected) < required_count:
            raise ValueError(
                f"{domain}:{operation} produced only "
                f"{len(quota_selected)} unique-answer candidates after "
                f"filtering; {required_count} are required"
            )

        selected.extend(quota_selected)

    selected = sorted(
        selected,
        key=lambda record: (
            str(record["domain"]),
            str(record["metadata"]["operation"]),
            str(record["example_id"]),
        ),
    )

    answers = [
        str(record["metadata"]["expected_answer"])
        for record in selected
    ]

    if len(answers) != len(set(answers)):
        raise RuntimeError(
            "internal error: selected candidates contain duplicate answers"
        )

    audit = {
        "eligible_counts": dict(sorted(eligible_counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "duplicate_answer_skips": dict(
            sorted(duplicate_answer_skips.items())
        ),
        "selection_quotas": [
            {
                "domain": domain,
                "operation": operation,
                "count": count,
            }
            for domain, operation, count in SELECTION_QUOTAS
        ],
    }

    return selected, audit

def transform_record(
    record: Mapping[str, Any],
    *,
    group_index: int,
) -> dict[str, Any]:
    metadata = record["metadata"]
    operation = str(metadata["operation"])
    question = str(record["question"])
    group_id = f"reasoning_{group_index:04d}"

    instruction = (
        "Solve the arithmetic problem. "
        "Do not show intermediate work. "
        "Output exactly: FINAL: <number>"
    )

    original_prompt = (
        f"{instruction}\n"
        f"{question} ="
    )
    resolved_prompt = (
        f"{instruction}\n"
        "Use the following strategy internally before answering:\n"
        f"{reasoning_scaffold(record)}\n"
        f"{question} ="
    )
    control_prompt = (
        f"{instruction}\n"
        f"Additional information: {irrelevant_instruction(operation)}\n"
        f"{question} ="
    )

    return {
        "group_id": group_id,
        "base_id": str(record["example_id"]),
        "dataset_name": "ifi_arith",
        "source": "reasoning",
        "gold_answers": [
            str(answer) for answer in record["reference_answers"]
        ],
        "original": {
            "prompt": original_prompt,
            "optimal_action": "reason_more",
            "evidence": None,
            "clarification": None,
        },
        "resolved": {
            "prompt": resolved_prompt,
            "optimal_action": "answer",
            "evidence": None,
            "clarification": None,
        },
        "irrelevant_control": {
            "prompt": control_prompt,
            "optimal_action": "reason_more",
            "evidence": None,
            "clarification": None,
        },
        "audit": {
            "single_source": True,
            "minimal_difference": True,
            "answer_unchanged": True,
            "resolved_intervention_valid": True,
            "control_non_resolving": True,
            "review_status": "pending",
            "notes": (
                "Selected with domain-operation quotas and trivial-pattern "
                "filters. Model screening is required before approval."
            ),
        },
        "provenance": {
            "source_example_id": str(record["example_id"]),
            "source_id": str(record.get("source_id", "")),
            "domain": str(record["domain"]),
            "operation": operation,
            "expression": str(metadata["expression"]),
            "operand_a": int(metadata["operand_a"]),
            "operand_b": int(metadata["operand_b"]),
            "expected_answer": int(metadata["expected_answer"]),
            "normalized_question_hash": str(
                metadata.get("normalized_question_hash", "")
            ),
            "selection_score": list(difficulty_score(record)[:-1]),
            "triviality_reasons": triviality_reasons(record),
        },
    }


def write_jsonl(path: Path, payloads: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True)
                + "\n"
            )

    temporary.replace(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare(
    *,
    input_paths: Mapping[str, Path],
    output_path: Path,
    audit_path: Path,
    overwrite: bool,
) -> None:
    for path in (output_path, audit_path):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite existing artifact: {path}"
            )

    records_by_domain = {
        domain: load_jsonl(path)
        for domain, path in input_paths.items()
    }

    selected, selection_audit = select_candidates(records_by_domain)

    if len(selected) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"expected {EXPECTED_TOTAL} selected records, got {len(selected)}"
        )

    groups = [
        transform_record(record, group_index=index)
        for index, record in enumerate(selected)
    ]

    write_jsonl(output_path, groups)

    domain_counts = Counter(
        group["provenance"]["domain"] for group in groups
    )
    operation_counts = Counter(
        group["provenance"]["operation"] for group in groups
    )

    write_json(
        audit_path,
        {
            "group_count": len(groups),
            "record_count_after_expansion": len(groups) * 3,
            "domain_counts": dict(sorted(domain_counts.items())),
            "operation_counts": dict(sorted(operation_counts.items())),
            "review_status_counts": {
                "pending": len(groups),
                "approved": 0,
                "rejected": 0,
            },
            "manual_review_required": True,
            "model_screening_required": True,
            "output_path": str(output_path),
            "source_paths": {
                domain: str(path)
                for domain, path in input_paths.items()
            },
            **selection_audit,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare diverse reasoning-intervention candidates."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=DEFAULT_AUDIT,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()

    prepare(
        input_paths=INPUT_PATHS,
        output_path=arguments.output,
        audit_path=arguments.audit_output,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
