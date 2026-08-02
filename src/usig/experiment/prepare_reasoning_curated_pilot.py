"""Select and transform real IFI-ARITH records for the reasoning dry pilot."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUTS = (
    PROJECT_ROOT / "data/normalized/ifi_arith/larger_integer.jsonl",
    PROJECT_ROOT / "data/normalized/ifi_arith/moderate_multiplicative.jsonl",
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/uncertainty_flow/pilot_v1/curated/reasoning_groups.jsonl"
)

DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "outputs/uncertainty_flow/pilot_v1/audits/"
    "reasoning_candidate_audit.json"
)

TARGET_OPERATION_COUNTS = {
    "addition": 3,
    "subtraction": 2,
    "multiplication": 3,
    "division": 2,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON objects from a JSONL file."""

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object"
                )

            records.append(payload)

    return records


def validate_source_record(record: dict[str, Any]) -> None:
    """Validate fields required by the reasoning transformation."""

    required = {
        "example_id",
        "dataset",
        "domain",
        "question",
        "reference_answers",
        "metadata",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(
            f"source record is missing fields: {', '.join(missing)}"
        )

    metadata = record["metadata"]
    if not isinstance(metadata, dict):
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


def difficulty_key(record: dict[str, Any]) -> tuple[int, int, str]:
    """Prefer examples with larger operands and more digits."""

    metadata = record["metadata"]
    operand_a = abs(int(metadata["operand_a"]))
    operand_b = abs(int(metadata["operand_b"]))

    digit_count = len(str(operand_a)) + len(str(operand_b))
    magnitude = max(operand_a, operand_b)

    return (-digit_count, -magnitude, str(record["example_id"]))


def reasoning_scaffold(
    operation: str,
    operand_a: int,
    operand_b: int,
) -> str:
    """Return an operation-specific scaffold without revealing the answer."""

    if operation == "addition":
        return (
            "Work column by column from right to left. Track every carry, "
            "then combine the resulting place values."
        )

    if operation == "subtraction":
        return (
            "Work column by column from right to left. Borrow where needed "
            "and verify the result by adding it back to the smaller term."
        )

    if operation == "multiplication":
        return (
            f"Decompose {operand_b} into place values. Multiply "
            f"{operand_a} by each part separately, then add the partial "
            "products."
        )

    if operation == "division":
        return (
            "Find the integer quotient by checking how many equal groups fit. "
            "Verify it by multiplying the proposed quotient by the divisor."
        )

    raise ValueError(f"unsupported operation: {operation}")


def irrelevant_instruction(operation: str) -> str:
    """Return a matched but non-resolving control instruction."""

    controls = {
        "addition": (
            "Arithmetic symbols have been used in written mathematics for "
            "centuries. Read the expression carefully before responding."
        ),
        "subtraction": (
            "Subtraction is commonly represented by a horizontal minus sign. "
            "Read the expression carefully before responding."
        ),
        "multiplication": (
            "Multiplication may be represented by a cross, dot, or adjacency. "
            "Read the expression carefully before responding."
        ),
        "division": (
            "Division may be represented by a slash or division symbol. "
            "Read the expression carefully before responding."
        ),
    }

    try:
        return controls[operation]
    except KeyError as error:
        raise ValueError(f"unsupported operation: {operation}") from error


def select_candidates(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select exactly ten operation-balanced deterministic candidates."""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        validate_source_record(record)
        operation = str(record["metadata"]["operation"])

        if operation in TARGET_OPERATION_COUNTS:
            buckets[operation].append(record)

    selected: list[dict[str, Any]] = []

    for operation, target_count in TARGET_OPERATION_COUNTS.items():
        ordered = sorted(buckets[operation], key=difficulty_key)

        if len(ordered) < target_count:
            raise ValueError(
                f"operation {operation!r} has only {len(ordered)} eligible "
                f"records; {target_count} are required"
            )

        selected.extend(ordered[:target_count])

    return sorted(
        selected,
        key=lambda record: (
            str(record["metadata"]["operation"]),
            str(record["example_id"]),
        ),
    )


def transform_record(
    record: dict[str, Any],
    *,
    group_index: int,
) -> dict[str, Any]:
    """Convert one normalized IFI-ARITH record into a curated group."""

    metadata = record["metadata"]
    operation = str(metadata["operation"])
    operand_a = int(metadata["operand_a"])
    operand_b = int(metadata["operand_b"])
    question = str(record["question"])
    source_prompt = (
        "Solve the arithmetic problem. Return only the final numeric answer."
    )

    group_id = f"reasoning_{group_index:04d}"

    original_prompt = f"{source_prompt}\n{question} ="

    scaffold = reasoning_scaffold(
        operation=operation,
        operand_a=operand_a,
        operand_b=operand_b,
    )
    resolved_prompt = (
        f"{source_prompt}\n"
        f"Use this reasoning scaffold: {scaffold}\n"
        f"{question} ="
    )

    control = irrelevant_instruction(operation)
    control_prompt = (
        f"{source_prompt}\n"
        f"Additional information: {control}\n"
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
                "Automatically selected from IFI-ARITH. Manual review "
                "required before approval."
            ),
        },
        "provenance": {
            "source_example_id": str(record["example_id"]),
            "source_id": str(record.get("source_id", "")),
            "domain": str(record["domain"]),
            "operation": operation,
            "expression": str(metadata["expression"]),
            "operand_a": operand_a,
            "operand_b": operand_b,
            "expected_answer": metadata["expected_answer"],
            "normalized_question_hash": str(
                metadata.get("normalized_question_hash", "")
            ),
        },
    }


def write_jsonl(path: Path, payloads: Iterable[dict[str, Any]]) -> None:
    """Write JSONL atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True)
                + "\n"
            )

    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare(
    *,
    input_paths: tuple[Path, ...],
    output_path: Path,
    audit_path: Path,
    overwrite: bool,
) -> None:
    """Prepare ten real IFI-ARITH reasoning groups."""

    for path in (output_path, audit_path):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite existing artifact: {path}"
            )

    records: list[dict[str, Any]] = []
    for path in input_paths:
        records.extend(load_jsonl(path))

    selected = select_candidates(records)
    groups = [
        transform_record(record, group_index=index)
        for index, record in enumerate(selected)
    ]

    write_jsonl(output_path, groups)

    operation_counts = Counter(
        group["provenance"]["operation"] for group in groups
    )
    domain_counts = Counter(
        group["provenance"]["domain"] for group in groups
    )

    write_json(
        audit_path,
        {
            "group_count": len(groups),
            "record_count_after_expansion": len(groups) * 3,
            "operation_counts": dict(sorted(operation_counts.items())),
            "domain_counts": dict(sorted(domain_counts.items())),
            "review_status_counts": {
                "pending": len(groups),
                "approved": 0,
                "rejected": 0,
            },
            "manual_review_required": True,
            "output_path": str(output_path),
            "source_paths": [str(path) for path in input_paths],
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare real IFI-ARITH reasoning pilot groups."
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
        input_paths=DEFAULT_INPUTS,
        output_path=arguments.output,
        audit_path=arguments.audit_output,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
