from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from usig.experiment.repair_v3 import interpretation_label, read_jsonl, write_json, write_jsonl


def audit(dataset: str, predictions: Path, normalized: Path, metadata: Path, output: Path) -> dict[str, Any]:
    rows = read_jsonl(predictions)
    canonical = {
        item["example_id"]: item
        for path in normalized.parent.glob("*.jsonl")
        for item in read_jsonl(path)
    }
    model = json.loads(metadata.read_text())
    lengths = [item["generated_token_count"] for item in rows]
    failures = []
    labels = Counter()
    for item in rows:
        record = canonical[item["example_id"]]
        if dataset == "ambignq":
            interpretations = record["interpretations"] or [{
                "interpretation_id": "single_answer",
                "reference_answers": record["reference_answers"],
            }]
            label = interpretation_label(item["response"], interpretations)
            labels["unresolved" if label["unresolved"] else label["label"]] += 1
            category = (
                "unresolved"
                if label["ordinal_label"] is None
                else "useful"
                if label["ordinal_label"] > 0
                else "fully_incorrect"
            )
        else:
            category = "token_limit" if item["token_limit_reached"] else (
                "correct" if item["binary_correctness"] else "incorrect"
            )
            labels[category] += 1
        if category not in ("correct", "useful"):
            failures.append({
                "example_id": item["example_id"],
                "question": record["question"],
                "answerable": record.get("answerable"),
                "response": item["response"],
                "generated_token_count": item["generated_token_count"],
                "stop_reason": item.get("stop_reason"),
                "category": category,
            })
    report = {
        "dataset": dataset,
        "requested_count": 100,
        "collected_count": len(rows),
        "label_counts": dict(labels),
        "truncation_count": sum(item["token_limit_reached"] for item in rows),
        "truncation_rate": sum(item["token_limit_reached"] for item in rows) / len(rows),
        "parsing_failure_count": sum(item.get("evaluation_metrics", {}).get("parsing_status") not in (None, "ok") for item in rows),
        "eos_rate": sum(item.get("stop_reason") == "eos_token" for item in rows) / len(rows),
        "token_limit_rate": sum(item.get("stop_reason") == "token_limit" for item in rows) / len(rows),
        "generated_tokens": {
            "minimum": min(lengths), "median": statistics.median(lengths),
            "mean": statistics.mean(lengths), "maximum": max(lengths),
        },
        "model": model["model"],
        "prompt_versions": model["prompt_versions"],
        "max_new_tokens": model["max_new_tokens"],
        "evaluator_version": model.get("evaluator_version"),
        "failure_example_count": len(failures),
    }
    write_json(output / "audit_report.json", report)
    write_jsonl(output / "failure_examples.jsonl", failures)
    return report


def decision(dataset: str, gate_path: Path, labels_path: Path | None, output: Path) -> dict[str, Any]:
    gate = json.loads(gate_path.read_text())
    result = {
        "dataset": dataset,
        "technical_gate_passed": gate["gate_passed"],
        "technical_gate_reasons": gate["gate_reasons"],
        "full_collection_authorized": gate["gate_passed"],
        "failure_type": None if gate["gate_passed"] else "technical_or_class_balance",
    }
    if dataset == "ambignq":
        labels = json.loads(labels_path.read_text()) if labels_path else {}
        result.update({
            "automatic_labels": labels.get("class_counts"),
            "unresolved_count": labels.get("unresolved_count"),
            "manual_audit_status": labels.get("automatic_label_status"),
            "full_collection_authorized": False,
            "failure_type": "evaluator_validation_pending",
            "decision_reason": "Manual label precision and disagreement are not yet measured.",
        })
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("squad", "ambignq"), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    if args.gate:
        result = decision(args.dataset, args.gate, args.labels, args.output)
    else:
        result = audit(args.dataset, args.predictions, args.normalized, args.metadata, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
