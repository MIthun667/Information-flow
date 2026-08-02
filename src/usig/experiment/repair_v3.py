from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from usig.evaluation.audit_rules import evaluate_interpretation_segments
from usig.data.normalization.text import normalize_answer
from usig.experiment.compact_analysis import reliability_label
from usig.experiment.records import canonical_json

VERSION = "v3"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        for value in values:
            handle.write(canonical_json(value) + "\n")
    temporary.replace(path)


def interpretation_label(
    response: str, interpretations: list[dict[str, Any]]
) -> dict[str, Any]:
    diagnostics = evaluate_interpretation_segments(response, interpretations)
    alias_owners: dict[str, set[str]] = {}
    for interpretation in interpretations:
        identifier = str(interpretation.get("interpretation_id"))
        for alias in interpretation["reference_answers"]:
            normalized = normalize_answer(alias)
            if normalized:
                alias_owners.setdefault(normalized, set()).add(identifier)
    normalized_response = normalize_answer(response)
    shared_matches = sorted(
        alias
        for alias, owners in alias_owners.items()
        if len(owners) > 1 and alias in normalized_response
    )
    covered = diagnostics["covered_interpretations"]
    total = diagnostics["interpretation_count"]
    segments = diagnostics["segments"]
    matched_segment_count = len(
        {
            segment
            for item in diagnostics["interpretations"]
            for segment in item["matched_segment_indices"]
        }
    )
    precision = (
        min(1.0, matched_segment_count / len(segments))
        if segments
        else 0.0
    )
    coverage = covered / total if total else 0.0
    if covered == 0:
        label = "incorrect"
    elif covered == total and precision == 1.0:
        label = "fully_correct"
    else:
        label = "partially_correct"
    unresolved_reason = (
        "shared_alias_matches_multiple_interpretations"
        if shared_matches
        else None
    )
    return {
        "label": None if unresolved_reason else label,
        "provisional_label": label,
        "ordinal_label": (
            None
            if unresolved_reason
            else {"incorrect": 0, "partially_correct": 1, "fully_correct": 2}[label]
        ),
        "unresolved": unresolved_reason is not None,
        "unresolved_reason": unresolved_reason,
        "shared_alias_matches": shared_matches,
        "interpretation_count": total,
        "covered_interpretations": covered,
        "interpretation_coverage": coverage,
        "matched_segment_count": matched_segment_count,
        "response_segment_count": len(segments),
        "interpretation_precision": precision,
        "unsupported_interpretation_count": max(0, len(segments) - matched_segment_count),
        "coverage_score": coverage,
        "fully_wrong_target": None if unresolved_reason else int(label == "incorrect"),
        "incomplete_target": None if unresolved_reason else int(label != "fully_correct"),
        "diagnostics": diagnostics,
    }


def relabel_ambignq(
    normalized_path: Path,
    predictions_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    canonical = {item["example_id"]: item for item in read_jsonl(normalized_path)}
    predictions = read_jsonl(predictions_path)
    rows = []
    for prediction in predictions:
        record = canonical[prediction["example_id"]]
        interpretations = record["interpretations"] or [
            {
                "interpretation_id": "single_answer",
                "disambiguated_question": record["question"],
                "reference_answers": record["reference_answers"],
            }
        ]
        label = interpretation_label(prediction["response"], interpretations)
        row = {
            "version": VERSION,
            "example_id": prediction["example_id"],
            "source_prediction_checksum": prediction["record_checksum"],
            **label,
        }
        row["label_checksum"] = hashlib.sha256(
            canonical_json(row).encode()
        ).hexdigest()
        rows.append(row)
    counts = Counter(item["label"] for item in rows if not item["unresolved"])
    unresolved_count = sum(item["unresolved"] for item in rows)
    fully_wrong_counts = Counter(
        item["fully_wrong_target"] for item in rows if not item["unresolved"]
    )
    incomplete_counts = Counter(
        item["incomplete_target"] for item in rows if not item["unresolved"]
    )
    report = {
        "version": VERSION,
        "source": str(predictions_path),
        "sample_count": len(rows),
        "class_counts": dict(sorted(counts.items())),
        "unresolved_count": unresolved_count,
        "automatic_label_status": "provisional_pending_manual_audit",
        "manual_audit_label_precision": None,
        "manual_audit_disagreement_rate": None,
        "targets": {
            "fully_wrong_vs_at_least_partially_correct": {
                "negative_count": fully_wrong_counts[0],
                "positive_count": fully_wrong_counts[1],
                "positive_class": "fully_wrong",
                "reliability_status": reliability_label(
                    min(fully_wrong_counts.values(), default=0)
                ),
            },
            "incomplete_vs_fully_complete": {
                "negative_count": incomplete_counts[0],
                "positive_count": incomplete_counts[1],
                "positive_class": "incomplete",
                "reliability_status": reliability_label(
                    min(incomplete_counts.values(), default=0)
                ),
            },
        },
        "mean_interpretation_coverage": sum(
            item["interpretation_coverage"] for item in rows
        )
        / len(rows),
        "mean_interpretation_precision": sum(
            item["interpretation_precision"] for item in rows
        )
        / len(rows),
    }
    write_jsonl(output_directory / "interpretation_labels.jsonl", rows)
    write_json(output_directory / "class_count_report.json", report)
    audit_rows = []
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = "unresolved" if row["unresolved"] else str(row["label"])
        by_label.setdefault(key, []).append(row)
    for label, candidates in sorted(by_label.items()):
        count = len(candidates) if label == "unresolved" else min(20, len(candidates))
        for row in sorted(candidates, key=lambda item: item["example_id"])[:count]:
            record = canonical[row["example_id"]]
            prediction = next(
                item for item in predictions if item["example_id"] == row["example_id"]
            )
            audit_rows.append(
                {
                    "example_id": row["example_id"],
                    "stratum": label,
                    "question": record["question"],
                    "interpretations": record["interpretations"],
                    "response": prediction["response"],
                    "automatic_label": row["label"],
                    "automatic_coverage": row["coverage_score"],
                    "human_label": None,
                    "human_coverage": None,
                    "agreement": None,
                    "audit_status": "pending_independent_manual_review",
                }
            )
    write_jsonl(output_directory / "manual_audit_sample.jsonl", audit_rows)
    write_json(
        output_directory / "manual_audit_status.json",
        {
            "sample_count": len(audit_rows),
            "stratum_counts": dict(Counter(item["stratum"] for item in audit_rows)),
            "completed_count": 0,
            "label_precision": None,
            "disagreement_rate": None,
            "status": "pending_independent_manual_review",
        },
    )
    return report


def prepare_truthfulqa_mc_manifest(
    normalized_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    records = sorted(read_jsonl(normalized_path), key=lambda item: item["example_id"])
    if limit is not None:
        records = records[:limit]
    rows = []
    for order, record in enumerate(records):
        options = [
            *[
                {
                    "text": answer,
                    "correct": True,
                    "mc1_correct": answer == record["metadata"]["best_answer"],
                }
                for answer in record["reference_answers"]
            ],
            *[
                {"text": answer, "correct": False, "mc1_correct": False}
                for answer in record["incorrect_reference_answers"]
            ],
        ]
        seed = int(hashlib.sha256(record["example_id"].encode()).hexdigest()[:16], 16)
        keyed = sorted(
            options,
            key=lambda option: hashlib.sha256(
                f"{seed}:{option['text']}".encode()
            ).hexdigest(),
        )
        rows.append(
            {
                "version": VERSION,
                "example_id": record["example_id"],
                "group_id": record["group_id"],
                "selection_order": order,
                "sampling_stratum": (
                    f"category:{record.get('category')}|"
                    f"type:{record['metadata'].get('adversarial_type')}"
                ),
                "option_count": len(keyed),
                "correct_option_index": next(
                    index for index, option in enumerate(keyed) if option["mc1_correct"]
                ),
                "correct_option_indices": [
                    index for index, option in enumerate(keyed) if option["correct"]
                ],
                "mc_protocols": ["MC1", "MC2"],
                "option_order_checksum": hashlib.sha256(
                    canonical_json([option["text"] for option in keyed]).encode()
                ).hexdigest(),
            }
        )
    write_jsonl(output_path, rows)
    return {
        "version": VERSION,
        "sample_count": len(rows),
        "manifest": str(output_path),
        "checksum": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }


def gate_report(
    predictions_path: Path,
    output_path: Path,
    *,
    expected_count: int,
    maximum_failure_rate: float = 0.05,
) -> dict[str, Any]:
    rows = read_jsonl(predictions_path)
    failures = sum(item.get("unresolved_label", False) for item in rows)
    rate = failures / len(rows) if rows else 1.0
    class_counts = Counter(item.get("binary_error") for item in rows)
    minority = min(
        (class_counts.get(0, 0), class_counts.get(1, 0)), default=0
    )
    result = {
        "version": VERSION,
        "sample_count": len(rows),
        "expected_count": expected_count,
        "failure_count": failures,
        "failure_rate": rate,
        "maximum_failure_rate": maximum_failure_rate,
        "class_counts": {
            "correct": class_counts.get(0, 0),
            "incorrect": class_counts.get(1, 0),
        },
        "reliability_status": reliability_label(minority),
        "minimum_required_per_class": 20,
        "passed": (
            len(rows) == expected_count
            and rate <= maximum_failure_rate
            and minority >= 20
        ),
    }
    write_json(output_path, result)
    return result


def analyze_ambignq_targets(
    labels_path: Path,
    destination: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    import math
    import numpy as np
    from scipy.stats import spearmanr
    from usig.experiment.compact_analysis import (
        _feature_arrays,
        _load_collection,
        balanced_splits,
        comparison_predictions,
    )
    from usig.experiment.extended_analysis import (
        SPLIT_SEEDS,
        _paired_bootstrap,
        _score_summary,
    )

    predictions, signatures = _load_collection(destination)
    labels_by_id = {item["example_id"]: item for item in read_jsonl(labels_path)}
    manifest = {item["example_id"]: item for item in read_jsonl(manifest_path)}
    results = {}
    continuous_scores: dict[str, tuple[list[str], np.ndarray]] = {}
    for target in ("fully_wrong_target", "incomplete_target"):
        relabeled = []
        for prediction in predictions:
            value = labels_by_id[prediction["example_id"]][target]
            if value is None:
                continue
            relabeled.append(
                {**prediction, "binary_error": value, "binary_correctness": not value}
            )
        arrays, labels, identifiers = _feature_arrays(relabeled, signatures)
        strata = [manifest[identifier]["sampling_stratum"] for identifier in identifiers]
        folds = min(5, min(Counter(labels.tolist()).values(), default=0))
        if folds < 2:
            results[target] = {
                "reason": "class_deficient",
                "reliability_status": reliability_label(
                    min(Counter(labels.tolist()).values(), default=0)
                ),
            }
            continue
        scores = {"probability_plus_length": [], "probability_length_compact_ifi": []}
        for seed in SPLIT_SEEDS:
            splits = balanced_splits(labels, strata, folds=folds, seed=seed)
            for name, families in (
                ("probability_plus_length", ("P", "L")),
                ("probability_length_compact_ifi", ("P", "L", "C")),
            ):
                predicted, _ = comparison_predictions(
                    arrays, labels, splits, families
                )
                scores[name].append(predicted)
        baseline = np.mean(scores["probability_plus_length"], axis=0)
        candidate = np.mean(scores["probability_length_compact_ifi"], axis=0)
        continuous_scores[target] = (identifiers, candidate)
        results[target] = {
            "probability_plus_length": _score_summary(labels, baseline),
            "probability_length_compact_ifi": {
                **_score_summary(labels, candidate),
                "gain_over_probability_plus_length": _paired_bootstrap(
                    labels, baseline, candidate, seed=2026
                ),
            },
            "split_seeds": list(SPLIT_SEEDS),
        }
    coverage_ranking = {"reason": "class_deficient"}
    if "fully_wrong_target" in continuous_scores:
        identifiers, uncertainty = continuous_scores["fully_wrong_target"]
        quality = np.asarray(
            [labels_by_id[identifier]["coverage_score"] for identifier in identifiers],
            dtype=float,
        )
        order = np.argsort(uncertainty)
        cumulative_risk = np.cumsum(1.0 - quality[order]) / np.arange(
            1, len(quality) + 1
        )
        correlation = spearmanr(uncertainty, -quality)
        coverage_ranking = {
            "sample_count": len(quality),
            "spearman_uncertainty_vs_negative_coverage": float(correlation.statistic),
            "spearman_p_value": float(correlation.pvalue),
            "coverage_quality_aurc": float(cumulative_risk.mean()),
            "risk_at_80_percent_coverage": float(
                cumulative_risk[max(0, math.ceil(0.8 * len(quality)) - 1)]
            ),
            "risk_at_90_percent_coverage": float(
                cumulative_risk[max(0, math.ceil(0.9 * len(quality)) - 1)]
            ),
            "unresolved_excluded_count": sum(
                item["unresolved"] for item in labels_by_id.values()
            ),
        }
    report = {
        "version": VERSION,
        "targets": results,
        "continuous_coverage_ranking": coverage_ranking,
    }
    write_json(output_path, report)
    return report


def report_v3(repair_root: Path, output_json: Path, output_markdown: Path) -> dict[str, Any]:
    artifacts = {}
    for path in sorted(repair_root.rglob("*.json")):
        if path.name in {
            "class_count_report.json",
            "target_analysis.json",
            "calibration_gate.json",
            "diagnostics.json",
            "depth_analysis.json",
            "power_analysis.json",
            "high_confidence_false_v3.json",
        }:
            artifacts[str(path.relative_to(repair_root))] = json.loads(
                path.read_text(encoding="utf-8")
            )
    result = {"version": VERSION, "artifact_count": len(artifacts), "artifacts": artifacts}
    write_json(output_json, result)
    lines = [
        "# IFI Version 3 repair report",
        "",
        f"Included report artifacts: {len(artifacts)}",
        "",
    ]
    for name, value in artifacts.items():
        lines.extend([f"## {name}", "", "```json", json.dumps(value, indent=2), "```", ""])
    if output_markdown.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {output_markdown}")
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    ambig = sub.add_parser("ambignq-labels")
    ambig.add_argument("--normalized", type=Path, required=True)
    ambig.add_argument("--predictions", type=Path, required=True)
    ambig.add_argument("--output-directory", type=Path, required=True)
    manifest = sub.add_parser("truthfulqa-mc-manifest")
    manifest.add_argument("--normalized", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--limit", type=int)
    gate = sub.add_parser("gate")
    gate.add_argument("--predictions", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--expected-count", type=int, required=True)
    analysis = sub.add_parser("ambignq-analysis")
    analysis.add_argument("--labels", type=Path, required=True)
    analysis.add_argument("--destination", type=Path, required=True)
    analysis.add_argument("--manifest", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("report")
    report.add_argument("--repair-root", type=Path, required=True)
    report.add_argument("--output-json", type=Path, required=True)
    report.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "ambignq-labels":
        result = relabel_ambignq(
            args.normalized, args.predictions, args.output_directory
        )
    elif args.action == "truthfulqa-mc-manifest":
        result = prepare_truthfulqa_mc_manifest(
            args.normalized, args.output, limit=args.limit
        )
    elif args.action == "gate":
        result = gate_report(
            args.predictions, args.output, expected_count=args.expected_count
        )
    elif args.action == "ambignq-analysis":
        result = analyze_ambignq_targets(
            args.labels, args.destination, args.manifest, args.output
        )
    else:
        result = report_v3(
            args.repair_root, args.output_json, args.output_markdown
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
