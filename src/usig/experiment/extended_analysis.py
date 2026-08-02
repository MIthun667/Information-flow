from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from usig.experiment.compact_analysis import (
    _aurc,
    _feature_arrays,
    _load_collection,
    balanced_splits,
    comparison_predictions,
    reliability_label,
)
from usig.experiment.large_collection import COMPACT_PROBABILITY_NAMES

SPLIT_SEEDS = (2026, 2027, 2028, 2029, 2030)
ANALYSIS_VERSION = "v2"
BASELINE = ("P", "L")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _paired_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    draws: int = 2000,
) -> dict[str, Any]:
    valid = np.isfinite(baseline) & np.isfinite(candidate)
    labels, baseline, candidate = labels[valid], baseline[valid], candidate[valid]
    observed = float(
        roc_auc_score(labels, candidate) - roc_auc_score(labels, baseline)
    )
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        sample = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[sample])) != 2:
            continue
        differences.append(
            roc_auc_score(labels[sample], candidate[sample])
            - roc_auc_score(labels[sample], baseline[sample])
        )
    return {
        "difference": observed,
        "paired_bootstrap_95_ci": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
        "bootstrap_draws_requested": draws,
        "bootstrap_draws_valid": len(differences),
    }


def _score_summary(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(scores)
    y, s = labels[valid], scores[valid]
    counts = Counter(y.tolist())
    if len(counts) != 2:
        return {
            "sample_count": len(y),
            "correct_count": counts.get(0, 0),
            "incorrect_count": counts.get(1, 0),
            "reliability_status": reliability_label(min(counts.values(), default=0)),
            "auroc": None,
            "auprc": None,
            "aurc": None,
        }
    clipped = np.clip(s, 1e-7, 1 - 1e-7)
    bins = np.linspace(0.0, 1.0, 11)
    expected_calibration_error = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (clipped >= lower) & (
            clipped <= upper if upper == 1.0 else clipped < upper
        )
        if in_bin.any():
            expected_calibration_error += float(in_bin.mean()) * abs(
                float(clipped[in_bin].mean()) - float(y[in_bin].mean())
            )
    order = np.argsort(s)
    sorted_errors = y[order]
    def risk_at_coverage(coverage: float) -> float:
        retained = max(1, int(math.ceil(len(y) * coverage)))
        return float(sorted_errors[:retained].mean())
    def coverage_at_risk(maximum_risk: float) -> float:
        cumulative = np.cumsum(sorted_errors) / np.arange(1, len(y) + 1)
        valid = np.flatnonzero(cumulative <= maximum_risk)
        return float((valid[-1] + 1) / len(y)) if len(valid) else 0.0
    return {
        "sample_count": len(y),
        "correct_count": counts[0],
        "incorrect_count": counts[1],
        "reliability_status": reliability_label(min(counts.values())),
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "aurc": _aurc(y, s),
        "brier_score": float(np.mean((clipped - y) ** 2)),
        "expected_calibration_error_10_bin": expected_calibration_error,
        "negative_log_likelihood": float(
            -np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))
        ),
        "risk_at_80_percent_coverage": risk_at_coverage(0.8),
        "risk_at_90_percent_coverage": risk_at_coverage(0.9),
        "coverage_at_5_percent_risk": coverage_at_risk(0.05),
        "coverage_at_10_percent_risk": coverage_at_risk(0.10),
    }


def _secondary_arrays(
    destination: Path, identifiers: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    records = {
        item["example_id"]: item
        for item in _read_jsonl(
            destination / "signature_ablations/collection.jsonl"
        )
    }
    def selected(record: dict[str, Any]) -> dict[str, Any]:
        return record.get("selected_option", record)

    token_names = (
        "first_token_instability",
        "last_token_instability",
        "maximum_token_instability",
        "mean_token_instability",
        "minimum_token_instability",
        "token_instability_range",
        "token_instability_roughness",
        "token_instability_slope",
        "token_position_of_maximum_instability",
    )
    depth_names = tuple(
        sorted(selected(records[identifiers[0]])["individual_layer_regions"])
    )
    token = np.asarray(
        [
            [
                selected(records[key])["individual_token_dynamics"][name]
                for name in token_names
            ]
            for key in identifiers
        ],
        dtype=float,
    )
    depth = np.asarray(
        [
            [
                selected(records[key])["individual_layer_regions"][name]
                for name in depth_names
            ]
            for key in identifiers
        ],
        dtype=float,
    )
    return token, depth


def _trivia_labels(
    predictions: list[dict[str, Any]], label_variant: str
) -> np.ndarray:
    if label_variant == "strict":
        return np.asarray([item["binary_error"] for item in predictions], dtype=int)
    if label_variant == "alias":
        return np.asarray(
            [
                int(not item["evaluation_diagnostics"]["concise_suffix"]["match"])
                for item in predictions
            ],
            dtype=int,
        )
    if label_variant == "verified":
        return np.asarray(
            [
                int(
                    not (
                        item["evaluation_diagnostics"]["concise_suffix"]["match"]
                        and item["evaluation_diagnostics"]["containment"][
                            "maximum_token_f1"
                        ]
                        >= 0.8
                    )
                )
                for item in predictions
            ],
            dtype=int,
        )
    raise ValueError(f"Unknown TriviaQA label variant: {label_variant}")


def analyze(
    destination: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    non_truncated_only: bool,
    label_variant: str = "strict",
) -> dict[str, Any]:
    predictions, signatures = _load_collection(destination)
    if non_truncated_only:
        predictions = [item for item in predictions if not item["token_limit_reached"]]
    if destination.name == "truthfulqa":
        categories = Counter(
            item.get("evaluation_diagnostics", {}).get(
                "lexical_category", "unmatched_response"
            )
            for item in predictions
        )
        result = {
            "dataset": destination.name,
            "non_truncated_only": non_truncated_only,
            "sample_count": len(predictions),
            "lexical_diagnostic_counts": dict(sorted(categories.items())),
            "reliability_status": "diagnostic_only",
            "reason": "no_definitive_binary_truthfulness_labels",
        }
        _write(output_path, result)
        return result
    arrays, labels, identifiers = _feature_arrays(predictions, signatures)
    prediction_by_id = {item["example_id"]: item for item in predictions}
    ordered_predictions = [prediction_by_id[key] for key in identifiers]
    is_triviaqa = bool(ordered_predictions) and all(
        item.get("dataset") == "triviaqa" for item in ordered_predictions
    )
    if is_triviaqa:
        labels = _trivia_labels(ordered_predictions, label_variant)
    token, depth = _secondary_arrays(destination, identifiers)
    arrays.update({"T": token, "D": depth})
    manifest = {item["example_id"]: item for item in _read_jsonl(manifest_path)}
    strata = [manifest[key]["sampling_stratum"] for key in identifiers]
    folds = min(5, min(Counter(labels.tolist()).values(), default=0))
    feature_sets = {
        "probability_plus_length": BASELINE,
        "probability_length_token_dynamics": ("P", "L", "T"),
        "probability_length_depth_dynamics": ("P", "L", "D"),
        "probability_length_token_and_depth": ("P", "L", "T", "D"),
    }
    result: dict[str, Any] = {
        "dataset": destination.name,
        "analysis_version": ANALYSIS_VERSION,
        "non_truncated_only": non_truncated_only,
        "triviaqa_label_variant": label_variant if is_triviaqa else None,
        "split_seeds": list(SPLIT_SEEDS),
        "excluded_truncated_count": sum(
            item["token_limit_reached"]
            for item in _load_collection(destination)[0]
        )
        if non_truncated_only
        else 0,
        "comparisons": {},
    }
    if folds < 2:
        result["reason"] = "class_deficient"
        _write(output_path, result)
        return result
    scores_by_name: dict[str, list[np.ndarray]] = {
        name: [] for name in feature_sets
    }
    for seed in SPLIT_SEEDS:
        splits = balanced_splits(labels, strata, folds=folds, seed=seed)
        for name, families in feature_sets.items():
            scores, _ = comparison_predictions(arrays, labels, splits, families)
            scores_by_name[name].append(scores)
    baseline = np.mean(scores_by_name["probability_plus_length"], axis=0)
    for name, scores in scores_by_name.items():
        mean_scores = np.mean(scores, axis=0)
        seed_metrics = [_score_summary(labels, item) for item in scores]
        summary = _score_summary(labels, mean_scores)
        summary["seed_aurocs"] = [item["auroc"] for item in seed_metrics]
        if name != "probability_plus_length":
            summary["gain_over_probability_plus_length"] = _paired_bootstrap(
                labels, baseline, mean_scores, seed=2026
            )
        result["comparisons"][name] = summary
    _write(output_path, result)
    return result


def consolidated_report(
    source_root: Path, output_json: Path, output_markdown: Path
) -> dict[str, Any]:
    rows = []
    for destination in sorted(path for path in source_root.iterdir() if path.is_dir()):
        prediction_path = destination / "predictions/collection.jsonl"
        if not prediction_path.exists():
            continue
        predictions = _read_jsonl(prediction_path)
        counts = Counter(item["binary_error"] for item in predictions)
        metrics_path = destination / "confound_controlled_metrics/compact_comparisons.json"
        comparison = {}
        baseline = {}
        if metrics_path.exists() and destination.name != "truthfulqa":
            metrics = json.loads(metrics_path.read_text())
            comparison = metrics["comparisons"].get(
                "probability_length_residual_compact_ifi", {}
            )
            baseline = metrics["comparisons"].get("probability_plus_length", {})
        rows.append(
            {
                "dataset": destination.name,
                "sample_count": len(predictions),
                "correct_count": counts.get(0, 0),
                "incorrect_count": counts.get(1, 0),
                "unresolved_count": counts.get(None, 0),
                "truncation_count": sum(item["token_limit_reached"] for item in predictions),
                "truncation_rate": sum(item["token_limit_reached"] for item in predictions)
                / len(predictions),
                "auroc": comparison.get("auroc"),
                "auprc": comparison.get("auprc"),
                "aurc": comparison.get("aurc"),
                "reliability_status": comparison.get("reliability_status", "diagnostic_only"),
                "auroc_gain_over_probability_plus_length": (
                    comparison["auroc"] - baseline["auroc"]
                    if comparison.get("auroc") is not None
                    and baseline.get("auroc") is not None
                    else None
                ),
            }
        )
    payload = {"version": "qwen_1_5b_v1", "datasets": rows}
    _write(output_json, payload)
    lines = [
        "# Qwen2.5-1.5B IFI Version 1 dataset report",
        "",
        "| Dataset | N | Correct | Incorrect | Unresolved | Truncation | AUROC | AUPRC | AURC | Reliability | ΔAUROC vs P+L |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    def fmt(value: Any) -> str:
        return "—" if value is None else f"{value:.4f}"
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['sample_count']} | {row['correct_count']} | "
            f"{row['incorrect_count']} | {row['unresolved_count']} | "
            f"{row['truncation_rate']:.1%} | {fmt(row['auroc'])} | "
            f"{fmt(row['auprc'])} | {fmt(row['aurc'])} | "
            f"{row['reliability_status']} | "
            f"{fmt(row['auroc_gain_over_probability_plus_length'])} |"
        )
    if output_markdown.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {output_markdown}")
    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def calibration_gate(destination: Path, output_path: Path) -> dict[str, Any]:
    predictions, _ = _load_collection(destination)
    sample = predictions[:100]
    truncation_rate = sum(item["token_limit_reached"] for item in sample) / len(sample)
    parsing_failures = sum(
        item["evaluation_metrics"].get("parsing_status") != "ok"
        for item in sample
    )
    result = {
        "sample_count": len(sample),
        "required_sample_count": 100,
        "required_max_new_tokens": 256,
        "maximum_truncation_rate": 0.05,
        "observed_truncation_rate": truncation_rate,
        "final_answer_parsing_failures": parsing_failures,
        "passed": len(sample) == 100 and truncation_rate <= 0.05 and parsing_failures == 0,
    }
    _write(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    report = sub.add_parser("report")
    report.add_argument("--source-root", type=Path, required=True)
    report.add_argument("--output-json", type=Path, required=True)
    report.add_argument("--output-markdown", type=Path, required=True)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--destination", type=Path, required=True)
    analysis.add_argument("--manifest", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--non-truncated-only", action="store_true")
    analysis.add_argument(
        "--label-variant", choices=("strict", "alias", "verified"), default="strict"
    )
    gate = sub.add_parser("calibration-gate")
    gate.add_argument("--destination", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "report":
        result = consolidated_report(
            args.source_root, args.output_json, args.output_markdown
        )
    elif args.action == "calibration-gate":
        result = calibration_gate(args.destination, args.output)
    else:
        result = analyze(
            args.destination,
            args.manifest,
            args.output,
            non_truncated_only=args.non_truncated_only,
            label_variant=args.label_variant,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
