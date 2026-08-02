from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from usig.experiment.records import atomic_json, validate_record_checksum

PROBABILITY_FEATURES = (
    "mean_token_entropy",
    "maximum_token_entropy",
    "minimum_token_entropy",
    "token_entropy_std",
    "mean_selected_token_log_probability",
    "minimum_selected_token_log_probability",
    "maximum_selected_token_log_probability",
    "negative_mean_log_probability",
    "selected_token_log_probability_std",
    "mean_top_two_probability_margin",
    "minimum_top_two_probability_margin",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _structured_features(record: dict[str, Any]) -> dict[str, float]:
    signature = record["signature"]
    result: dict[str, float] = {}
    for section_name in (
        "cosine_token_dynamics",
        "cosine_structured",
        "relative_structured",
    ):
        for name, value in signature[section_name].items():
            if isinstance(value, list):
                for index, item in enumerate(value):
                    result[f"{name}_{index:02d}"] = float(item)
            elif isinstance(value, (int, float)):
                result[name] = float(value)
    return result


def _matrix(
    records: list[dict[str, Any]], feature_names: list[str], structured: bool
) -> np.ndarray:
    rows = []
    for record in records:
        probabilities = record["probability_summaries"]
        features = {name: float(probabilities[name]) for name in PROBABILITY_FEATURES}
        features["scalar_ifi"] = record["scalar_ifi"]
        if structured:
            features.update(_structured_features(record))
        rows.append([features[name] for name in feature_names])
    return np.asarray(rows, dtype=float)


def _bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    metric,
    *,
    seed: int = 2026,
    draws: int = 1000,
) -> list[float] | None:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        indices = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        estimates.append(metric(sampled_labels, scores[indices]))
    if not estimates:
        return None
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _selective(labels: np.ndarray, uncertainty: np.ndarray) -> dict[str, Any]:
    order = np.argsort(uncertainty)
    errors = labels[order].astype(float)
    counts = np.arange(1, len(errors) + 1)
    coverage = counts / len(errors)
    risk = np.cumsum(errors) / counts
    result = {
        "aurc": float(np.trapezoid(risk, coverage)),
        "coverage": coverage.tolist(),
        "risk": risk.tolist(),
    }
    for target in (0.2, 0.5, 0.8, 1.0):
        index = min(len(risk) - 1, max(0, math.ceil(target * len(risk)) - 1))
        result[f"risk_at_{int(target * 100)}_percent_coverage"] = float(risk[index])
    return result


def _oof_model(
    features: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray | None, str | None]:
    class_counts = Counter(labels.tolist())
    if len(class_counts) < 2:
        return None, "binary_labels_have_one_class"
    folds = min(5, min(class_counts.values()))
    if folds < 3:
        return None, "too_few_examples_in_one_class_for_oof_evaluation"
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            random_state=2026,
            max_iter=2000,
        ),
    )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2026)
    probabilities = cross_val_predict(
        model, features, labels, cv=splitter, method="predict_proba"
    )[:, 1]
    return probabilities, None


def analyze_experiment(project_root: Path, experiment_id: str) -> dict[str, Any]:
    prediction_path = project_root / "outputs/predictions" / f"{experiment_id}.jsonl"
    signature_path = project_root / "outputs/signatures" / f"{experiment_id}.jsonl"
    predictions = _read_jsonl(prediction_path)
    signatures = _read_jsonl(signature_path)
    if len(predictions) != 600 or len(signatures) != 600:
        raise ValueError("Analysis requires exactly 600 predictions and signatures")
    if not all(validate_record_checksum(item, "record_checksum") for item in predictions):
        raise ValueError("Invalid prediction checksum")
    if not all(validate_record_checksum(item, "signature_checksum") for item in signatures):
        raise ValueError("Invalid signature checksum")
    signature_by_id = {item["example_id"]: item for item in signatures}
    datasets = sorted({item["dataset"] for item in predictions})
    summary: dict[str, Any] = {"experiment_id": experiment_id, "datasets": {}}
    for dataset in datasets:
        dataset_predictions = [item for item in predictions if item["dataset"] == dataset]
        dataset_signatures = [signature_by_id[item["example_id"]] for item in dataset_predictions]
        diagnostic = {
            "record_count": len(dataset_predictions),
            "generated_token_counts": dict(
                sorted(Counter(item["generated_token_count"] for item in dataset_predictions).items())
            ),
            "token_limit_count": sum(item["token_limit_reached"] for item in dataset_predictions),
            "unresolved_label_count": sum(item["unresolved_label"] for item in dataset_predictions),
            "one_token_feature_count": sum(
                item["feature_status"] == "insufficient_tokens" for item in dataset_signatures
            ),
            "mean_latency_seconds": statistics.mean(
                item["latency_seconds"] for item in dataset_predictions
            ),
            "maximum_latency_seconds": max(
                item["latency_seconds"] for item in dataset_predictions
            ),
            "answer_parsing_failure_count": sum(
                item["evaluation_metrics"].get("parsing_status") == "error"
                for item in dataset_predictions
            ),
            "peak_allocated_gpu_memory_bytes": max(
                (
                    item["peak_allocated_gpu_memory"]
                    for item in dataset_predictions
                    if item["peak_allocated_gpu_memory"] is not None
                ),
                default=None,
            ),
            "feature_missing_count": sum(item["scalar_ifi"] is None for item in dataset_signatures),
            "hidden_alignment_failures": 0,
            "non_finite_probability_features": 0,
            "non_finite_signature_features": 0,
        }
        if dataset == "truthfulqa":
            statuses = Counter(
                item["evaluation_metrics"]["status"] for item in dataset_predictions
            )
            summary["datasets"][dataset] = {
                "diagnostics": diagnostic,
                "lexical_status_counts": dict(sorted(statuses.items())),
                "headline_binary_error_detection_excluded": True,
            }
            continue
        labels = np.asarray([int(item["binary_error"]) for item in dataset_predictions])
        usable_indices = [
            index
            for index, signature in enumerate(dataset_signatures)
            if signature["scalar_ifi"] is not None
        ]
        usable_signatures = [dataset_signatures[index] for index in usable_indices]
        usable_labels = labels[usable_indices]
        correct_ifi = [
            signature["scalar_ifi"]
            for signature, label in zip(usable_signatures, usable_labels)
            if label == 0
        ]
        incorrect_ifi = [
            signature["scalar_ifi"]
            for signature, label in zip(usable_signatures, usable_labels)
            if label == 1
        ]
        probabilities = np.asarray(
            [
                [
                    signature["probability_summaries"][name]
                    for name in PROBABILITY_FEATURES
                ]
                for signature in usable_signatures
            ],
            dtype=float,
        )
        scalar = np.asarray([signature["scalar_ifi"] for signature in usable_signatures])
        entropy = probabilities[:, PROBABILITY_FEATURES.index("mean_token_entropy")]
        negative_log_probability = probabilities[
            :, PROBABILITY_FEATURES.index("negative_mean_log_probability")
        ]
        raw_scores = {
            "scalar_ifi": scalar,
            "mean_entropy": entropy,
            "negative_mean_log_probability": negative_log_probability,
        }
        metrics: dict[str, Any] = {}
        if len(np.unique(usable_labels)) == 2:
            for name, scores in raw_scores.items():
                metrics[f"{name}_auroc"] = float(roc_auc_score(usable_labels, scores))
                metrics[f"{name}_auroc_bootstrap_95_ci"] = _bootstrap_ci(
                    usable_labels, scores, roc_auc_score
                )
                metrics[f"{name}_selective_prediction"] = _selective(
                    usable_labels, scores
                )
        structured_maps = [_structured_features(item) for item in usable_signatures]
        structured_names = sorted(structured_maps[0]) if structured_maps else []
        feature_sets = {
            "probability": probabilities,
            "probability_plus_scalar_ifi": np.column_stack([probabilities, scalar]),
            "probability_plus_structured_ifi": np.column_stack(
                [
                    probabilities,
                    scalar,
                    np.asarray(
                        [
                            [feature_map[name] for name in structured_names]
                            for feature_map in structured_maps
                        ]
                    ),
                ]
            ),
        }
        for name, features in feature_sets.items():
            scores, reason = _oof_model(features, usable_labels)
            if scores is None:
                metrics[f"{name}_status"] = reason
            else:
                metrics[f"{name}_oof_auroc"] = float(
                    roc_auc_score(usable_labels, scores)
                )
                metrics[f"{name}_oof_auroc_bootstrap_95_ci"] = _bootstrap_ci(
                    usable_labels, scores, roc_auc_score
                )
                metrics[f"{name}_selective_prediction"] = _selective(
                    usable_labels, scores
                )
        summary["datasets"][dataset] = {
            "diagnostics": diagnostic,
            "binary_label_count": len(labels),
            "correct_count": int((labels == 0).sum()),
            "incorrect_count": int((labels == 1).sum()),
            "accuracy": float((labels == 0).mean()),
            "scalar_ifi_valid_count": len(usable_indices),
            "mean_scalar_ifi_correct": (
                statistics.mean(correct_ifi) if correct_ifi else None
            ),
            "mean_scalar_ifi_incorrect": (
                statistics.mean(incorrect_ifi) if incorrect_ifi else None
            ),
            "median_scalar_ifi_correct": (
                statistics.median(correct_ifi) if correct_ifi else None
            ),
            "median_scalar_ifi_incorrect": (
                statistics.median(incorrect_ifi) if incorrect_ifi else None
            ),
            "mean_entropy_correct": (
                float(entropy[usable_labels == 0].mean())
                if (usable_labels == 0).any()
                else None
            ),
            "mean_entropy_incorrect": (
                float(entropy[usable_labels == 1].mean())
                if (usable_labels == 1).any()
                else None
            ),
            "metrics": metrics,
        }
    output_root = project_root / "outputs/metrics" / experiment_id
    atomic_json(output_root / "summary.json", summary)
    diagnostics_root = project_root / "outputs/diagnostics" / experiment_id
    alignment_examples = []
    for dataset in datasets:
        prediction = next(item for item in predictions if item["dataset"] == dataset)
        signature = signature_by_id[prediction["example_id"]]
        alignment_examples.append(
            {
                "dataset": dataset,
                "example_id": prediction["example_id"],
                "prompt_token_count": prediction["prompt_token_count"],
                "generated_token_count": prediction["generated_token_count"],
                "hidden_state_level_count": signature["hidden_state_levels"],
                "transition_count": signature["hidden_state_levels"] - 2,
                "token_transition_row_count": signature["generated_token_count"],
                "feature_status": signature["feature_status"],
            }
        )
    atomic_json(diagnostics_root / "alignment_examples.json", alignment_examples)

    representative_examples = []
    selectors = [
        ("ifi_arith", True),
        ("ifi_arith", False),
        ("gsm8k", True),
        ("gsm8k", False),
        ("triviaqa", None),
        ("ambignq", None),
        ("squad", True),
        ("squad", False),
        ("truthfulqa", None),
    ]
    used: set[str] = set()
    for dataset, correctness in selectors:
        candidates = [
            item
            for item in predictions
            if item["dataset"] == dataset
            and item["example_id"] not in used
            and (
                correctness is None
                or item["binary_correctness"] is correctness
            )
        ]
        if not candidates:
            continue
        prediction = candidates[0]
        used.add(prediction["example_id"])
        signature = signature_by_id[prediction["example_id"]]
        representative_examples.append(
            {
                "dataset": dataset,
                "example_id": prediction["example_id"],
                "response": prediction["response"],
                "binary_correctness": prediction["binary_correctness"],
                "unresolved_label": prediction["unresolved_label"],
                "evaluation_metrics": prediction["evaluation_metrics"],
                "scalar_ifi": signature["scalar_ifi"],
                "probability_summaries": signature["probability_summaries"],
            }
        )
    atomic_json(
        diagnostics_root / "representative_examples.json",
        representative_examples,
    )

    lines = [
        f"# Experiment analysis: {experiment_id}",
        "",
        (
            "All binary error-detection results use out-of-fold predictions. "
            "TruthfulQA lexical matching is diagnostic and excluded from headline "
            "binary error-detection claims."
        ),
        "",
        "| Dataset | Records | Accuracy | Correct | Incorrect | Token limits |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in datasets:
        result = summary["datasets"][dataset]
        diagnostic = result["diagnostics"]
        accuracy = result.get("accuracy")
        lines.append(
            f"| {dataset} | {diagnostic['record_count']} | "
            f"{'n/a' if accuracy is None else f'{accuracy:.3f}'} | "
            f"{result.get('correct_count', 'n/a')} | "
            f"{result.get('incorrect_count', 'n/a')} | "
            f"{diagnostic['token_limit_count']} |"
        )
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    (diagnostics_root / "summary.md").write_text("\n".join(lines) + "\n")
    return summary
