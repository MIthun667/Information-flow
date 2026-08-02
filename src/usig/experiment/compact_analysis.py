from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from usig.experiment.large_collection import (
    COMPACT_IFI_NAMES,
    COMPACT_PROBABILITY_NAMES,
)
from usig.experiment.records import canonical_json, validate_record_checksum

LENGTH_NAMES = (
    "prompt_token_count",
    "generated_token_count",
    "response_character_count",
    "token_limit_reached",
)
COMPARISONS = {
    "probability": ("P",),
    "length_controls": ("L",),
    "probability_plus_length": ("P", "L"),
    "scalar_ifi": ("I",),
    "probability_plus_scalar_ifi": ("P", "I"),
    "probability_length_scalar_ifi": ("P", "L", "I"),
    "compact_ifi": ("C",),
    "probability_plus_compact_ifi": ("P", "C"),
    "probability_length_compact_ifi": ("P", "L", "C"),
    "probability_length_residual_scalar_ifi": ("P", "L", "I_RESIDUAL"),
    "probability_length_residual_compact_ifi": ("P", "L", "C_RESIDUAL"),
}
REGULARIZATION_VALUES = (0.1, 1.0, 10.0)


def reliability_label(minority_count: int) -> str:
    if minority_count < 20:
        return "descriptive_only"
    if minority_count < 50:
        return "exploratory"
    if minority_count < 100:
        return "preliminary"
    return "suitable_for_primary_analysis"


def gsm8k_decision(correct_count: int) -> dict[str, Any]:
    if correct_count < 20:
        decision = "do_not_perform_predictive_uncertainty_evaluation"
    elif correct_count < 50:
        decision = "exploratory_evaluation_only"
    elif correct_count < 100:
        decision = "continue_larger_sample_with_reliability_warnings"
    else:
        decision = "suitable_for_planned_larger_error_prediction_evaluation"
    return {"correct_count": correct_count, "decision": decision}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite analysis artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _load_collection(
    destination: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    predictions = _read_jsonl(destination / "predictions/collection.jsonl")
    signatures = _read_jsonl(
        destination / "compact_signatures/collection.jsonl"
    )
    if not all(
        validate_record_checksum(item, "record_checksum") for item in predictions
    ):
        raise ValueError("Prediction checksum failure")
    if not all(
        validate_record_checksum(item, "signature_checksum") for item in signatures
    ):
        raise ValueError("Compact signature checksum failure")
    signature_by_id = {item["example_id"]: item for item in signatures}
    if {item["example_id"] for item in predictions} != set(signature_by_id):
        raise ValueError("Prediction-signature join failure")
    return predictions, signature_by_id


def write_gsm8k_decision(destination: Path) -> dict[str, Any]:
    predictions, _ = _load_collection(destination)
    correct = sum(item["binary_correctness"] is True for item in predictions)
    result = {
        **gsm8k_decision(correct),
        "sample_count": len(predictions),
        "incorrect_count": sum(
            item["binary_correctness"] is False for item in predictions
        ),
        "unresolved_count": sum(
            item["binary_correctness"] is None for item in predictions
        ),
    }
    checksum = hashlib.sha256(canonical_json(result).encode()).hexdigest()
    result["decision_checksum"] = checksum
    decision_path = (
        destination / "class_balance_decisions/gsm8k_calibration.json"
    )
    if decision_path.exists():
        if json.loads(decision_path.read_text()) != result:
            raise FileExistsError(
                f"Existing GSM8K decision conflicts with current results: {decision_path}"
            )
    else:
        _atomic_json(decision_path, result)
    return result


def lexical_diagnostic_summary(
    destination: Path, output_path: Path
) -> dict[str, Any]:
    predictions, _ = _load_collection(destination)
    categories = Counter()
    for item in predictions:
        diagnostics = item.get("evaluation_diagnostics", {})
        category = diagnostics.get("lexical_category")
        if category is None:
            category = (
                item.get("evaluation_metrics", {}).get("status")
                or "unmatched_response"
            )
        categories[category] += 1
    result = {
        "sample_count": len(predictions),
        "lexical_diagnostic_counts": dict(sorted(categories.items())),
        "definitive_truthfulness_labels_assigned": False,
        "headline_binary_uncertainty_metrics": False,
    }
    _atomic_json(output_path, result)
    return result


def balanced_splits(
    labels: np.ndarray,
    strata: list[str],
    *,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    assignments: list[list[int]] = [[] for _ in range(folds)]
    buckets: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, (label, stratum) in enumerate(zip(labels, strata)):
        buckets[(int(label), stratum)].append(index)
    rng = np.random.default_rng(seed)
    label_offsets: dict[int, int] = defaultdict(int)
    keys = sorted(buckets)
    rng.shuffle(keys)
    for key in keys:
        indices = np.asarray(buckets[key])
        rng.shuffle(indices)
        label = key[0]
        start = label_offsets[label]
        for offset, index in enumerate(indices):
            assignments[(start + offset) % folds].append(int(index))
        label_offsets[label] = (start + len(indices)) % folds
    universe = set(range(len(labels)))
    return [
        (
            np.asarray(sorted(universe - set(validation)), dtype=int),
            np.asarray(sorted(validation), dtype=int),
        )
        for validation in assignments
    ]


def fold_constant_mask(training: np.ndarray) -> np.ndarray:
    if training.ndim != 2:
        raise ValueError("Training matrix must be two-dimensional")
    return np.ptp(training, axis=0) > 1e-12


def reject_duplicate_features(matrix: np.ndarray, names: tuple[str, ...]) -> None:
    duplicates = []
    for left in range(matrix.shape[1]):
        for right in range(left + 1, matrix.shape[1]):
            if np.array_equal(matrix[:, left], matrix[:, right]):
                duplicates.append((names[left], names[right]))
    if duplicates:
        raise ValueError(f"Exact duplicate primary features: {duplicates}")


def fold_residuals(
    training_confound: np.ndarray,
    validation_confound: np.ndarray,
    training_ifi: np.ndarray,
    validation_ifi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    confound_scaler = StandardScaler()
    scaled_training = confound_scaler.fit_transform(training_confound)
    scaled_validation = confound_scaler.transform(validation_confound)
    model = Ridge(alpha=1.0)
    model.fit(scaled_training, training_ifi)
    training_prediction = model.predict(scaled_training)
    validation_prediction = model.predict(scaled_validation)
    if training_ifi.ndim == 2 and training_prediction.ndim == 1:
        training_prediction = training_prediction[:, None]
        validation_prediction = validation_prediction[:, None]
    return (
        training_ifi - training_prediction,
        validation_ifi - validation_prediction,
    )


def _nested_regularization(
    matrix: np.ndarray, labels: np.ndarray, *, seed: int
) -> float:
    counts = Counter(labels.tolist())
    inner_folds = min(3, min(counts.values()))
    if inner_folds < 2:
        return 1.0
    splitter = StratifiedKFold(
        n_splits=inner_folds, shuffle=True, random_state=seed
    )
    results = {}
    for regularization in REGULARIZATION_VALUES:
        scores = np.full(len(labels), np.nan)
        for train, validation in splitter.split(matrix, labels):
            mask = fold_constant_mask(matrix[train])
            if not mask.any():
                continue
            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(matrix[train][:, mask])
            scaled_validation = scaler.transform(matrix[validation][:, mask])
            model = LogisticRegression(
                C=regularization,
                class_weight="balanced",
                random_state=seed,
                max_iter=2000,
            )
            model.fit(scaled_train, labels[train])
            scores[validation] = model.predict_proba(scaled_validation)[:, 1]
        valid = np.isfinite(scores)
        results[regularization] = (
            roc_auc_score(labels[valid], scores[valid])
            if valid.any() and len(np.unique(labels[valid])) == 2
            else -math.inf
        )
    return max(REGULARIZATION_VALUES, key=lambda value: (results[value], -value))


def _fit_fold(
    training: np.ndarray,
    validation: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    *,
    seed: int,
    validation_labels: np.ndarray | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    mask = fold_constant_mask(training)
    detail = {
        "removed_constant_features": int((~mask).sum()),
        "active_feature_indices": np.flatnonzero(mask).astype(int).tolist(),
        "training_class_counts": dict(
            sorted(Counter(labels[train_index].tolist()).items())
        ),
        "validation_class_counts": dict(
            sorted(
                Counter(
                    (
                        labels[validation_index]
                        if validation_labels is None
                        else validation_labels
                    ).tolist()
                ).items()
            )
        ),
    }
    if not mask.any() or len(detail["training_class_counts"]) < 2:
        return None, detail
    regularization = _nested_regularization(
        training[:, mask], labels[train_index], seed=seed
    )
    scaler = StandardScaler()
    scaled_training = scaler.fit_transform(training[:, mask])
    scaled_validation = scaler.transform(validation[:, mask])
    model = LogisticRegression(
        C=regularization,
        class_weight="balanced",
        random_state=seed,
        max_iter=2000,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaled_training, labels[train_index])
    detail.update(
        {
            "regularization": regularization,
            "convergence_warning_count": sum(
                issubclass(item.category, ConvergenceWarning) for item in caught
            ),
            "standardized_coefficients": model.coef_[0].astype(float).tolist(),
        }
    )
    return model.predict_proba(scaled_validation)[:, 1], detail


def _aurc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    risks = np.cumsum(labels[order]) / np.arange(1, len(labels) + 1)
    coverage = np.arange(1, len(labels) + 1) / len(labels)
    return float(np.trapezoid(risks, coverage))


def _metrics(labels: np.ndarray, scores: np.ndarray, folds: list[Any]) -> dict[str, Any]:
    valid = np.isfinite(scores)
    valid_labels = labels[valid]
    valid_scores = scores[valid]
    counts = Counter(valid_labels.tolist())
    result = {
        "sample_count": int(valid.sum()),
        "correct_count": counts.get(0, 0),
        "incorrect_count": counts.get(1, 0),
        "minority_class_count": min(counts.values(), default=0),
        "valid_fold_count": sum(item["defined"] for item in folds),
        "undefined_fold_count": sum(not item["defined"] for item in folds),
        "reliability_status": reliability_label(min(counts.values(), default=0)),
        "folds": folds,
    }
    if len(counts) < 2:
        return {**result, "auroc": None, "auprc": None, "aurc": None}
    result.update(
        {
            "auroc": float(roc_auc_score(valid_labels, valid_scores)),
            "auprc": float(average_precision_score(valid_labels, valid_scores)),
            "aurc": _aurc(valid_labels, valid_scores),
        }
    )
    rng = np.random.default_rng(2026)
    bootstrap = []
    for _ in range(1000):
        indices = rng.integers(0, len(valid_labels), len(valid_labels))
        if len(np.unique(valid_labels[indices])) < 2:
            continue
        bootstrap.append(
            roc_auc_score(valid_labels[indices], valid_scores[indices])
        )
    result["auroc_bootstrap_95_ci"] = [
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    ]
    return result


def _feature_arrays(
    predictions: list[dict[str, Any]],
    signatures: dict[str, dict[str, Any]],
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    usable = [
        item
        for item in predictions
        if signatures[item["example_id"]]["compact_ifi"]["scalar_ifi"] is not None
        and item["binary_error"] is not None
    ]
    identifiers = [item["example_id"] for item in usable]
    arrays = {
        "P": np.asarray(
            [
                [
                    signatures[item["example_id"]]["probability"][name]
                    for name in COMPACT_PROBABILITY_NAMES
                ]
                for item in usable
            ],
            dtype=float,
        ),
        "L": np.asarray(
            [
                [
                    float(item["prompt_token_count"]),
                    float(item["generated_token_count"]),
                    float(item["response_character_count"]),
                    float(item["token_limit_reached"]),
                ]
                for item in usable
            ]
        ),
        "I": np.asarray(
            [
                [
                    signatures[item["example_id"]]["compact_ifi"][
                        "scalar_ifi"
                    ]
                ]
                for item in usable
            ]
        ),
        "C": np.asarray(
            [
                [
                    signatures[item["example_id"]]["compact_ifi"][name]
                    for name in COMPACT_IFI_NAMES
                ]
                for item in usable
            ]
        ),
    }
    labels = np.asarray([item["binary_error"] for item in usable], dtype=int)
    return arrays, labels, identifiers


def comparison_predictions(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    comparison: tuple[str, ...],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    scores = np.full(len(labels), np.nan)
    fold_details = []
    for fold, (train, validation) in enumerate(splits):
        confounds_train = np.column_stack([arrays["P"][train], arrays["L"][train]])
        confounds_validation = np.column_stack(
            [arrays["P"][validation], arrays["L"][validation]]
        )
        parts_train = []
        parts_validation = []
        for family in comparison:
            if family.endswith("_RESIDUAL"):
                base = family.removesuffix("_RESIDUAL")
                residual_train, residual_validation = fold_residuals(
                    confounds_train,
                    confounds_validation,
                    arrays[base][train],
                    arrays[base][validation],
                )
                parts_train.append(residual_train)
                parts_validation.append(residual_validation)
            else:
                parts_train.append(arrays[family][train])
                parts_validation.append(arrays[family][validation])
        training = np.column_stack(parts_train)
        validation_matrix = np.column_stack(parts_validation)
        predicted, detail = _fit_fold(
            training,
            validation_matrix,
            labels,
            train,
            validation,
            seed=2026 + fold,
        )
        detail.update({"fold": fold, "defined": predicted is not None})
        fold_details.append(detail)
        if predicted is not None:
            scores[validation] = predicted
    return scores, fold_details


def analyze_collection(
    destination: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    subset: str | None = None,
) -> dict[str, Any]:
    predictions, signatures = _load_collection(destination)
    if subset is not None:
        expected = subset == "answerable"
        predictions = [
            item
            for item in predictions
            if item["evaluation_metrics"].get("answerable") is expected
        ]
    arrays, labels, identifiers = _feature_arrays(predictions, signatures)
    reject_duplicate_features(arrays["P"], COMPACT_PROBABILITY_NAMES)
    reject_duplicate_features(arrays["C"], COMPACT_IFI_NAMES)
    manifest = {item["example_id"]: item for item in _read_jsonl(manifest_path)}
    strata = [manifest[identifier]["sampling_stratum"] for identifier in identifiers]
    minority = min(Counter(labels.tolist()).values())
    folds = min(5, minority)
    if folds < 2:
        result = {
            "sample_count": len(labels),
            "reliability_status": reliability_label(minority),
            "reason": "class_deficient",
            "comparisons": {},
        }
        _atomic_json(output_path, result)
        return result
    splits = balanced_splits(labels, strata, folds=folds, seed=2026)
    split_checksum = hashlib.sha256(
        canonical_json(
            [
                [train.tolist(), validation.tolist()]
                for train, validation in splits
            ]
        ).encode()
    ).hexdigest()
    results = {}
    for name, comparison in COMPARISONS.items():
        scores, details = comparison_predictions(
            arrays, labels, splits, comparison
        )
        result = _metrics(labels, scores, details)
        repeated = []
        for seed in range(2026, 2036):
            repeated_splits = balanced_splits(
                labels, strata, folds=folds, seed=seed
            )
            repeated_scores, repeated_details = comparison_predictions(
                arrays, labels, repeated_splits, comparison
            )
            repeated_result = _metrics(
                labels, repeated_scores, repeated_details
            )
            if repeated_result["auroc"] is not None:
                repeated.append(repeated_result["auroc"])
        result["repeated_split_auroc_range"] = (
            [min(repeated), max(repeated)] if repeated else None
        )
        result["feature_count"] = sum(
            arrays[
                "I"
                if family == "I_RESIDUAL"
                else "C"
                if family == "C_RESIDUAL"
                else family
            ].shape[1]
            for family in comparison
        )
        assembled = np.column_stack(
            [
                arrays[
                    "I"
                    if family == "I_RESIDUAL"
                    else "C"
                    if family == "C_RESIDUAL"
                    else family
                ]
                for family in comparison
            ]
        )
        result["feature_matrix_rank"] = int(np.linalg.matrix_rank(assembled))
        result["rank_deficient"] = (
            result["feature_matrix_rank"] < result["feature_count"]
        )
        results[name] = result
    report = {
        "split_checksum": split_checksum,
        "positive_class": "binary_error",
        "score_orientation": "higher_is_greater_error_risk",
        "comparisons": results,
    }
    _atomic_json(output_path, report)
    return report


def transfer_analysis(
    source_destination: Path,
    shift_destinations: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    source_predictions, source_signatures = _load_collection(source_destination)
    source_arrays, source_labels, _ = _feature_arrays(
        source_predictions, source_signatures
    )
    feature_sets = {
        "probability": ("P",),
        "length": ("L",),
        "raw_scalar_ifi": ("I",),
        "compact_ifi": ("C",),
        "probability_length_scalar_ifi": ("P", "L", "I"),
        "probability_length_compact_ifi": ("P", "L", "C"),
        "probability_length_residual_scalar_ifi": ("P", "L", "I_RESIDUAL"),
        "probability_length_residual_compact_ifi": ("P", "L", "C_RESIDUAL"),
    }
    output = {"source_sample_count": len(source_labels), "shifts": {}}
    for shift_destination in shift_destinations:
        shift_predictions, shift_signatures = _load_collection(shift_destination)
        shift_arrays, shift_labels, _ = _feature_arrays(
            shift_predictions, shift_signatures
        )
        shift_results = {}
        for name, comparison in feature_sets.items():
            train = np.arange(len(source_labels))
            validation = np.arange(len(shift_labels))
            confound_train = np.column_stack(
                [source_arrays["P"], source_arrays["L"]]
            )
            confound_validation = np.column_stack(
                [shift_arrays["P"], shift_arrays["L"]]
            )
            train_parts = []
            validation_parts = []
            for family in comparison:
                if family in {"I_RESIDUAL", "C_RESIDUAL"}:
                    base = "I" if family == "I_RESIDUAL" else "C"
                    residual_train, residual_validation = fold_residuals(
                        confound_train,
                        confound_validation,
                        source_arrays[base],
                        shift_arrays[base],
                    )
                    train_parts.append(residual_train)
                    validation_parts.append(residual_validation)
                else:
                    train_parts.append(source_arrays[family])
                    validation_parts.append(shift_arrays[family])
            predicted, detail = _fit_fold(
                np.column_stack(train_parts),
                np.column_stack(validation_parts),
                source_labels,
                train,
                validation,
                seed=2026,
                validation_labels=shift_labels,
            )
            if predicted is None:
                shift_results[name] = {"auroc": None, "reason": "undefined_fit"}
            else:
                pseudo_folds = [{**detail, "defined": True}]
                shift_results[name] = _metrics(
                    shift_labels, predicted, pseudo_folds
                )
        output["shifts"][shift_destination.name] = shift_results
    _atomic_json(output_path, output)
    return output


def feature_diagnostics(
    destination: Path, output_path: Path
) -> dict[str, Any]:
    predictions, signatures = _load_collection(destination)
    rows = []
    vectors: dict[str, list[float]] = defaultdict(list)
    metadata = {
        "prompt_length": [item["prompt_token_count"] for item in predictions],
        "generated_length": [item["generated_token_count"] for item in predictions],
        "response_characters": [
            item["response_character_count"] for item in predictions
        ],
        "probability_uncertainty": [
            signatures[item["example_id"]]["probability"][
                "negative_mean_log_probability"
            ]
            for item in predictions
        ],
        "error": [
            item["binary_error"]
            if item["binary_error"] is not None
            else math.nan
            for item in predictions
        ],
    }
    for item in predictions:
        signature = signatures[item["example_id"]]
        for family in ("probability", "compact_ifi"):
            for name, value in signature[family].items():
                if value is not None:
                    vectors[f"{family}.{name}"].append(float(value))
    for name, values in sorted(vectors.items()):
        valid_count = len(values)
        row = {
            "feature": name,
            "valid_count": valid_count,
            "missing_count": len(predictions) - valid_count,
            "minimum": min(values),
            "maximum": max(values),
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
            "unique_count": len(set(values)),
            "constant": len(set(values)) == 1,
            "near_constant": len(set(values)) / valid_count <= 0.02,
        }
        if valid_count == len(predictions):
            for metadata_name, metadata_values in metadata.items():
                valid = [
                    (value, other)
                    for value, other in zip(values, metadata_values)
                    if math.isfinite(other)
                ]
                correlation = (
                    spearmanr(
                        [item[0] for item in valid],
                        [item[1] for item in valid],
                    ).statistic
                    if len(valid) >= 3
                    and len({item[0] for item in valid}) > 1
                    and len({item[1] for item in valid}) > 1
                    else None
                )
                row[f"spearman_{metadata_name}"] = (
                    float(correlation)
                    if correlation is not None and math.isfinite(correlation)
                    else None
                )
        rows.append(row)
    matrix = np.column_stack([vectors[name] for name in sorted(vectors)])
    hashes = defaultdict(list)
    for index, name in enumerate(sorted(vectors)):
        hashes[hashlib.sha256(matrix[:, index].tobytes()).hexdigest()].append(name)
    report = {
        "features": rows,
        "duplicate_features": [
            names for names in hashes.values() if len(names) > 1
        ],
        "matrix_rank": int(np.linalg.matrix_rank(matrix)),
        "feature_count": matrix.shape[1],
        "sample_count": matrix.shape[0],
        "sample_to_feature_ratio": matrix.shape[0] / matrix.shape[1],
    }
    _atomic_json(output_path, report)
    return report


def arithmetic_protocols(
    destination: Path, manifest_path: Path, output_path: Path
) -> dict[str, Any]:
    predictions, signatures = _load_collection(destination)
    arrays, labels, identifiers = _feature_arrays(predictions, signatures)
    manifest = {item["example_id"]: item for item in _read_jsonl(manifest_path)}
    operations = [
        next(
            part.split(":", 1)[1]
            for part in manifest[identifier]["sampling_stratum"].split("|")
            if part.startswith("operation:")
        )
        for identifier in identifiers
    ]
    seeds = [
        next(
            part.split(":", 1)[1]
            for part in manifest[identifier]["sampling_stratum"].split("|")
            if part.startswith("seed:")
        )
        for identifier in identifiers
    ]
    folds = min(5, min(Counter(labels.tolist()).values()))
    shared_splits = balanced_splits(
        labels,
        [f"operation:{operation}|seed:{seed}" for operation, seed in zip(operations, seeds)],
        folds=folds,
        seed=2026,
    )

    def evaluate(
        local_arrays: dict[str, np.ndarray],
        local_labels: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]],
        comparison: tuple[str, ...],
    ) -> dict[str, Any]:
        scores, details = comparison_predictions(
            local_arrays, local_labels, splits, comparison
        )
        return _metrics(local_labels, scores, details)

    operation_names = sorted(set(operations))
    operation_one_hot = np.asarray(
        [
            [float(operation == candidate) for candidate in operation_names[:-1]]
            for operation in operations
        ]
    )
    controlled_arrays = {**arrays, "O": operation_one_hot}
    report: dict[str, Any] = {
        "split_checksum": hashlib.sha256(
            canonical_json(
                [
                    [train.tolist(), validation.tolist()]
                    for train, validation in shared_splits
                ]
            ).encode()
        ).hexdigest(),
        "operation_controlled": {
            "probability_length": evaluate(
                controlled_arrays, labels, shared_splits, ("P", "L")
            ),
            "probability_length_operation": evaluate(
                controlled_arrays, labels, shared_splits, ("P", "L", "O")
            ),
            "probability_length_operation_compact_ifi": evaluate(
                controlled_arrays,
                labels,
                shared_splits,
                ("P", "L", "O", "C"),
            ),
        },
        "operation_specific": {},
        "leave_one_operation_out": {},
    }
    for operation in operation_names:
        indices = np.asarray(
            [index for index, value in enumerate(operations) if value == operation]
        )
        local_labels = labels[indices]
        local_arrays = {name: value[indices] for name, value in arrays.items()}
        minority = min(Counter(local_labels.tolist()).values(), default=0)
        local_folds = min(5, minority)
        if local_folds < 2:
            report["operation_specific"][operation] = {
                "reason": "class_deficient",
                "reliability_status": reliability_label(minority),
            }
        else:
            local_splits = balanced_splits(
                local_labels,
                [seeds[index] for index in indices],
                folds=local_folds,
                seed=2026,
            )
            report["operation_specific"][operation] = {
                "probability_length": evaluate(
                    local_arrays, local_labels, local_splits, ("P", "L")
                ),
                "probability_length_compact_ifi": evaluate(
                    local_arrays,
                    local_labels,
                    local_splits,
                    ("P", "L", "C"),
                ),
            }
        validation = indices
        train = np.asarray(
            [index for index in range(len(labels)) if index not in set(validation)]
        )
        held_out = {}
        for name, comparison in {
            "probability_length": ("P", "L"),
            "probability_length_compact_ifi": ("P", "L", "C"),
        }.items():
            scores, details = comparison_predictions(
                arrays, labels, [(train, validation)], comparison
            )
            held_out[name] = _metrics(labels, scores, details)
        report["leave_one_operation_out"][operation] = held_out
    _atomic_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact uncertainty analysis.")
    parser.add_argument(
        "action",
        choices=(
            "gsm8k-decision",
            "compact",
            "residualized",
            "arithmetic",
            "transfer",
            "features",
            "lexical",
        ),
    )
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-destination", type=Path)
    parser.add_argument("--shift-destination", type=Path, action="append", default=[])
    parser.add_argument("--subset", choices=("answerable", "unanswerable"))
    args = parser.parse_args()
    if args.action == "gsm8k-decision":
        result = write_gsm8k_decision(args.destination)
    elif args.action in {"compact", "residualized"}:
        result = analyze_collection(
            args.destination,
            args.manifest,
            args.output,
            subset=args.subset,
        )
    elif args.action == "arithmetic":
        result = arithmetic_protocols(
            args.destination, args.manifest, args.output
        )
    elif args.action == "transfer":
        result = transfer_analysis(
            args.source_destination, args.shift_destination, args.output
        )
    elif args.action == "features":
        result = feature_diagnostics(args.destination, args.output)
    else:
        result = lexical_diagnostic_summary(args.destination, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
