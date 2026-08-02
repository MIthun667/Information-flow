from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from usig.data.normalization.text import normalize_answer
from usig.evaluation.audit_rules import (
    concise_alias_match,
    conservative_abstention,
    containment_diagnostics,
    evaluate_interpretation_segments,
)
from usig.experiment.analysis import PROBABILITY_FEATURES
from usig.experiment.collection import (
    EXPECTED_MANIFEST_CHECKSUM,
    validate_collection,
)
from usig.experiment.records import (
    atomic_json,
    canonical_json,
    checksum_record,
    validate_record_checksum,
)

EXPERIMENT_ID = "qwen_ifi_66b0032f646fc519"
EXPECTED_PREDICTION_CHECKSUM = (
    "ee131679054b616852d8db5de67d2c36109a0d1a0783e613f7a17f15b6829769"
)
EXPECTED_SIGNATURE_CHECKSUM = (
    "7f6050271d1e2d1136783163a44bba0b02c29bb88526dd2b7964cab9db435f9f"
)
EVALUATOR_CONFIGURATION = {
    "normalization": "existing_nfkc_lower_punctuation_articles_whitespace",
    "concise_alias_extra_token_limit": 8,
    "concise_alias_position": "response_suffix_only",
    "parenthetical_handling": True,
    "ambignq_segmentation": "newline_semicolon_bullet",
    "ambignq_interpretation_isolation": True,
    "ambignq_numbered_list_alignment": "one_based_position",
    "squad_abstention": "explicit_conservative_phrase_list",
    "truthfulqa_status": "diagnostic_only",
}
ANALYSIS_CONFIGURATION = {
    "seed": 2026,
    "folds": "min(5, minority_count), at least 3",
    "positive_class": "binary_error",
    "score_direction": "higher_is_greater_error_risk",
    "bootstrap_draws": 1000,
    "identical_split_reuse": True,
    "gsm8k_leave_one_out": True,
    "near_constant_unique_fraction": 0.02,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    value = spearmanr(left, right).statistic
    return None if not math.isfinite(value) else float(value)


def _feature_families(
    prediction: dict[str, Any], signature: dict[str, Any]
) -> dict[str, dict[str, float]]:
    structured = signature["signature"]
    cosine = structured["cosine_structured"]
    relative = structured["relative_structured"]
    layer_regions = {
        name: float(value)
        for name, value in cosine.items()
        if "fixed_depth_profile" not in name and isinstance(value, (int, float))
    }
    fixed_depth = {
        f"cosine_fixed_depth_{index:02d}": float(value)
        for index, value in enumerate(cosine["cosine_fixed_depth_profile"])
    }
    relative_features: dict[str, float] = {}
    for name, value in relative.items():
        if isinstance(value, list):
            relative_features.update(
                {
                    f"{name}_{index:02d}": float(item)
                    for index, item in enumerate(value)
                }
            )
        elif isinstance(value, (int, float)):
            relative_features[name] = float(value)
    return {
        "probability": {
            name: float(signature["probability_summaries"][name])
            for name in PROBABILITY_FEATURES
        },
        "scalar_ifi": (
            {}
            if signature["scalar_ifi"] is None
            else {"scalar_ifi": float(signature["scalar_ifi"])}
        ),
        "token_dynamics": {
            name: float(value)
            for name, value in structured["cosine_token_dynamics"].items()
            if isinstance(value, (int, float))
        },
        "layer_regions": layer_regions,
        "fixed_depth_profile": fixed_depth,
        "relative_transitions": relative_features,
        "length_generation_metadata": {
            "prompt_token_count": float(prediction["prompt_token_count"]),
            "generated_token_count": float(prediction["generated_token_count"]),
            "response_character_length": float(len(prediction["response"])),
            "token_limit_status": float(prediction["token_limit_reached"]),
        },
    }


def _integrity(
    root: Path,
    predictions: list[dict[str, Any]],
    signatures: list[dict[str, Any]],
    canonical: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    prediction_path = root / "outputs/predictions" / f"{EXPERIMENT_ID}.jsonl"
    signature_path = root / "outputs/signatures" / f"{EXPERIMENT_ID}.jsonl"
    prediction_ids = [item["example_id"] for item in predictions]
    signature_ids = [item["example_id"] for item in signatures]
    prompt_failures = [
        item["example_id"]
        for item in predictions
        if hashlib.sha256(item["rendered_prompt"].encode()).hexdigest()
        != item["prompt_checksum"]
    ]
    result = {
        "passed": True,
        "prediction_checksum": _sha256(prediction_path),
        "signature_checksum": _sha256(signature_path),
        "expected_prediction_checksum": EXPECTED_PREDICTION_CHECKSUM,
        "expected_signature_checksum": EXPECTED_SIGNATURE_CHECKSUM,
        "manifest_checksum": EXPECTED_MANIFEST_CHECKSUM,
        "prediction_count": len(predictions),
        "signature_count": len(signatures),
        "prediction_dataset_counts": dict(
            sorted(Counter(item["dataset"] for item in predictions).items())
        ),
        "signature_dataset_counts": dict(
            sorted(Counter(item["dataset"] for item in signatures).items())
        ),
        "prediction_duplicate_identifier_count": len(prediction_ids)
        - len(set(prediction_ids)),
        "signature_duplicate_identifier_count": len(signature_ids)
        - len(set(signature_ids)),
        "prediction_unique_group_count": len(
            {item["group_id"] for item in predictions}
        ),
        "prediction_signature_missing": sorted(set(prediction_ids) - set(signature_ids)),
        "signature_prediction_missing": sorted(set(signature_ids) - set(prediction_ids)),
        "canonical_missing": sorted(set(prediction_ids) - set(canonical)),
        "experiment_identifier_failures": sum(
            item["experiment_id"] != EXPERIMENT_ID for item in predictions + signatures
        ),
        "prediction_record_checksum_failures": sum(
            not validate_record_checksum(item, "record_checksum")
            for item in predictions
        ),
        "signature_record_checksum_failures": sum(
            not validate_record_checksum(item, "signature_checksum")
            for item in signatures
        ),
        "prompt_checksum_failures": prompt_failures,
    }
    result["passed"] = (
        result["prediction_checksum"] == EXPECTED_PREDICTION_CHECKSUM
        and result["signature_checksum"] == EXPECTED_SIGNATURE_CHECKSUM
        and len(predictions) == len(signatures) == 600
        and set(result["prediction_dataset_counts"].values()) == {100}
        and set(result["signature_dataset_counts"].values()) == {100}
        and not any(
            (
                result["prediction_duplicate_identifier_count"],
                result["signature_duplicate_identifier_count"],
                result["prediction_signature_missing"],
                result["signature_prediction_missing"],
                result["canonical_missing"],
                result["experiment_identifier_failures"],
                result["prediction_record_checksum_failures"],
                result["signature_record_checksum_failures"],
                result["prompt_checksum_failures"],
            )
        )
    )
    return result


def _response_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    responses = [item["response"] for item in records]
    normalized = [item["normalized_response"] or "" for item in records]
    return {
        "record_count": len(records),
        "response_character_length": _summary_numbers([len(item) for item in responses]),
        "normalized_response_character_length": _summary_numbers(
            [len(item) for item in normalized]
        ),
        "generated_token_count": _summary_numbers(
            [item["generated_token_count"] for item in records]
        ),
        "token_limit_count": sum(item["token_limit_reached"] for item in records),
        "empty_response_count": sum(not item.strip() for item in responses),
        "repeated_response_record_count": sum(
            count for count in Counter(responses).values() if count > 1
        ),
        "most_common_responses": Counter(responses).most_common(10),
        "parsing_status": dict(
            sorted(
                Counter(
                    item["evaluation_metrics"].get("parsing_status", "not_applicable")
                    for item in records
                ).items()
            )
        ),
        "correctness": dict(
            sorted(Counter(str(item["binary_correctness"]) for item in records).items())
        ),
        "unresolved_labels": dict(
            sorted(Counter(str(item["unresolved_label"]) for item in records).items())
        ),
    }


def _summary_numbers(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def _revised_evaluation(
    prediction: dict[str, Any], canonical: dict[str, Any]
) -> dict[str, Any]:
    dataset = prediction["dataset"]
    response = prediction["response"]
    original = prediction["binary_correctness"]
    if dataset == "triviaqa":
        details = concise_alias_match(response, canonical["reference_answers"])
        revised = bool(details["match"])
        status = details["rule"] or (
            "token_overlap" if details["maximum_token_f1"] > 0 else "clearly_incorrect"
        )
    elif dataset == "ambignq":
        interpretations = canonical["interpretations"] or [
            {
                "interpretation_id": "single_answer",
                "reference_answers": canonical["reference_answers"],
            }
        ]
        details = evaluate_interpretation_segments(response, interpretations)
        revised = bool(details["any_interpretation_match"])
        status = (
            "multiple_interpretations"
            if details["multiple_interpretations_covered"]
            else "one_interpretation"
            if details["any_interpretation_match"]
            else "high_overlap_unresolved"
            if details["any_interpretation_token_f1"] >= 0.5
            else "clearly_incorrect"
        )
    elif dataset == "squad" and not canonical["answerable"]:
        details = conservative_abstention(response)
        revised = bool(details["abstained"])
        status = details["rule"] or "unsupported_answer"
    elif dataset == "truthfulqa":
        correct = containment_diagnostics(response, canonical["reference_answers"])
        incorrect = containment_diagnostics(
            response, canonical["incorrect_reference_answers"]
        )
        difference = correct["maximum_token_f1"] - incorrect["maximum_token_f1"]
        if correct["normalized_exact_match"] and incorrect["normalized_exact_match"]:
            status = "both"
        elif correct["normalized_exact_match"]:
            status = "correct_reference_leaning"
        elif incorrect["normalized_exact_match"]:
            status = "incorrect_reference_leaning"
        elif abs(difference) < 0.05:
            status = "neither"
        elif difference > 0:
            status = "correct_reference_leaning"
        else:
            status = "incorrect_reference_leaning"
        details = {
            "correct_references": correct,
            "incorrect_references": incorrect,
            "relative_lexical_similarity": difference,
        }
        revised = None
    else:
        details = prediction["evaluation_metrics"]
        revised = original
        status = "unchanged"
    record = {
        "experiment_id": EXPERIMENT_ID,
        "example_id": prediction["example_id"],
        "dataset": dataset,
        "original_binary_correctness": original,
        "audit_binary_correctness": revised,
        "diagnostic_status": status,
        "diagnostics": details,
    }
    record["evaluation_checksum"] = checksum_record(record, "evaluation_checksum")
    return record


def _feature_audit(
    predictions: list[dict[str, Any]],
    signatures: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, float]]]]:
    maps = {
        item["example_id"]: _feature_families(item, signatures[item["example_id"]])
        for item in predictions
    }
    rows = []
    vectors: dict[str, list[float]] = defaultdict(list)
    token_counts = [float(item["generated_token_count"]) for item in predictions]
    errors = [
        float(item["binary_error"])
        if item["binary_error"] is not None
        else math.nan
        for item in predictions
    ]
    for example_id, families in maps.items():
        del example_id
        for family, features in families.items():
            for name, value in features.items():
                vectors[f"{family}.{name}"].append(value)
    for qualified_name, values in sorted(vectors.items()):
        family, name = qualified_name.split(".", 1)
        valid_error = [
            (value, error)
            for value, error in zip(values, errors)
            if math.isfinite(error)
        ]
        unique = len(set(values))
        rows.append(
            {
                "family": family,
                "feature": name,
                "valid_count": len(values),
                "missing_count": len(predictions) - len(values),
                "minimum": min(values),
                "maximum": max(values),
                "mean": statistics.mean(values),
                "standard_deviation": statistics.pstdev(values),
                "unique_count": unique,
                "constant": unique == 1,
                "near_constant": unique / len(values) <= 0.02,
                "generated_token_spearman": _spearman(values, token_counts),
                "error_label_spearman": _spearman(
                    [item[0] for item in valid_error],
                    [item[1] for item in valid_error],
                ),
            }
        )
    duplicate_groups = defaultdict(list)
    for name, values in vectors.items():
        duplicate_groups[hashlib.sha256(canonical_json(values).encode()).hexdigest()].append(
            name
        )
    return {
        "features": rows,
        "constant_features": [
            f"{item['family']}.{item['feature']}" for item in rows if item["constant"]
        ],
        "near_constant_features": [
            f"{item['family']}.{item['feature']}"
            for item in rows
            if item["near_constant"]
        ],
        "duplicate_feature_groups": [
            group for group in duplicate_groups.values() if len(group) > 1
        ],
    }, maps


def _matrix(
    identifiers: list[str],
    maps: dict[str, dict[str, dict[str, float]]],
    families: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    names = sorted(
        {
            f"{family}.{name}"
            for identifier in identifiers
            for family in families
            for name in maps[identifier][family]
        }
    )
    matrix = np.asarray(
        [
            [
                maps[identifier][family][name]
                for family, name in (qualified.split(".", 1) for qualified in names)
            ]
            for identifier in identifiers
        ],
        dtype=float,
    )
    return matrix, names


def _fit_splits(
    matrix: np.ndarray,
    labels: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    scores = np.full(len(labels), np.nan)
    fold_details = []
    undefined = 0
    convergence_warnings = 0
    for fold, (train, validation) in enumerate(splits):
        train_counts = Counter(labels[train].tolist())
        validation_counts = Counter(labels[validation].tolist())
        if len(train_counts) < 2:
            undefined += 1
            continue
        scaler = StandardScaler()
        train_matrix = scaler.fit_transform(matrix[train])
        validation_matrix = scaler.transform(matrix[validation])
        model = LogisticRegression(
            class_weight="balanced", random_state=2026, max_iter=2000
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_matrix, labels[train])
        convergence_warnings += sum(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        scores[validation] = model.predict_proba(validation_matrix)[:, 1]
        fold_details.append(
            {
                "fold": fold,
                "training_class_counts": dict(sorted(train_counts.items())),
                "validation_class_counts": dict(sorted(validation_counts.items())),
                "coefficient_l2_norm": float(np.linalg.norm(model.coef_)),
            }
        )
    valid = np.isfinite(scores)
    class_counts = Counter(labels[valid].tolist())
    result: dict[str, Any] = {
        "sample_count": int(valid.sum()),
        "positive_class_count": int(class_counts.get(1, 0)),
        "negative_class_count": int(class_counts.get(0, 0)),
        "valid_folds": len(fold_details),
        "undefined_folds": undefined,
        "convergence_warning_count": convergence_warnings,
        "folds": fold_details,
        "positive_class": "binary_error",
        "score_direction": "higher_is_greater_error_risk",
    }
    if len(class_counts) == 2:
        result["auroc"] = float(roc_auc_score(labels[valid], scores[valid]))
        result["auprc"] = float(average_precision_score(labels[valid], scores[valid]))
        rng = np.random.default_rng(2026)
        bootstrap_auroc = []
        bootstrap_auprc = []
        valid_labels = labels[valid]
        valid_scores = scores[valid]
        for _ in range(1000):
            indices = rng.integers(0, len(valid_labels), len(valid_labels))
            if len(np.unique(valid_labels[indices])) < 2:
                continue
            bootstrap_auroc.append(
                roc_auc_score(valid_labels[indices], valid_scores[indices])
            )
            bootstrap_auprc.append(
                average_precision_score(
                    valid_labels[indices], valid_scores[indices]
                )
            )
        result["auroc_bootstrap_95_ci"] = [
            float(np.quantile(bootstrap_auroc, 0.025)),
            float(np.quantile(bootstrap_auroc, 0.975)),
        ]
        result["auprc_bootstrap_95_ci"] = [
            float(np.quantile(bootstrap_auprc, 0.025)),
            float(np.quantile(bootstrap_auprc, 0.975)),
        ]
    else:
        result["auroc"] = None
        result["auprc"] = None
        result["auroc_bootstrap_95_ci"] = None
        result["auprc_bootstrap_95_ci"] = None
    result["exploratory"] = min(class_counts.values(), default=0) < 30
    result["small_class_warning"] = min(class_counts.values(), default=0) < 20
    result["score_minimum"] = float(np.nanmin(scores)) if valid.any() else None
    result["score_maximum"] = float(np.nanmax(scores)) if valid.any() else None
    result["scores"] = scores.tolist()
    return result


def _direct_score_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    counts = Counter(labels.tolist())
    result = {
        "sample_count": len(labels),
        "positive_class_count": counts.get(1, 0),
        "negative_class_count": counts.get(0, 0),
        "positive_class": "binary_error",
        "score_direction": "higher_is_greater_error_risk",
        "exploratory": min(counts.values(), default=0) < 30,
        "small_class_warning": min(counts.values(), default=0) < 20,
    }
    if len(counts) < 2:
        return {**result, "auroc": None, "auprc": None}
    result["auroc"] = float(roc_auc_score(labels, scores))
    result["auprc"] = float(average_precision_score(labels, scores))
    rng = np.random.default_rng(2026)
    auroc_values = []
    auprc_values = []
    for _ in range(1000):
        indices = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) < 2:
            continue
        auroc_values.append(roc_auc_score(labels[indices], scores[indices]))
        auprc_values.append(
            average_precision_score(labels[indices], scores[indices])
        )
    result["auroc_bootstrap_95_ci"] = [
        float(np.quantile(auroc_values, 0.025)),
        float(np.quantile(auroc_values, 0.975)),
    ]
    result["auprc_bootstrap_95_ci"] = [
        float(np.quantile(auprc_values, 0.025)),
        float(np.quantile(auprc_values, 0.975)),
    ]
    return result


def _stability_audit(
    predictions: list[dict[str, Any]],
    maps: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    comparisons = {
        "probability": ("probability",),
        "scalar_ifi_logistic": ("scalar_ifi",),
        "probability_plus_scalar_ifi": ("probability", "scalar_ifi"),
        "token_dynamics": ("token_dynamics",),
        "layer_regions": ("layer_regions",),
        "fixed_depth_profile": ("fixed_depth_profile",),
        "relative_transitions": ("relative_transitions",),
        "probability_plus_token_dynamics": ("probability", "token_dynamics"),
        "probability_plus_layer_regions": ("probability", "layer_regions"),
        "probability_plus_fixed_depth_profile": (
            "probability",
            "fixed_depth_profile",
        ),
        "probability_plus_relative_transitions": (
            "probability",
            "relative_transitions",
        ),
        "probability_plus_complete_structured_ifi": (
            "probability",
            "scalar_ifi",
            "token_dynamics",
            "layer_regions",
            "fixed_depth_profile",
            "relative_transitions",
        ),
    }
    output = {}
    for dataset in ("gsm8k", "ifi_arith", "squad"):
        records = [item for item in predictions if item["dataset"] == dataset]
        identifiers = [
            item["example_id"]
            for item in records
            if maps[item["example_id"]]["scalar_ifi"]
        ]
        labels = np.asarray(
            [
                next(
                    item["binary_error"]
                    for item in records
                    if item["example_id"] == identifier
                )
                for identifier in identifiers
            ]
        )
        counts = Counter(labels.tolist())
        folds = min(5, min(counts.values()))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2026)
        shared_splits = list(splitter.split(np.zeros(len(labels)), labels))
        dataset_result: dict[str, Any] = {
            "class_counts": dict(sorted(counts.items())),
            "shared_split_checksum": hashlib.sha256(
                canonical_json(
                    [
                        [train.tolist(), validation.tolist()]
                        for train, validation in shared_splits
                    ]
                ).encode()
            ).hexdigest(),
            "feature_sets": {},
        }
        scalar_matrix, _ = _matrix(identifiers, maps, ("scalar_ifi",))
        dataset_result["raw_scalar_ifi_directional"] = _direct_score_metrics(
            labels, scalar_matrix[:, 0]
        )
        for name, families in comparisons.items():
            matrix, feature_names = _matrix(identifiers, maps, families)
            result = _fit_splits(matrix, labels, shared_splits)
            scores = np.asarray(result["scores"])
            result["lowest_risk_example_ids"] = [
                identifiers[index]
                for index in np.argsort(scores)[:5]
            ]
            result["highest_risk_example_ids"] = [
                identifiers[index]
                for index in np.argsort(scores)[-5:][::-1]
            ]
            result.update(
                {
                    "feature_count": matrix.shape[1],
                    "matrix_rank": int(np.linalg.matrix_rank(matrix)),
                    "condition_number": (
                        float(np.linalg.cond(matrix))
                        if matrix.shape[0] >= matrix.shape[1]
                        else None
                    ),
                    "constant_feature_count": int(
                        (np.ptp(matrix, axis=0) == 0).sum()
                    ),
                    "feature_names": feature_names,
                }
            )
            result.pop("scores")
            if dataset == "gsm8k":
                repeated = []
                for seed in range(2026, 2036):
                    repeated_splitter = StratifiedKFold(
                        n_splits=folds, shuffle=True, random_state=seed
                    )
                    repeated_result = _fit_splits(
                        matrix,
                        labels,
                        list(
                            repeated_splitter.split(
                                np.zeros(len(labels)), labels
                            )
                        ),
                    )
                    repeated.append(repeated_result["auroc"])
                result["repeated_stratified_auroc"] = repeated
                result["repeated_stratified_auroc_range"] = [
                    min(repeated),
                    max(repeated),
                ]
            dataset_result["feature_sets"][name] = result
        if dataset == "gsm8k":
            matrix, _ = _matrix(identifiers, maps, ("probability",))
            loo = LeaveOneOut()
            loo_result = _fit_splits(matrix, labels, list(loo.split(matrix)))
            loo_result.pop("scores")
            dataset_result["leave_one_out_probability"] = loo_result
        output[dataset] = dataset_result
    return output


def _confounds(
    predictions: list[dict[str, Any]],
    signatures: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for dataset in sorted({item["dataset"] for item in predictions}):
        records = [item for item in predictions if item["dataset"] == dataset]
        usable = [
            item
            for item in records
            if signatures[item["example_id"]]["scalar_ifi"] is not None
        ]
        scalar = [
            signatures[item["example_id"]]["scalar_ifi"] for item in usable
        ]
        probability = [
            signatures[item["example_id"]]["probability_summaries"][
                "negative_mean_log_probability"
            ]
            for item in usable
        ]
        fields = {
            "prompt_token_count": [item["prompt_token_count"] for item in usable],
            "generated_token_count": [
                item["generated_token_count"] for item in usable
            ],
            "response_character_length": [len(item["response"]) for item in usable],
            "token_limit_status": [
                int(item["token_limit_reached"]) for item in usable
            ],
            "probability_uncertainty": probability,
        }
        errors = [
            item["binary_error"] for item in usable if item["binary_error"] is not None
        ]
        residual_error = None
        labelled_indices = [
            index
            for index, item in enumerate(usable)
            if item["binary_error"] is not None
        ]
        if labelled_indices:
            design = np.column_stack(
                [
                    np.ones(len(usable)),
                    fields["generated_token_count"],
                    fields["probability_uncertainty"],
                ]
            )
            coefficients = np.linalg.lstsq(
                design, np.asarray(scalar), rcond=None
            )[0]
            residuals = np.asarray(scalar) - design @ coefficients
            residual_error = _spearman(
                [float(residuals[index]) for index in labelled_indices],
                errors,
            )
        row = {
            "dataset": dataset,
            "valid_scalar_count": len(scalar),
            "scalar_associations": {
                name: _spearman(scalar, values) for name, values in fields.items()
            },
            "scalar_error_spearman": (
                _spearman(
                    [
                        value
                        for value, item in zip(scalar, usable)
                        if item["binary_error"] is not None
                    ],
                    errors,
                )
                if errors
                else None
            ),
            "residualized_scalar_error_spearman": residual_error,
            "residualization_covariates": [
                "generated_token_count",
                "negative_mean_log_probability",
            ],
            "scalar_by_token_limit": {
                str(status): (
                    statistics.mean(
                        value
                        for value, item in zip(scalar, usable)
                        if int(item["token_limit_reached"]) == status
                    )
                    if any(
                        int(item["token_limit_reached"]) == status for item in usable
                    )
                    else None
                )
                for status in (0, 1)
            },
        }
        rows.append(row)
    return {"datasets": rows}


def _inspection_rows(
    predictions: list[dict[str, Any]],
    canonical: dict[str, dict[str, Any]],
    revised: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = []
    for dataset in sorted({item["dataset"] for item in predictions}):
        records = [item for item in predictions if item["dataset"] == dataset]
        if dataset in {"ambignq", "triviaqa", "truthfulqa"}:
            chosen = records
        else:
            correct = [item for item in records if item["binary_correctness"] is True]
            incorrect = sorted(
                (item for item in records if item["binary_correctness"] is False),
                key=lambda item: hashlib.sha256(item["example_id"].encode()).hexdigest(),
            )[:20]
            chosen = correct + incorrect
        for item in chosen:
            source = canonical[item["example_id"]]
            selected.append(
                {
                    "dataset": dataset,
                    "example_id": item["example_id"],
                    "question": source["question"],
                    "response": item["response"],
                    "normalized_response": item["normalized_response"],
                    "reference_answers": source["reference_answers"],
                    "incorrect_reference_answers": source[
                        "incorrect_reference_answers"
                    ],
                    "original_evaluator_result": item["evaluation_metrics"],
                    "original_binary_correctness": item["binary_correctness"],
                    "audit_result": revised[item["example_id"]],
                }
            )
    return selected


def _subgroup_diagnostics(
    predictions: list[dict[str, Any]],
    canonical: dict[str, dict[str, Any]],
    maps: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    def grouped(records: list[dict[str, Any]], key) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[str(key(record))].append(record)
        return {
            name: {
                "count": len(items),
                "correct_count": sum(
                    item["binary_correctness"] is True for item in items
                ),
                "accuracy": statistics.mean(
                    item["binary_correctness"] is True for item in items
                ),
                "mean_scalar_ifi": statistics.mean(
                    maps[item["example_id"]]["scalar_ifi"]["scalar_ifi"]
                    for item in items
                    if maps[item["example_id"]]["scalar_ifi"]
                ),
                "mean_generated_tokens": statistics.mean(
                    item["generated_token_count"] for item in items
                ),
            }
            for name, items in sorted(groups.items())
        }

    arithmetic = [
        item for item in predictions if item["dataset"] == "ifi_arith"
    ]
    squad = [item for item in predictions if item["dataset"] == "squad"]
    operations = {
        item["example_id"]: canonical[item["example_id"]]["metadata"]["operation"]
        for item in arithmetic
    }
    leave_one_operation_out = {}
    identifiers = [item["example_id"] for item in arithmetic]
    labels = np.asarray([item["binary_error"] for item in arithmetic])
    for feature_name, families in {
        "probability": ("probability",),
        "probability_plus_complete_structured_ifi": (
            "probability",
            "scalar_ifi",
            "token_dynamics",
            "layer_regions",
            "fixed_depth_profile",
            "relative_transitions",
        ),
    }.items():
        matrix, _ = _matrix(identifiers, maps, families)
        family_results = {}
        for operation in sorted(set(operations.values())):
            validation = np.asarray(
                [
                    index
                    for index, identifier in enumerate(identifiers)
                    if operations[identifier] == operation
                ]
            )
            train = np.asarray(
                [index for index in range(len(identifiers)) if index not in validation]
            )
            result = _fit_splits(matrix, labels, [(train, validation)])
            result.pop("scores")
            family_results[operation] = result
        leave_one_operation_out[feature_name] = family_results

    answer_type = {
        item["example_id"]: (
            "unanswerable"
            if not canonical[item["example_id"]]["answerable"]
            else "answerable"
        )
        for item in squad
    }
    abstention_type = {
        item["example_id"]: (
            "explicit_abstention"
            if conservative_abstention(item["response"])["abstained"]
            else "generated_answer"
        )
        for item in squad
    }
    return {
        "ifi_arith": {
            "by_operation": grouped(
                arithmetic, lambda item: operations[item["example_id"]]
            ),
            "by_source_seed": grouped(
                arithmetic,
                lambda item: canonical[item["example_id"]]["metadata"]["seed"],
            ),
            "by_response_length_quartile": grouped(
                arithmetic,
                lambda item: min(3, item["generated_token_count"] // 8),
            ),
            "by_answer_magnitude": grouped(
                arithmetic,
                lambda item: (
                    "under_100"
                    if abs(
                        int(canonical[item["example_id"]]["reference_answers"][0])
                    )
                    < 100
                    else "100_or_more"
                ),
            ),
            "leave_one_operation_out": leave_one_operation_out,
        },
        "squad": {
            "by_answerability": grouped(
                squad, lambda item: answer_type[item["example_id"]]
            ),
            "by_abstention_behavior": grouped(
                squad, lambda item: abstention_type[item["example_id"]]
            ),
            "by_response_length_quartile": grouped(
                squad, lambda item: min(3, item["generated_token_count"] // 12)
            ),
        },
    }


def _dataset_evaluator_diagnostics(
    predictions: list[dict[str, Any]],
    canonical: dict[str, dict[str, Any]],
    revised: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    trivia = [item for item in predictions if item["dataset"] == "triviaqa"]
    trivia_categories = Counter()
    trivia_measurements = []
    for item in trivia:
        result = revised[item["example_id"]]
        details = result["diagnostics"]
        if item["evaluation_metrics"]["exact_match"]:
            category = "exact_alias_match"
        elif details["normalized_exact_match"]:
            category = "normalized_alias_match"
        elif result["audit_binary_correctness"]:
            category = "alias_in_concise_response"
        elif details["answer_containment"]:
            category = "alias_contained_diagnostic_only"
        elif details["response_containment"]:
            category = "response_contained_in_alias"
        elif details["maximum_token_f1"] >= 0.5:
            category = "token_overlap_match"
        elif details["maximum_token_f1"] < 0.2:
            category = "clearly_incorrect"
        else:
            category = "unclear"
        trivia_categories[category] += 1
        trivia_measurements.append(
            {
                "example_id": item["example_id"],
                "category": category,
                "exact_match": item["evaluation_metrics"]["exact_match"],
                "maximum_token_f1": details["maximum_token_f1"],
                "answer_containment": details["answer_containment"],
                "response_containment": details["response_containment"],
            }
        )

    ambig = [item for item in predictions if item["dataset"] == "ambignq"]
    ambig_details = [revised[item["example_id"]] for item in ambig]
    truthful = [item for item in predictions if item["dataset"] == "truthfulqa"]
    truthful_details = [revised[item["example_id"]] for item in truthful]
    squad = [item for item in predictions if item["dataset"] == "squad"]
    answerable = [
        item for item in squad if canonical[item["example_id"]]["answerable"]
    ]
    unanswerable = [
        item for item in squad if not canonical[item["example_id"]]["answerable"]
    ]
    false_abstentions = [
        item
        for item in answerable
        if conservative_abstention(item["response"])["abstained"]
    ]
    accepted_unanswerable = [
        item
        for item in unanswerable
        if conservative_abstention(item["response"])["abstained"]
    ]
    malformed_abstentions = [
        item
        for item in unanswerable
        if (
            "unknown" in normalize_answer(item["response"])
            or "unanswer" in normalize_answer(item["response"])
        )
        and not conservative_abstention(item["response"])["abstained"]
    ]
    return {
        "triviaqa": {
            "classification_counts": dict(sorted(trivia_categories.items())),
            "measurements": trivia_measurements,
            "primary_metric_recommendation": "normalized exact alias match",
            "diagnostic_metric_recommendation": [
                "maximum token F1",
                "answer containment",
                "response containment",
                "predeclared concise suffix alias match",
            ],
        },
        "ambignq": {
            "any_interpretation_exact_match_count": sum(
                item["evaluation_metrics"]["any_interpretation_exact_match"]
                for item in ambig
            ),
            "any_interpretation_segment_match_count": sum(
                item["diagnostics"]["any_interpretation_match"]
                for item in ambig_details
            ),
            "at_least_one_high_overlap_interpretation_count": sum(
                item["diagnostics"]["any_interpretation_token_f1"] >= 0.5
                for item in ambig_details
            ),
            "multiple_covered_interpretation_count": sum(
                item["diagnostics"]["multiple_interpretations_covered"]
                for item in ambig_details
            ),
            "clearly_incorrect_count": sum(
                item["diagnostic_status"] == "clearly_incorrect"
                for item in ambig_details
            ),
            "unresolved_count": sum(
                item["diagnostic_status"] == "high_overlap_unresolved"
                for item in ambig_details
            ),
            "records": ambig_details,
        },
        "truthfulqa": {
            "classification_counts": dict(
                sorted(
                    Counter(
                        item["diagnostic_status"] for item in truthful_details
                    ).items()
                )
            ),
            "records": truthful_details,
            "definitive_truthfulness_labels_assigned": False,
        },
        "squad": {
            "answerable_count": len(answerable),
            "unanswerable_count": len(unanswerable),
            "answerable_exact_match_count": sum(
                item["evaluation_metrics"]["exact_match"] for item in answerable
            ),
            "answerable_mean_token_f1": statistics.mean(
                item["evaluation_metrics"]["token_f1"] for item in answerable
            ),
            "original_unanswerable_abstention_count": sum(
                item["binary_correctness"] is True for item in unanswerable
            ),
            "conservative_unanswerable_abstention_count": len(
                accepted_unanswerable
            ),
            "false_abstention_count": len(false_abstentions),
            "unsupported_answer_count": len(unanswerable)
            - len(accepted_unanswerable),
            "malformed_abstention_count": len(malformed_abstentions),
            "false_abstention_example_ids": [
                item["example_id"] for item in false_abstentions
            ],
            "malformed_abstention_example_ids": [
                item["example_id"] for item in malformed_abstentions
            ],
        },
    }


def _write_markdown(
    root: Path,
    response_summary: dict[str, Any],
    revised: list[dict[str, Any]],
    inspections: list[dict[str, Any]],
    stability: dict[str, Any],
    confounds: dict[str, Any],
    feature_audit: dict[str, Any],
    class_balance: dict[str, Any],
    subgroup_diagnostics: dict[str, Any],
    evaluator_diagnostics: dict[str, Any],
) -> None:
    response_lines = [
        "# Response audit",
        "",
        "| Dataset | Records | Mean chars | Mean tokens | Token limits | Empty | Repeated |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, result in response_summary.items():
        response_lines.append(
            f"| {dataset} | {result['record_count']} | "
            f"{result['response_character_length']['mean']:.2f} | "
            f"{result['generated_token_count']['mean']:.2f} | "
            f"{result['token_limit_count']} | {result['empty_response_count']} | "
            f"{result['repeated_response_record_count']} |"
        )
    response_lines.extend(
        [
            "",
            (
                "The detailed inspection table is stored in "
                "`inspected_responses.json` to preserve multiline answers and "
                "references without lossy Markdown."
            ),
        ]
    )
    (root / "response_audit.md").write_text("\n".join(response_lines) + "\n")

    counts = defaultdict(Counter)
    changes = defaultdict(Counter)
    for item in revised:
        counts[item["dataset"]][item["diagnostic_status"]] += 1
        if item["original_binary_correctness"] != item["audit_binary_correctness"]:
            changes[item["dataset"]]["changed"] += 1
    evaluator_lines = [
        "# Evaluator audit",
        "",
        (
            "Revised labels are separate diagnostics. Frozen predictions and their "
            "original labels were not modified."
        ),
        "",
        "| Dataset | Diagnostic distributions | Changed binary labels |",
        "|---|---|---:|",
    ]
    for dataset in sorted(counts):
        evaluator_lines.append(
            f"| {dataset} | {dict(sorted(counts[dataset].items()))} | "
            f"{changes[dataset]['changed']} |"
        )
    (root / "evaluator_audit.md").write_text("\n".join(evaluator_lines) + "\n")

    stability_lines = [
        "# Cross-validation stability audit",
        "",
        (
            "The positive class is error and every score is oriented so larger "
            "values mean greater predicted error risk. All comparisons within a "
            "dataset reuse the identical split checksum."
        ),
        "",
        "| Dataset | Feature set | Features | AUROC | AUPRC | Minority | Exploratory |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for dataset, result in stability.items():
        for name, metrics in result["feature_sets"].items():
            minority = min(
                metrics["positive_class_count"], metrics["negative_class_count"]
            )
            stability_lines.append(
                f"| {dataset} | {name} | {metrics['feature_count']} | "
                f"{metrics['auroc']:.3f} | {metrics['auprc']:.3f} | "
                f"{minority} | {metrics['exploratory']} |"
            )
    (root / "cross_validation_stability.md").write_text(
        "\n".join(stability_lines) + "\n"
    )

    confound_lines = [
        "# Confound audit",
        "",
        (
            "| Dataset | Scalar–prompt tokens | Scalar–generated tokens | "
            "Scalar–response chars | Scalar–probability uncertainty | Scalar–error |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in confounds["datasets"]:
        association = result["scalar_associations"]

        def show(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        confound_lines.append(
            f"| {result['dataset']} | {show(association['prompt_token_count'])} | "
            f"{show(association['generated_token_count'])} | "
            f"{show(association['response_character_length'])} | "
            f"{show(association['probability_uncertainty'])} | "
            f"{show(result['scalar_error_spearman'])} |"
        )
    (root / "confound_report.md").write_text("\n".join(confound_lines) + "\n")

    feature_lines = [
        "# Feature audit",
        "",
        f"Features audited: {len(feature_audit['features'])}.",
        f"Constant features: {len(feature_audit['constant_features'])}.",
        f"Near-constant features: {len(feature_audit['near_constant_features'])}.",
        f"Exact duplicate groups: {len(feature_audit['duplicate_feature_groups'])}.",
        "",
        "Complete per-feature statistics are in `feature_audit.json`.",
    ]
    (root / "feature_audit.md").write_text("\n".join(feature_lines) + "\n")

    balance_lines = [
        "# Class-balance report",
        "",
        "| Dataset | Error labels |",
        "|---|---|",
    ]
    for dataset, counts in class_balance.items():
        balance_lines.append(f"| {dataset} | {counts} |")
    (root / "class_balance_report.md").write_text(
        "\n".join(balance_lines) + "\n"
    )

    comparison_lines = [
        "# Feature-family comparison",
        "",
        (
            "All comparisons reuse identical within-dataset splits. Results with "
            "fewer than 30 observations in the smaller class are exploratory and "
            "are not ranked."
        ),
        "",
        (
            "See `feature_family_comparison.json` for fold balances, confidence "
            "intervals, matrix diagnostics, and repeated-split results."
        ),
        "",
        (
            "IFI-ARITH operation and source-seed summaries and "
            "leave-one-operation-out diagnostics are stored in "
            "`subgroup_diagnostics.json`."
        ),
    ]
    (root / "feature_family_comparison.md").write_text(
        "\n".join(comparison_lines) + "\n"
    )

    gsm = stability["gsm8k"]["feature_sets"]
    arithmetic = stability["ifi_arith"]["feature_sets"]
    squad = stability["squad"]["feature_sets"]
    findings_lines = [
        "# Audit findings and recommendations",
        "",
        "## Evaluators",
        "",
        (
            "TriviaQA's zero strict accuracy is primarily a response-format mismatch: "
            f"{evaluator_diagnostics['triviaqa']['classification_counts'].get('alias_in_concise_response', 0)} "
            "responses end in a valid alias after a short explanatory prefix. "
            "Containment alone remains diagnostic because it also credits false "
            "relations. Normalized exact alias match should remain the primary metric."
        ),
        "",
        (
            "AmbigNQ's whole-response exact comparison cannot recognize segmented "
            f"answers. The audit found {evaluator_diagnostics['ambignq']['at_least_one_high_overlap_interpretation_count']} "
            "high-overlap records and "
            f"{evaluator_diagnostics['ambignq']['clearly_incorrect_count']} clearly "
            "incorrect records. Interpretation-isolated segmentation is suitable as "
            "a diagnostic and only deterministic aligned matches alter audit labels."
        ),
        "",
        (
            "TruthfulQA produced lexical leanings, not truthfulness judgments. Keep "
            "these diagnostic and use the benchmark's established multiple-choice "
            "protocol or preregistered human assessment in later work."
        ),
        "",
        "## Stability",
        "",
        (
            f"GSM8K has only 3 correct and 97 incorrect records. Probability-only "
            f"OOF AUROC is {gsm['probability']['auroc']:.3f}; complete structured "
            f"IFI is {gsm['probability_plus_complete_structured_ifi']['auroc']:.3f}. "
            "The complete matrix has 132 columns, rank 100, and only two correct "
            "training examples per stratified fold. Repeated-split ranges move "
            "materially. The collapse is high-dimensional small-minority instability, "
            "not a score-direction bug or convergence failure."
        ),
        "",
        (
            f"IFI-ARITH probability-only AUROC is {arithmetic['probability']['auroc']:.3f} "
            f"and complete structured IFI is "
            f"{arithmetic['probability_plus_complete_structured_ifi']['auroc']:.3f}. "
            "However, scalar IFI correlates strongly with generated length, operation "
            "accuracy ranges from 0.08 to 0.72, the complete matrix has 132 columns "
            "for 100 records, and leave-one-operation-out results vary sharply. The "
            "gain is exploratory and plausibly reflects response-length/operation "
            "structure rather than robust residual uncertainty."
        ),
        "",
        (
            f"SQuAD probability-only AUROC is {squad['probability']['auroc']:.3f} "
            f"and complete structured IFI is "
            f"{squad['probability_plus_complete_structured_ifi']['auroc']:.3f}. "
            "Only seven records are correct, just two unanswerable records explicitly "
            "abstain, and residualized scalar IFI has near-zero error association. "
            "The small gain is exploratory and cannot establish residual uncertainty."
        ),
        "",
        "## Next experiment",
        "",
        (
            "Use `Qwen/Qwen2.5-1.5B-Instruct` as the next controlled same-family "
            "model. Target at least 500 examples per balanced/evaluable dataset, "
            "1,500 for SQuAD, and 2,000 for GSM8K unless a stronger model raises "
            "its correct rate. Design for at least 30—and preferably 100—"
            "observations in the smaller class."
        ),
    ]
    (root / "findings_and_recommendations.md").write_text(
        "\n".join(findings_lines) + "\n"
    )

    change_lines = [
        "# Evaluator change log",
        "",
        "| Area | Demonstrated failure | Separate audit rule |",
        "|---|---|---|",
        (
            "| TriviaQA | Full-sentence responses fail whole-string alias equality | "
            "Short response ending in an alias, at most eight prefix tokens |"
        ),
        (
            "| Parentheticals | A parenthetical qualifier prevents equality | "
            "One comparison with a single parenthetical span removed |"
        ),
        (
            "| AmbigNQ | Lists are compared as one string | Split explicit lists, "
            "isolate interpretations, and align numbered positions |"
        ),
        (
            "| SQuAD | Explicit insufficiency phrases are rejected | Fixed conservative "
            "phrase list independent of answer content |"
        ),
        "",
        (
            "No external judge, learned threshold, pilot-label threshold tuning, or "
            "individual exception was used. Original evaluator outputs remain frozen."
        ),
    ]
    (root / "evaluator_change_log.md").write_text(
        "\n".join(change_lines) + "\n"
    )

    del inspections, subgroup_diagnostics


def run_audit(project_root: Path) -> dict[str, Any]:
    validation = validate_collection(project_root)
    prediction_path = (
        project_root / "outputs/predictions" / f"{EXPERIMENT_ID}.jsonl"
    )
    signature_path = (
        project_root / "outputs/signatures" / f"{EXPERIMENT_ID}.jsonl"
    )
    predictions = _read_jsonl(prediction_path)
    signature_records = _read_jsonl(signature_path)
    signatures = {item["example_id"]: item for item in signature_records}
    integrity = _integrity(
        project_root, predictions, signature_records, validation["records"]
    )
    if not integrity["passed"]:
        raise ValueError(f"Artifact integrity failed: {integrity}")

    evaluator_rule_source_checksum = _sha256(
        project_root / "src/usig/evaluation/audit_rules.py"
    )
    evaluator_checksum = hashlib.sha256(
        canonical_json(
            {
                "configuration": EVALUATOR_CONFIGURATION,
                "rule_source_checksum": evaluator_rule_source_checksum,
            }
        ).encode()
    ).hexdigest()
    analysis_checksum = hashlib.sha256(
        canonical_json(ANALYSIS_CONFIGURATION).encode()
    ).hexdigest()
    audit_identity = (
        "qwen_ifi_audit_"
        + hashlib.sha256(
            f"{EXPERIMENT_ID}:{evaluator_checksum}:{analysis_checksum}".encode()
        ).hexdigest()[:16]
    )
    output_root = project_root / "outputs/audits" / audit_identity

    revised = [
        _revised_evaluation(item, validation["records"][item["example_id"]])
        for item in predictions
    ]
    revised_by_id = {item["example_id"]: item for item in revised}
    response_summary = {
        dataset: _response_summary(
            [item for item in predictions if item["dataset"] == dataset]
        )
        for dataset in sorted({item["dataset"] for item in predictions})
    }
    feature_audit, maps = _feature_audit(predictions, signatures)
    stability = _stability_audit(predictions, maps)
    confounds = _confounds(predictions, signatures)
    inspections = _inspection_rows(
        predictions, validation["records"], revised_by_id
    )
    class_balance = {
        dataset: dict(
            sorted(
                Counter(
                    str(item["binary_error"])
                    for item in predictions
                    if item["dataset"] == dataset
                ).items()
            )
        )
        for dataset in sorted({item["dataset"] for item in predictions})
    }
    subgroup_diagnostics = _subgroup_diagnostics(
        predictions, validation["records"], maps
    )
    evaluator_diagnostics = _dataset_evaluator_diagnostics(
        predictions, validation["records"], revised_by_id
    )

    metadata = {
        "audit_identity": audit_identity,
        "source_experiment_id": EXPERIMENT_ID,
        "evaluator_configuration": EVALUATOR_CONFIGURATION,
        "evaluator_configuration_checksum": evaluator_checksum,
        "evaluator_rule_source_checksum": evaluator_rule_source_checksum,
        "analysis_configuration": ANALYSIS_CONFIGURATION,
        "analysis_configuration_checksum": analysis_checksum,
        "integrity": integrity,
    }
    atomic_json(output_root / "audit_metadata.json", metadata)
    atomic_json(output_root / "response_summary.json", response_summary)
    atomic_json(output_root / "revised_evaluation.json", revised)
    atomic_json(output_root / "inspected_responses.json", inspections)
    atomic_json(output_root / "feature_audit.json", feature_audit)
    atomic_json(output_root / "class_balance.json", class_balance)
    atomic_json(output_root / "confound_diagnostics.json", confounds)
    atomic_json(output_root / "feature_family_comparison.json", stability)
    atomic_json(output_root / "cross_validation_diagnostics.json", stability)
    atomic_json(output_root / "subgroup_diagnostics.json", subgroup_diagnostics)
    atomic_json(output_root / "evaluator_diagnostics.json", evaluator_diagnostics)
    atomic_json(
        output_root / "evaluator_change_log.json",
        {
            "changes": [
                {
                    "area": "concise_aliases",
                    "previous_failure": "required the entire response to equal an alias",
                    "rule": "accept a normalized alias at the end of a concise response with at most eight prefix tokens",
                },
                {
                    "area": "parenthetical_text",
                    "previous_failure": "parenthetical qualifiers prevented deterministic alias equality",
                    "rule": "compare once with a single parenthetical span removed",
                },
                {
                    "area": "ambignq_segments",
                    "previous_failure": "multi-answer lists were compared as one string",
                    "rule": "split newline, semicolon, and bullet segments while keeping interpretations isolated",
                },
                {
                    "area": "squad_abstention",
                    "previous_failure": "only empty and literal unanswerable were accepted",
                    "rule": "accept a fixed conservative list of explicit insufficiency phrases",
                },
            ],
            "thresholds_selected_from_labels": False,
            "external_judge_used": False,
        },
    )
    atomic_json(
        output_root / "recommendations.json",
        {
            "next_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "reason": "same model family with greater capacity permits a controlled scale comparison",
            "minimum_per_dataset": {
                "ifi_arith": 500,
                "gsm8k": 2000,
                "squad": 1500,
                "triviaqa": 500,
                "ambignq": 500,
                "truthfulqa": 500,
            },
            "sampling_note": "Use enough correct and incorrect outcomes to target at least 30, preferably 100, examples in the smaller class; GSM8K needs a stronger model or substantially more examples.",
            "truthfulqa": "Keep lexical comparisons diagnostic; use the benchmark's established multiple-choice protocols or predeclared human evaluation in a later experiment.",
        },
    )
    _write_markdown(
        output_root,
        response_summary,
        revised,
        inspections,
        stability,
        confounds,
        feature_audit,
        class_balance,
        subgroup_diagnostics,
        evaluator_diagnostics,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a completed Qwen experiment.")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    args = parser.parse_args()
    print(json.dumps(run_audit(args.project_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
