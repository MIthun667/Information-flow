from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from statistics import NormalDist
from typing import Any

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
from usig.experiment.records import canonical_json
from usig.experiment.repair_v3 import read_jsonl, write_json, write_jsonl
from usig.data.normalization.text import normalize_question

VERSION = "scientific_v3"


def _secondary(destination: Path) -> dict[str, dict[str, Any]]:
    return {
        item["example_id"]: item
        for item in read_jsonl(destination / "signature_ablations/collection.jsonl")
    }


def _matrix(
    records: dict[str, dict[str, Any]],
    identifiers: list[str],
    family: str,
    names: list[str],
) -> np.ndarray:
    return np.asarray(
        [[records[key][family][name] for name in names] for key in identifiers],
        dtype=float,
    )


def _coefficient_stability(
    fold_details: list[list[dict[str, Any]]], feature_count: int
) -> dict[str, Any]:
    vectors = []
    for seed_folds in fold_details:
        for fold in seed_folds:
            if not fold.get("defined"):
                continue
            vector = np.zeros(feature_count)
            active = fold["active_feature_indices"]
            coefficients = fold["standardized_coefficients"]
            vector[np.asarray(active, dtype=int)] = coefficients
            vectors.append(vector)
    matrix = np.asarray(vectors)
    return {
        "fit_count": len(vectors),
        "mean": matrix.mean(axis=0).tolist(),
        "standard_deviation": matrix.std(axis=0).tolist(),
        "positive_sign_fraction": (matrix > 0).mean(axis=0).tolist(),
        "mean_l2_norm": float(np.linalg.norm(matrix, axis=1).mean()),
        "l2_norm_standard_deviation": float(np.linalg.norm(matrix, axis=1).std()),
    }


def squad_depth_analysis(
    destination: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    predictions, signatures = _load_collection(destination)
    predictions = [item for item in predictions if not item["token_limit_reached"]]
    arrays, labels, identifiers = _feature_arrays(predictions, signatures)
    prediction_by_id = {item["example_id"]: item for item in predictions}
    secondary = _secondary(destination)
    example = secondary[identifiers[0]]
    token_names = sorted(
        name
        for name, value in example["individual_token_dynamics"].items()
        if isinstance(value, (int, float)) and name != "token_instability_std"
    )
    region_names = {}
    region_keys = {"early": "E", "middle": "M", "late": "A"}
    for region in ("early", "middle", "late"):
        region_names[region] = sorted(
            name
            for name, value in example["individual_layer_regions"].items()
            if name.startswith(f"cosine_{region}_")
            and isinstance(value, (int, float))
        )
        arrays[region_keys[region]] = _matrix(
            secondary, identifiers, "individual_layer_regions", region_names[region]
        )
    arrays["T"] = _matrix(
        secondary, identifiers, "individual_token_dynamics", token_names
    )
    # L is reserved for generated length; late-layer features use A.
    arrays["D"] = np.column_stack([arrays["E"], arrays["M"], arrays["A"]])
    arrays["J"] = np.column_stack([arrays["T"], arrays["D"]])
    manifest = {item["example_id"]: item for item in read_jsonl(manifest_path)}
    strata = [manifest[key]["sampling_stratum"] for key in identifiers]
    comparisons = {
        "probability": ("P",),
        "length": ("L",),
        "probability_plus_length": ("P", "L"),
        "scalar_ifi": ("I",),
        "token_ifi": ("T",),
        "early_layer_ifi": ("E",),
        "middle_layer_ifi": ("M",),
        "late_layer_ifi": ("A",),
        "early_middle_ifi": ("E", "M"),
        "middle_late_ifi": ("M", "A"),
        "early_late_ifi": ("E", "A"),
        "full_depth_ifi": ("D",),
        "joint_token_depth_ifi": ("J",),
        "probability_plus_ifi": ("P", "J"),
        "probability_length_joint_ifi": ("P", "L", "J"),
        "probability_length_residual_joint_ifi": ("P", "L", "J_RESIDUAL"),
    }
    all_scores: dict[str, list[np.ndarray]] = {name: [] for name in comparisons}
    all_details: dict[str, list[list[dict[str, Any]]]] = {
        name: [] for name in comparisons
    }
    split_checksums = {}
    folds = min(5, min(Counter(labels.tolist()).values()))
    for seed in SPLIT_SEEDS:
        splits = balanced_splits(labels, strata, folds=folds, seed=seed)
        split_checksums[str(seed)] = hashlib.sha256(
            canonical_json(
                [[train.tolist(), validation.tolist()] for train, validation in splits]
            ).encode()
        ).hexdigest()
        for name, families in comparisons.items():
            scores, details = comparison_predictions(arrays, labels, splits, families)
            all_scores[name].append(scores)
            all_details[name].append(details)
    if len(set(split_checksums.values())) != len(SPLIT_SEEDS):
        raise ValueError("SQuAD split seeds did not produce distinct assignments")
    baseline = np.mean(all_scores["probability_plus_length"], axis=0)
    results = {}
    for name, score_sets in all_scores.items():
        mean_scores = np.mean(score_sets, axis=0)
        result = _score_summary(labels, mean_scores)
        result["per_seed"] = [_score_summary(labels, scores) for scores in score_sets]
        aurocs = [item["auroc"] for item in result["per_seed"]]
        result["auroc_seed_mean"] = statistics.mean(aurocs)
        result["auroc_seed_standard_deviation"] = statistics.pstdev(aurocs)
        feature_count = sum(
            arrays[family.removesuffix("_RESIDUAL")].shape[1]
            for family in comparisons[name]
        )
        result["coefficient_stability"] = _coefficient_stability(
            all_details[name], feature_count
        )
        if name != "probability_plus_length":
            result["gain_over_probability_plus_length"] = _paired_bootstrap(
                labels, baseline, mean_scores, seed=2026
            )
        results[name] = result
    profiles = np.asarray(
        [secondary[key]["layer_profile_32"] for key in identifiers], dtype=float
    )
    answerable = np.asarray(
        [
            bool(prediction_by_id[key]["evaluation_metrics"]["answerable"])
            for key in identifiers
        ]
    )
    baseline_threshold = float(np.quantile(baseline, 0.25))
    missed = [
        {
            "example_id": key,
            "probability_length_error_score": float(score),
        }
        for key, label, score in zip(identifiers, labels, baseline)
        if label == 1 and score <= baseline_threshold
    ]
    report = {
        "version": VERSION,
        "sample_count": len(labels),
        "excluded_truncated_count": sum(
            item["token_limit_reached"] for item in _load_collection(destination)[0]
        ),
        "targets_preserved": [
            "normalized_exact_match",
            "token_f1",
            "answerability_correctness",
            "answerable_subset",
            "unanswerable_subset",
        ],
        "split_checksums": split_checksums,
        "comparisons": results,
        "high_confidence_incorrect_missed_by_probability": {
            "definition": "incorrect and in lowest quartile of probability-plus-length error score",
            "threshold": baseline_threshold,
            "count": len(missed),
            "records": missed,
        },
        "depth_profiles": {
            "correct_mean": profiles[labels == 0].mean(axis=0).tolist(),
            "incorrect_mean": profiles[labels == 1].mean(axis=0).tolist(),
            "answerable_mean": profiles[answerable].mean(axis=0).tolist(),
            "unanswerable_mean": profiles[~answerable].mean(axis=0).tolist(),
        },
        "representation_baseline_limitations": [
            "final-layer pooling and hidden-state norms are unavailable because Version 2 intentionally did not store full hidden tensors",
            "SQuAD recollection is prohibited; raw fixed-depth profiles are retained as the available representation baseline",
        ],
    }
    write_json(output_path, report)
    return report


def trivia_power(
    alias_result_path: Path,
    output_path: Path,
    *,
    target_usable: int = 2000,
) -> dict[str, Any]:
    result = json.loads(alias_result_path.read_text(encoding="utf-8"))
    candidate = result["comparisons"]["probability_length_token_dynamics"]
    gain = candidate["gain_over_probability_plus_length"]
    lower, upper = gain["paired_bootstrap_95_ci"]
    current_n = candidate["sample_count"]
    observed = gain["difference"]
    standard_error = (upper - lower) / (2 * 1.96)
    projected_se = standard_error * math.sqrt(current_n / target_usable)
    z = abs(observed) / projected_se if projected_se else math.inf
    projected_power = NormalDist().cdf(z - 1.96) + NormalDist().cdf(-z - 1.96)
    current_inconclusive = lower <= 0 <= upper
    report = {
        "version": VERSION,
        "primary_target": "official_alias_aware",
        "current_usable_count": current_n,
        "current_correct_count": candidate["correct_count"],
        "current_incorrect_count": candidate["incorrect_count"],
        "current_gain": observed,
        "current_paired_95_ci": [lower, upper],
        "target_usable_count": target_usable,
        "projected_standard_error": projected_se,
        "approximate_two_sided_power_at_observed_effect": projected_power,
        "assumptions": [
            "record-level variance scales inversely with sample size",
            "future records have similar class balance and effect size",
            "normal approximation to paired bootstrap difference",
        ],
        "extension_recommended": (
            current_inconclusive and projected_power >= 0.8 and target_usable > current_n
        ),
        "decision_rule": "extend only when current CI crosses zero and projected power at 2000 usable records is at least 0.80",
    }
    write_json(output_path, report)
    return report


def trivia_extension_manifest(
    validation_path: Path,
    train_path: Path,
    existing_manifest_path: Path,
    power_path: Path,
    output_path: Path,
    *,
    count: int = 1000,
) -> dict[str, Any]:
    power = json.loads(power_path.read_text(encoding="utf-8"))
    if not power.get("extension_recommended"):
        raise PermissionError(
            "TriviaQA extension refused: the prespecified power rule did not recommend it"
        )
    existing = {item["example_id"] for item in read_jsonl(existing_manifest_path)}
    training_questions = {
        normalize_question(item["question"]) for item in read_jsonl(train_path)
    }
    candidates = sorted(
        (
            item
            for item in read_jsonl(validation_path)
            if item["example_id"] not in existing
            and normalize_question(item["question"]) not in training_questions
        ),
        key=lambda item: item["example_id"],
    )
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} non-overlapping TriviaQA records")
    source_checksum = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    rows = [
        {
            "canonical_record_checksum": hashlib.sha256(
                (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
            ).hexdigest(),
            "dataset": "triviaqa",
            "example_id": record["example_id"],
            "group_id": record["group_id"],
            "sampling_seed": 2026,
            "sampling_stratum": "nonoverlap_extension",
            "selection_order": index,
            "source_checksum": source_checksum,
            "source_split": record["split"],
        }
        for index, record in enumerate(candidates[:count])
    ]
    write_jsonl(output_path, rows)
    result = {
        "version": VERSION,
        "sample_count": len(rows),
        "overlap_with_existing_count": sum(
            item["example_id"] in existing for item in rows
        ),
        "prompt_version": "triviaqa_five_shot_short_v1",
        "official_alias_aware_evaluator_required": True,
        "checksum": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    return result


def truthfulqa_high_confidence_false(
    destination: Path, manifest_path: Path, output_path: Path
) -> dict[str, Any]:
    predictions, signatures = _load_collection(destination)
    predictions = [
        item
        for item in predictions
        if not item.get("token_limit_reached") and not item.get("unresolved_label")
    ]
    arrays, labels, identifiers = _feature_arrays(predictions, signatures)
    manifest = {item["example_id"]: item for item in read_jsonl(manifest_path)}
    strata = [manifest[item]["sampling_stratum"] for item in identifiers]
    baseline_sets, candidate_sets = [], []
    checksums = {}
    folds = min(5, min(Counter(labels.tolist()).values()))
    for seed in SPLIT_SEEDS:
        splits = balanced_splits(labels, strata, folds=folds, seed=seed)
        checksums[str(seed)] = hashlib.sha256(
            canonical_json(
                [[train.tolist(), test.tolist()] for train, test in splits]
            ).encode()
        ).hexdigest()
        baseline_sets.append(
            comparison_predictions(arrays, labels, splits, ("P", "L"))[0]
        )
        candidate_sets.append(
            comparison_predictions(arrays, labels, splits, ("P", "L", "C_RESIDUAL"))[0]
        )
    baseline = np.mean(baseline_sets, axis=0)
    candidate = np.mean(candidate_sets, axis=0)
    threshold = float(np.quantile(baseline, 0.25))
    subgroup = (labels == 1) & (baseline <= threshold)
    report = {
        "version": VERSION,
        "definition": "incorrect MC1 selections in the lowest quartile of probability-plus-length error risk",
        "sample_count": len(labels),
        "high_confidence_false_count": int(subgroup.sum()),
        "probability_length_threshold": threshold,
        "mean_baseline_risk_in_subgroup": (
            float(baseline[subgroup].mean()) if subgroup.any() else None
        ),
        "mean_residual_ifi_risk_in_subgroup": (
            float(candidate[subgroup].mean()) if subgroup.any() else None
        ),
        "mean_risk_increase_from_residual_ifi": (
            float((candidate[subgroup] - baseline[subgroup]).mean())
            if subgroup.any()
            else None
        ),
        "split_checksums": checksums,
        "record_ids": [
            identifier
            for identifier, selected in zip(identifiers, subgroup)
            if selected
        ],
    }
    write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    squad = sub.add_parser("squad-depth")
    squad.add_argument("--destination", type=Path, required=True)
    squad.add_argument("--manifest", type=Path, required=True)
    squad.add_argument("--output", type=Path, required=True)
    trivia = sub.add_parser("trivia-power")
    trivia.add_argument("--alias-result", type=Path, required=True)
    trivia.add_argument("--output", type=Path, required=True)
    extension = sub.add_parser("trivia-extension-manifest")
    extension.add_argument("--validation", type=Path, required=True)
    extension.add_argument("--train", type=Path, required=True)
    extension.add_argument("--existing-manifest", type=Path, required=True)
    extension.add_argument("--power", type=Path, required=True)
    extension.add_argument("--output", type=Path, required=True)
    truthful = sub.add_parser("truthfulqa-high-confidence-false")
    truthful.add_argument("--destination", type=Path, required=True)
    truthful.add_argument("--manifest", type=Path, required=True)
    truthful.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "squad-depth":
        result = squad_depth_analysis(args.destination, args.manifest, args.output)
    elif args.action == "trivia-power":
        result = trivia_power(args.alias_result, args.output)
    elif args.action == "trivia-extension-manifest":
        result = trivia_extension_manifest(
            args.validation,
            args.train,
            args.existing_manifest,
            args.power,
            args.output,
        )
    else:
        result = truthfulqa_high_confidence_false(
            args.destination, args.manifest, args.output
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
