from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def interior_transition_indices(count: int, lower: float = 0.1, upper: float = 0.9) -> list[int]:
    if count < 1 or not 0 <= lower < upper <= 1:
        raise ValueError("Invalid transition count or interior proportions")
    start = math.ceil(count * lower)
    stop = math.floor(count * upper)
    if stop <= start:
        return list(range(count))
    return list(range(start, stop))


def depth_regions(count: int) -> dict[str, list[int]]:
    if count < 3:
        raise ValueError("At least three transitions are required for depth regions")
    boundaries = np.array_split(np.arange(count), 3)
    return {
        name: [int(value) for value in region]
        for name, region in zip(("early", "middle", "late"), boundaries)
    }


def _slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0])


def _roughness(values: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(values)))) if len(values) > 1 else 0.0


def _profile_features(matrix: np.ndarray, prefix: str) -> dict[str, Any]:
    profile = matrix.mean(axis=0)
    source = np.linspace(0.0, 1.0, len(profile))
    target = np.linspace(0.0, 1.0, 32)
    fixed = np.interp(target, source, profile)
    maximum_index = int(np.argmax(profile))
    result: dict[str, Any] = {
        f"{prefix}_fixed_depth_profile": fixed.tolist(),
        f"{prefix}_profile_mean": float(profile.mean()),
        f"{prefix}_profile_std": float(profile.std(ddof=0)),
        f"{prefix}_profile_slope": _slope(profile),
        f"{prefix}_profile_roughness": _roughness(profile),
        f"{prefix}_profile_curvature": (
            float(np.mean(np.abs(np.diff(profile, n=2)))) if len(profile) > 2 else 0.0
        ),
        f"{prefix}_profile_maximum": float(profile.max()),
        f"{prefix}_profile_normalized_maximum_position": maximum_index
        / max(1, len(profile) - 1),
        f"{prefix}_layer_position_of_maximum_mean_transition": maximum_index,
    }
    regions = depth_regions(matrix.shape[1])
    region_means: dict[str, float] = {}
    for name, indices in regions.items():
        values = matrix[:, indices].reshape(-1)
        region_means[name] = float(values.mean())
        result.update(
            {
                f"{prefix}_{name}_mean": float(values.mean()),
                f"{prefix}_{name}_std": float(values.std(ddof=0)),
                f"{prefix}_{name}_maximum": float(values.max()),
                f"{prefix}_{name}_range": float(values.max() - values.min()),
            }
        )
    epsilon = 1e-8
    result[f"{prefix}_early_to_middle_ratio"] = region_means["early"] / (
        region_means["middle"] + epsilon
    )
    result[f"{prefix}_middle_to_late_ratio"] = region_means["middle"] / (
        region_means["late"] + epsilon
    )
    result[f"{prefix}_early_to_late_ratio"] = region_means["early"] / (
        region_means["late"] + epsilon
    )
    return result


def calculate_signature(
    cosine: torch.Tensor,
    relative: torch.Tensor,
) -> dict[str, Any]:
    if cosine.shape != relative.shape or cosine.ndim != 2:
        raise ValueError("Cosine and relative matrices must align as [tokens, transitions]")
    if not torch.isfinite(cosine).all() or not torch.isfinite(relative).all():
        raise ValueError("Non-finite transition matrices")
    indices = interior_transition_indices(cosine.shape[1])
    cosine_values = cosine.detach().cpu().numpy()
    relative_values = relative.detach().cpu().numpy()
    token_instability = cosine_values[:, indices].mean(axis=1)
    status = "ok" if len(token_instability) >= 2 else "insufficient_tokens"
    scalar_ifi = float(token_instability.std(ddof=0)) if status == "ok" else None
    maximum_token = int(np.argmax(token_instability))
    token_features = {
        "mean_token_instability": float(token_instability.mean()),
        "token_instability_std": float(token_instability.std(ddof=0)),
        "minimum_token_instability": float(token_instability.min()),
        "maximum_token_instability": float(token_instability.max()),
        "token_instability_range": float(token_instability.max() - token_instability.min()),
        "first_token_instability": float(token_instability[0]),
        "last_token_instability": float(token_instability[-1]),
        "token_instability_slope": _slope(token_instability),
        "token_instability_roughness": _roughness(token_instability),
        "token_position_of_maximum_instability": maximum_token,
    }
    result = {
        "feature_status": status,
        "scalar_ifi": scalar_ifi,
        "generated_tokens_used": int(cosine.shape[0]),
        "interior_transition_indices": indices,
        "interior_transition_count": len(indices),
        "population_standard_deviation": True,
        "cosine_token_dynamics": token_features,
        "cosine_structured": _profile_features(cosine_values, "cosine"),
        "relative_structured": _profile_features(relative_values, "relative"),
        "numerical_warning": None,
    }
    numeric_values: list[float] = []
    for section in ("cosine_token_dynamics", "cosine_structured", "relative_structured"):
        for value in result[section].values():
            if isinstance(value, list):
                numeric_values.extend(value)
            elif isinstance(value, (int, float)):
                numeric_values.append(float(value))
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("Non-finite structured signature")
    return result
