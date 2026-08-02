from __future__ import annotations

import math
from typing import Any

import torch


def extract_probability_features(
    scores: tuple[torch.Tensor, ...],
    selected_token_ids: list[int],
) -> dict[str, Any]:
    if len(scores) != len(selected_token_ids):
        raise ValueError("Generation scores and selected token IDs are misaligned")
    token_features: list[dict[str, float | int]] = []
    for index, (logits, token_id) in enumerate(zip(scores, selected_token_ids)):
        values = logits[0].detach().float()
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite generation logits")
        log_probs = torch.log_softmax(values, dim=-1)
        probs = torch.softmax(values, dim=-1)
        selected_log_prob = float(log_probs[token_id])
        selected_probability = float(probs[token_id])
        entropy = float(-(probs * log_probs).sum())
        top_values, top_indices = torch.topk(probs, k=2)
        rank = int((values > values[token_id]).sum().item()) + 1
        token_features.append(
            {
                "token_index": index,
                "token_id": token_id,
                "selected_log_probability": selected_log_prob,
                "selected_probability": selected_probability,
                "entropy": entropy,
                "selected_rank": rank,
                "top_two_probability_margin": float(top_values[0] - top_values[1]),
                "top_token_id": int(top_indices[0]),
            }
        )
    if not token_features:
        raise ValueError("No generated-token probability features")

    def values(name: str) -> list[float]:
        return [float(item[name]) for item in token_features]

    entropies = values("entropy")
    log_probs = values("selected_log_probability")
    margins = values("top_two_probability_margin")

    def population_std(items: list[float]) -> float:
        mean = sum(items) / len(items)
        return math.sqrt(sum((value - mean) ** 2 for value in items) / len(items))

    summary = {
        "mean_token_entropy": sum(entropies) / len(entropies),
        "maximum_token_entropy": max(entropies),
        "minimum_token_entropy": min(entropies),
        "token_entropy_std": population_std(entropies),
        "mean_selected_token_log_probability": sum(log_probs) / len(log_probs),
        "minimum_selected_token_log_probability": min(log_probs),
        "maximum_selected_token_log_probability": max(log_probs),
        "negative_mean_log_probability": -sum(log_probs) / len(log_probs),
        "selected_token_log_probability_std": population_std(log_probs),
        "mean_top_two_probability_margin": sum(margins) / len(margins),
        "minimum_top_two_probability_margin": min(margins),
        "generated_token_count": len(token_features),
    }
    if not all(math.isfinite(float(value)) for value in summary.values()):
        raise ValueError("Non-finite probability summary")
    return {"tokens": token_features, "summary": summary}
