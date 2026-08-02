from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional


def align_answer_hidden_states(
    hidden_states: tuple[torch.Tensor, ...],
    *,
    prompt_length: int,
    generated_token_count: int,
) -> torch.Tensor:
    if len(hidden_states) < 3:
        raise ValueError("Expected embedding plus at least two transformer hidden levels")
    sequence_length = hidden_states[0].shape[1]
    if prompt_length + generated_token_count != sequence_length:
        raise ValueError("Hidden-state sequence and generated-token boundary are misaligned")
    shapes = {tuple(state.shape) for state in hidden_states}
    if len(shapes) != 1:
        raise ValueError("Hidden-state levels have inconsistent shapes")
    # Exclude embedding output. Shape: answer tokens, transformer layers, hidden size.
    answer = torch.stack(
        [state[0, prompt_length:, :].detach().float().cpu() for state in hidden_states[1:]],
        dim=1,
    )
    if answer.shape[0] != generated_token_count:
        raise ValueError("Prompt-only or padding tokens entered answer hidden states")
    if not torch.isfinite(answer).all():
        raise ValueError("Non-finite answer hidden states")
    return answer


def transition_matrices(answer_hidden_states: torch.Tensor, epsilon: float = 1e-8) -> dict[str, torch.Tensor]:
    if answer_hidden_states.ndim != 3:
        raise ValueError("Expected [tokens, layers, hidden] answer states")
    if answer_hidden_states.shape[1] < 2:
        raise ValueError("At least two transformer layers are required")
    left = answer_hidden_states[:, :-1, :]
    right = answer_hidden_states[:, 1:, :]
    cosine = 1.0 - functional.cosine_similarity(left, right, dim=-1, eps=epsilon)
    relative = torch.linalg.vector_norm(right - left, dim=-1) / torch.clamp(
        torch.linalg.vector_norm(left, dim=-1), min=epsilon
    )
    if not torch.isfinite(cosine).all() or not torch.isfinite(relative).all():
        raise ValueError("Non-finite layer-transition features")
    return {"cosine": cosine, "relative": relative}


def completed_sequence_hidden_states(
    model: Any, full_token_ids: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    device = next(model.parameters()).device
    ids = full_token_ids.to(device)
    attention_mask = torch.ones_like(ids)
    with torch.inference_mode():
        outputs = model(
            input_ids=ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    if outputs.hidden_states is None:
        raise RuntimeError("Model did not return hidden states")
    return outputs.hidden_states
