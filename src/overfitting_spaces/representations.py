from __future__ import annotations

import torch


def transform_logits(logits: torch.Tensor, representation: str, epsilon: float = 1e-8) -> torch.Tensor:
    if representation == "raw_logits":
        return logits
    centered = logits - logits.mean(dim=-1, keepdim=True)
    if representation == "class_centered":
        return centered
    if representation == "scale_normalized":
        return centered / centered.norm(dim=-1, keepdim=True).clamp_min(epsilon)
    if representation == "probabilities":
        return logits.softmax(dim=-1)
    raise ValueError(f"unknown representation: {representation}")
