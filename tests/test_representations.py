import torch

from overfitting_spaces.representations import transform_logits


def test_centered_logits_remove_additive_offset():
    logits = torch.tensor([[1.0, 2.0, 4.0]])
    assert torch.allclose(transform_logits(logits, "class_centered"), transform_logits(logits + 100.0, "class_centered"))


def test_scale_normalization_has_unit_norm():
    value = transform_logits(torch.tensor([[1.0, 2.0, 4.0]]), "scale_normalized")
    assert torch.allclose(value.norm(dim=-1), torch.ones(1))
