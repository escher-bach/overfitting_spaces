import numpy as np

from overfitting_spaces.development_analysis import (
    CHECKPOINT_EPOCHS, MODELING_EPOCHS, Record, _inner_splits, transform_flatten,
    whole_run_uncertainty,
)


def test_checkpoint_and_modeling_contracts_are_sealed():
    assert len(CHECKPOINT_EPOCHS) == 41
    assert len(MODELING_EPOCHS) == 40
    assert 0 not in MODELING_EPOCHS


def test_group_splits_never_divide_a_trajectory():
    groups = np.repeat(np.array(["a", "b", "c", "d"]), 40)
    for fit, validation in _inner_splits(groups):
        assert set(groups[fit]).isdisjoint(set(groups[validation]))


def test_target_is_cumulative_minimum_gap_and_representations_shape():
    losses = np.array([2.0, 1.5, 1.7, 1.4])
    assert np.allclose(losses - np.minimum.accumulate(losses), [0.0, 0.0, 0.2, 0.0])
    logits = np.arange(20, dtype=np.float32).reshape(2, 10)
    for representation in ("raw_logits", "class_centered", "scale_normalized", "probabilities"):
        assert transform_flatten(logits, representation).shape == (20,)


def test_whole_run_uncertainty_keeps_complete_trajectories_and_is_seeded():
    records = [
        Record(f"run-{group}", group, epoch, float(epoch) / 100, 0.5, (0.0,), {})
        for group in range(12) for epoch in range(5, 205, 5)
    ]
    target = np.asarray([record.target for record in records])
    behavioral = target.copy()
    nuisance = target + 0.1
    first = whole_run_uncertainty(records, behavioral, nuisance, permutations=100, seed=7)
    second = whole_run_uncertainty(records, behavioral, nuisance, permutations=100, seed=7)
    assert first == second
    assert first["runs_improved"] == 12
    assert first["mean_paired_mae_improvement"] > 0
    assert len(first["per_run_paired_mae_improvement"]) == 12
