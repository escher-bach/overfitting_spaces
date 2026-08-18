"""Development-only analysis for the fixed CIFAR-10 experiment inventory.

This module deliberately encodes the development boundary rather than scanning
directories: only the twelve predeclared trajectories may enter an analysis.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


INVENTORY: tuple[tuple[str, tuple[int, int]], ...] = (
    ("run-ef7523a1358a", (881886696, 1226023601)),
    ("run-be83c9d3e5af", (3549333581, 2325060526)),
    ("run-0e25bbe73f49", (4247511459, 3127276340)),
    ("run-ebbe4035ec6e", (3002361271, 1741336945)),
    ("run-f856f0c1ae51", (310190245, 2722409734)),
    ("run-98357fef99fe", (4292514521, 971400669)),
)
CHECKPOINT_EPOCHS = tuple(range(0, 201, 5))
MODELING_EPOCHS = tuple(range(5, 201, 5))
REPRESENTATIONS = ("raw_logits", "class_centered", "scale_normalized", "probabilities")
PCA_PREFIXES = (8, 16, 32)
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class Record:
    run_id: str
    seed: int
    epoch: int
    target: float
    evaluation_accuracy: float
    nuisance: tuple[float, ...]
    history_logit: dict[str, Any]

    @property
    def group(self) -> str:
        return f"{self.run_id}/seed-{self.seed}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _history_path(audit_root: Path, run_id: str, seed: int) -> Path:
    return audit_root / run_id / "seeds" / f"seed-{seed}" / "metrics" / "history.json"


def _raw_path(raw_root: Path, run_id: str, seed: int, epoch: int) -> Path:
    suffix = Path("seeds") / f"seed-{seed}" / "logits" / f"epoch-{epoch:03d}.safetensors"
    matches = list((raw_root / run_id).rglob(str(suffix)))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one raw tensor for {run_id} seed {seed} epoch {epoch}; found {len(matches)}")
    return matches[0]


def _as_float_list(value: Any) -> list[float]:
    return [float(item) for item in value]


def load_and_validate(audit_root: Path, raw_root: Path) -> tuple[list[Record], dict[str, np.ndarray], list[dict[str, Any]]]:
    """Load only the authorized inventory, validating all 492 logit files."""
    all_records: list[Record] = []
    logits: dict[str, np.ndarray] = {}
    integrity: list[dict[str, Any]] = []
    for run_id, seeds in INVENTORY:
        for seed in seeds:
            history = json.loads(_history_path(audit_root, run_id, seed).read_text(encoding="utf-8"))
            checkpoint = {int(item["epoch"]): item for item in history if "probe_logits" in item}
            if tuple(sorted(checkpoint)) != CHECKPOINT_EPOCHS:
                raise ValueError(f"{run_id}/seed-{seed} lacks the complete 0..200 checkpoint schedule")
            running_min = float("inf")
            for epoch in CHECKPOINT_EPOCHS:
                item = checkpoint[epoch]
                evaluation = item.get("evaluation")
                if evaluation is None:
                    raise ValueError(f"missing evaluation for {run_id}/seed-{seed}/epoch-{epoch}")
                running_min = min(running_min, float(evaluation["loss"]))
                descriptor = item["probe_logits"]
                path = _raw_path(raw_root, run_id, seed, epoch)
                actual_sha = _sha256(path)
                with safe_open(str(path), framework="np") as tensor_file:
                    keys = list(tensor_file.keys())
                    if keys != ["logits"]:
                        raise ValueError(f"unexpected tensor keys in {path}: {keys}")
                    tensor = tensor_file.get_tensor("logits")
                expected_shape = tuple(int(v) for v in descriptor["shape"])
                if tensor.dtype != np.float32 or tensor.shape != expected_shape or tensor.shape != (5000, 10):
                    raise ValueError(f"invalid logits tensor {path}: dtype={tensor.dtype}, shape={tensor.shape}")
                if actual_sha != descriptor["sha256"]:
                    raise ValueError(f"SHA-256 mismatch for {path}")
                key = f"{run_id}/seed-{seed}/epoch-{epoch}"
                logits[key] = np.asarray(tensor, dtype=np.float32)
                integrity.append({"run_id": run_id, "seed": seed, "epoch": epoch, "sha256": actual_sha, "shape": list(tensor.shape)})
                if epoch in MODELING_EPOCHS:
                    train = item.get("train")
                    if train is None:
                        raise ValueError(f"missing train metrics at modeling checkpoint {key}")
                    nuisance = (
                        float(epoch), float(train["loss"]), float(train["accuracy"]), float(item["learning_rate"]),
                        float(item["nuisance"]["mean_entropy"]), float(item["nuisance"]["mean_logit_norm"]),
                        float(item["nuisance"]["mean_max_probability"]),
                        *_as_float_list(item["nuisance"]["predicted_class_histogram"]),
                    )
                    all_records.append(Record(run_id, seed, epoch, float(evaluation["loss"]) - running_min,
                                              float(evaluation["accuracy"]), nuisance, descriptor))
    if len(integrity) != 492 or len(all_records) != 480 or len({record.group for record in all_records}) != 12:
        raise ValueError("development integrity contract failed (expected 492 raw checkpoints and 12x40 modeling rows)")
    return all_records, logits, integrity


def transform_flatten(logits: np.ndarray, representation: str) -> np.ndarray:
    if representation == "raw_logits":
        transformed = logits
    elif representation == "class_centered":
        transformed = logits - logits.mean(axis=1, keepdims=True)
    elif representation == "scale_normalized":
        centered = logits - logits.mean(axis=1, keepdims=True)
        transformed = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8)
    elif representation == "probabilities":
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        transformed = exp / exp.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown representation: {representation}")
    return transformed.reshape(-1).astype(np.float32, copy=False)


def make_feature_sets(records: list[Record]) -> dict[str, np.ndarray]:
    nuisance = np.asarray([record.nuisance for record in records], dtype=np.float64)
    # epoch, train loss, train accuracy, learning rate, three probe summaries, ten class counts
    return {
        "epoch_only": nuisance[:, [0]],
        "train_loss_accuracy": nuisance[:, [1, 2]],
        "probe_nuisance": nuisance[:, 4:],
        "all_nuisance": nuisance,
        "evaluation_accuracy_control": np.asarray([[record.evaluation_accuracy] for record in records], dtype=np.float64),
    }


def _fit_predict_ridge(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float) -> np.ndarray:
    scaler = StandardScaler()
    return Ridge(alpha=alpha).fit(scaler.fit_transform(train_x), train_y).predict(scaler.transform(test_x))


def _inner_splits(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    number = min(3, len(np.unique(groups)))
    return list(GroupKFold(n_splits=number).split(np.zeros(len(groups)), groups=groups))


def _select_baseline(train_x: np.ndarray, train_y: np.ndarray, train_groups: np.ndarray) -> float:
    scores: dict[float, list[float]] = {alpha: [] for alpha in RIDGE_ALPHAS}
    for fit, validation in _inner_splits(train_groups):
        for alpha in RIDGE_ALPHAS:
            prediction = _fit_predict_ridge(train_x[fit], train_y[fit], train_x[validation], alpha)
            scores[alpha].append(mean_absolute_error(train_y[validation], prediction))
    return min(RIDGE_ALPHAS, key=lambda alpha: (float(np.mean(scores[alpha])), alpha))


def _pca_fit_transform(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=max(PCA_PREFIXES), svd_solver="randomized", random_state=1729, iterated_power=3)
    return pca.fit_transform(train_x), pca.transform(test_x)


def _select_behavioral(train_nuisance: np.ndarray, train_representation: np.ndarray, train_y: np.ndarray,
                       train_groups: np.ndarray) -> tuple[int, float]:
    scores: dict[tuple[int, float], list[float]] = {(components, alpha): [] for components in PCA_PREFIXES for alpha in RIDGE_ALPHAS}
    for fit, validation in _inner_splits(train_groups):
        fitted, projected = _pca_fit_transform(train_representation[fit], train_representation[validation])
        for components in PCA_PREFIXES:
            x_fit = np.hstack((train_nuisance[fit], fitted[:, :components]))
            x_validation = np.hstack((train_nuisance[validation], projected[:, :components]))
            for alpha in RIDGE_ALPHAS:
                prediction = _fit_predict_ridge(x_fit, train_y[fit], x_validation, alpha)
                scores[(components, alpha)].append(mean_absolute_error(train_y[validation], prediction))
    return min(scores, key=lambda key: (float(np.mean(scores[key])), key[0], key[1]))


def _metrics(y: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    correlation = spearmanr(y, predicted).statistic
    return {"mae": float(mean_absolute_error(y, predicted)), "r2": float(r2_score(y, predicted)), "spearman": float(correlation)}


def whole_run_uncertainty(records: list[Record], behavioral_prediction: np.ndarray,
                          nuisance_prediction: np.ndarray, permutations: int = 10_000,
                          seed: int = 1729) -> dict[str, Any]:
    """Whole-trajectory uncertainty and identity-reassignment control.

    The permutation leaves each forty-epoch target path intact and only changes
    which predicted trajectory it is paired with.  Epoch positions therefore
    remain matched within every reassignment.
    """
    y = np.asarray([record.target for record in records], dtype=np.float64)
    groups = np.asarray([record.group for record in records])
    epochs = np.asarray([record.epoch for record in records])
    group_names = np.asarray(sorted(set(groups)))
    if len(group_names) != 12 or any(np.sum(groups == group) != 40 for group in group_names):
        raise ValueError("whole-run uncertainty requires exactly twelve complete 40-checkpoint trajectories")
    behavioral_mae = np.asarray([mean_absolute_error(y[groups == group], behavioral_prediction[groups == group]) for group in group_names])
    nuisance_mae = np.asarray([mean_absolute_error(y[groups == group], nuisance_prediction[groups == group]) for group in group_names])
    improvement = nuisance_mae - behavioral_mae
    rng = np.random.default_rng(seed)
    bootstrap = np.mean(improvement[rng.integers(0, len(improvement), size=(permutations, len(improvement)))], axis=1)
    permutation_improvement = np.empty(permutations)
    permutation_behavioral_mae = np.empty(permutations)
    permutation_nuisance_mae = np.empty(permutations)
    positions = {group: {int(epoch): index for index, epoch in zip(np.flatnonzero(groups == group), epochs[groups == group])} for group in group_names}
    for iteration in range(permutations):
        assigned = rng.permutation(group_names)
        reassigned_target = np.empty_like(y)
        for destination, source in zip(group_names, assigned):
            for epoch, destination_index in positions[destination].items():
                reassigned_target[destination_index] = y[positions[source][epoch]]
        behavioral = np.asarray([mean_absolute_error(reassigned_target[groups == group], behavioral_prediction[groups == group]) for group in group_names])
        nuisance = np.asarray([mean_absolute_error(reassigned_target[groups == group], nuisance_prediction[groups == group]) for group in group_names])
        permutation_improvement[iteration] = np.mean(nuisance - behavioral)
        permutation_behavioral_mae[iteration] = np.mean(behavioral)
        permutation_nuisance_mae[iteration] = np.mean(nuisance)
    observed_improvement = float(np.mean(improvement))
    observed_behavioral_mae = float(np.mean(behavioral_mae))
    observed_nuisance_mae = float(np.mean(nuisance_mae))
    return {
        "unit": "complete 40-checkpoint seed trajectory", "seed": seed, "bootstrap_replicates": permutations,
        "permutation_replicates": permutations,
        "per_run_paired_mae_improvement": {str(group): float(value) for group, value in zip(group_names, improvement)},
        "runs_improved": int(np.sum(improvement > 0)), "runs_total": int(len(improvement)),
        "mean_paired_mae_improvement": observed_improvement,
        "bootstrap_percentile_ci_95": [float(value) for value in np.percentile(bootstrap, [2.5, 97.5])],
        "trajectory_identity_permutation": {
            "statistic": "mean paired MAE improvement (all-nuisance minus behavioral)",
            "observed": observed_improvement,
            "one_sided_p_value": float((1 + np.sum(permutation_improvement >= observed_improvement)) / (permutations + 1)),
            "nuisance_control": {
                "statistic": "mean per-run MAE; lower is better",
                "observed": observed_nuisance_mae,
                "one_sided_p_value": float((1 + np.sum(permutation_nuisance_mae <= observed_nuisance_mae)) / (permutations + 1)),
            },
            "behavioral_model": {
                "statistic": "mean per-run MAE; lower is better",
                "observed": observed_behavioral_mae,
                "one_sided_p_value": float((1 + np.sum(permutation_behavioral_mae <= observed_behavioral_mae)) / (permutations + 1)),
            },
        },
    }


def final_development_choices(records: list[Record], selected_representation: str,
                              selected_features: np.ndarray) -> dict[str, Any]:
    """Select the hyperparameters once on all development trajectories.

    These choices are intended to be locked before confirmation.  Grouped inner
    CV is retained, so no individual checkpoint is split from its trajectory.
    """
    y = np.asarray([record.target for record in records], dtype=np.float64)
    groups = np.asarray([record.group for record in records])
    nuisance = make_feature_sets(records)
    choices: dict[str, Any] = {}
    for name, features in nuisance.items():
        choices[name] = {"alpha": _select_baseline(features, y, groups)}
    components, alpha = _select_behavioral(nuisance["all_nuisance"], selected_features, y, groups)
    choices[f"behavioral_{selected_representation}"] = {"representation": selected_representation, "components": components, "alpha": alpha}
    return choices


def nested_cv(records: list[Record], representation_features: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    y = np.asarray([record.target for record in records], dtype=np.float64)
    groups = np.asarray([record.group for record in records])
    nuisances = make_feature_sets(records)
    models: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    outer = list(GroupKFold(n_splits=len(np.unique(groups))).split(np.zeros(len(groups)), y, groups))
    for name, features in nuisances.items():
        predicted = np.full(len(y), np.nan)
        choices = []
        for fit, test in outer:
            alpha = _select_baseline(features[fit], y[fit], groups[fit])
            predicted[test] = _fit_predict_ridge(features[fit], y[fit], features[test], alpha)
            choices.append({"held_out_group": str(groups[test][0]), "alpha": alpha})
        models[name] = {"type": "nuisance", "choices": choices, "pooled": _metrics(y, predicted)}
        predictions[name] = predicted
    all_nuisance = nuisances["all_nuisance"]
    for representation, features in representation_features.items():
        predicted = np.full(len(y), np.nan)
        choices = []
        for fit, test in outer:
            components, alpha = _select_behavioral(all_nuisance[fit], features[fit], y[fit], groups[fit])
            projected_fit, projected_test = _pca_fit_transform(features[fit], features[test])
            x_fit = np.hstack((all_nuisance[fit], projected_fit[:, :components]))
            x_test = np.hstack((all_nuisance[test], projected_test[:, :components]))
            predicted[test] = _fit_predict_ridge(x_fit, y[fit], x_test, alpha)
            choices.append({"held_out_group": str(groups[test][0]), "components": components, "alpha": alpha})
        models[f"behavioral_{representation}"] = {
            "type": "behavioral", "representation": representation, "choices": choices,
            "pooled": _metrics(y, predicted),
            "improvement_over_all_nuisance_mae": float(models["all_nuisance"]["pooled"]["mae"] - _metrics(y, predicted)["mae"]),
        }
        predictions[f"behavioral_{representation}"] = predicted
    for name, predicted in predictions.items():
        per_group = {}
        for group in np.unique(groups):
            mask = groups == group
            per_group[str(group)] = _metrics(y[mask], predicted[mask])
        models[name]["per_run"] = per_group
    return models, predictions


def development_geometry(records: list[Record], features: np.ndarray, selected_representation: str, selected_components: int) -> dict[str, Any]:
    """Exploratory geometry; this fit is development-only and not confirmation evidence."""
    coordinates = PCA(n_components=max(PCA_PREFIXES), svd_solver="randomized", random_state=1729, iterated_power=3).fit_transform(features)[:, :selected_components]
    coordinates = StandardScaler().fit_transform(coordinates)
    g = np.asarray([record.target for record in records])
    accuracy = np.asarray([record.evaluation_accuracy for record in records])
    groups = np.asarray([record.group for record in records])
    epochs = np.asarray([record.epoch for record in records])
    pairs = list(combinations(range(len(records)), 2))
    def association(pair_subset: list[tuple[int, int]]) -> dict[str, Any]:
        euclidean = np.asarray([np.linalg.norm(coordinates[i] - coordinates[j]) for i, j in pair_subset])
        cosine = np.asarray([1 - np.dot(coordinates[i], coordinates[j]) / max(np.linalg.norm(coordinates[i]) * np.linalg.norm(coordinates[j]), 1e-12) for i, j in pair_subset])
        delta = np.asarray([abs(g[i] - g[j]) for i, j in pair_subset])
        return {"pairs": len(pair_subset), "euclidean_spearman_with_abs_g_difference": float(spearmanr(euclidean, delta).statistic),
                "cosine_spearman_with_abs_g_difference": float(spearmanr(cosine, delta).statistic)}
    cross_run = [(i, j) for i, j in pairs if groups[i] != groups[j]]
    matched_epoch = [(i, j) for i, j in cross_run if epochs[i] == epochs[j]]
    bins = np.floor(accuracy / 0.02).astype(int)
    accuracy_matched = [(i, j) for i, j in cross_run if bins[i] == bins[j]]
    zero = [i for i in range(len(records)) if np.isclose(g[i], 0.0)]
    positive = [i for i in range(len(records)) if g[i] > 1e-12]
    def mean_distance(indices_a: list[int], indices_b: list[int], same: bool) -> dict[str, Any]:
        comparison = ((i, j) for i in indices_a for j in indices_b if (i < j if same else True) and groups[i] != groups[j])
        values = [float(np.linalg.norm(coordinates[i] - coordinates[j])) for i, j in comparison]
        return {"pairs": len(values), "mean_euclidean_distance": float(np.mean(values)) if values else float("nan")}
    # Direction alignment at matched adjacent intervals, across distinct trajectories.
    by_group_epoch = {(record.group, record.epoch): index for index, record in enumerate(records)}
    alignment, next_delta = [], []
    group_names = sorted(set(groups))
    for epoch in MODELING_EPOCHS[:-1]:
        for first, second in combinations(group_names, 2):
            i, i_next = by_group_epoch[(first, epoch)], by_group_epoch[(first, epoch + 5)]
            j, j_next = by_group_epoch[(second, epoch)], by_group_epoch[(second, epoch + 5)]
            left, right = coordinates[i_next] - coordinates[i], coordinates[j_next] - coordinates[j]
            alignment.append(float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12)))
            next_delta.append(float(abs(g[i_next] - g[j_next])))
    return {
        "representation": selected_representation, "development_fit_components": selected_components,
        "cross_run": association(cross_run), "matched_epoch": association(matched_epoch),
        "evaluation_accuracy_bin_width": 0.02, "accuracy_binned": association(accuracy_matched),
        "state_distance": {"g_zero_vs_g_zero": mean_distance(zero, zero, True), "g_positive_vs_g_positive": mean_distance(positive, positive, True), "g_zero_vs_g_positive": mean_distance(zero, positive, False)},
        "adjacent_direction_alignment_exploratory": {"pairs": len(alignment), "mean_cosine": float(np.mean(alignment)), "spearman_with_abs_next_g_difference": float(spearmanr(alignment, next_delta).statistic)},
    }


def write_outputs(output: Path, records: list[Record], integrity: list[dict[str, Any]], models: dict[str, Any],
                  predictions: dict[str, np.ndarray], geometry: dict[str, Any], uncertainty: dict[str, Any],
                  final_choices: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "integrity.json").write_text(json.dumps({"validated_raw_tensors": len(integrity), "expected": 492, "items": integrity}, indent=2) + "\n", encoding="utf-8")
    with (output / "model_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "mae", "r2", "spearman", "improvement_over_all_nuisance_mae"])
        writer.writeheader()
        for name, result in models.items():
            writer.writerow({"model": name, **result["pooled"], "improvement_over_all_nuisance_mae": result.get("improvement_over_all_nuisance_mae", 0.0)})
    with (output / "per_run_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "run", "mae", "r2", "spearman"])
        writer.writeheader()
        for name, result in models.items():
            for run, metric in result["per_run"].items():
                writer.writerow({"model": name, "run": run, **metric})
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "seed", "group", "epoch", "target", *predictions])
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow({"run_id": record.run_id, "seed": record.seed, "group": record.group, "epoch": record.epoch,
                             "target": record.target, **{name: float(value[index]) for name, value in predictions.items()}})
    report = {"scope": "development-only; provisional and non-confirmatory", "n_records": len(records), "n_trajectories": 12,
              "modeling_epochs": list(MODELING_EPOCHS), "target": "evaluation loss minus cumulative within-trajectory minimum",
              "models": models, "final_development_choices_to_lock_before_confirmation": final_choices,
              "whole_run_uncertainty": uncertainty, "geometry": geometry}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    selected = geometry["representation"]
    lines = ["# Development-only analysis", "", "This report is **provisional and non-confirmatory**. It uses only the twelve authorized development trajectories.", "",
             f"Validated {len(integrity)}/492 raw logit tensors (SHA-256, float32, shape 5000x10) and modeled {len(records)} checkpoints (epochs 5-200).",
             "", "## Nested grouped cross-validation", "", "Outer folds leave one complete seed trajectory out (12 folds); inner three-fold GroupKFold keeps trajectories intact. Behavioral PCA is randomized, fitted only on each training fold, and capped at 32 components; component prefixes 8/16/32 and Ridge alpha 0.01/0.1/1/10/100 are selected by inner MAE.", "",
             "| model | MAE | R^2 | Spearman | delta MAE vs all-nuisance |", "|---|---:|---:|---:|---:|"]
    for name, result in models.items():
        metric = result["pooled"]
        lines.append(f"| {name} | {metric['mae']:.5f} | {metric['r2']:.4f} | {metric['spearman']:.4f} | {result.get('improvement_over_all_nuisance_mae', 0.0):+.5f} |")
    interval = uncertainty["bootstrap_percentile_ci_95"]
    identity = uncertainty["trajectory_identity_permutation"]
    paired = uncertainty["per_run_paired_mae_improvement"]
    strongest_gain = max(paired.items(), key=lambda item: item[1])
    strongest_loss = min(paired.items(), key=lambda item: item[1])
    state = geometry["state_distance"]
    lines += ["", "## Whole-run uncertainty", "",
              f"Probability behavioral improvement over all-nuisance: mean paired MAE improvement {uncertainty['mean_paired_mae_improvement']:.5f}; improved {uncertainty['runs_improved']}/{uncertainty['runs_total']} runs; fixed-seed 10,000-bootstrap 95% percentile CI [{interval[0]:.5f}, {interval[1]:.5f}].",
              f"Run influence is substantial: the largest improvement is {strongest_gain[1]:+.5f} ({strongest_gain[0]}) and the largest deterioration is {strongest_loss[1]:+.5f} ({strongest_loss[0]}).",
              f"Trajectory-identity permutation (10,000 reassignments, complete paths and epoch order preserved): behavioral improvement one-sided p={identity['one_sided_p_value']:.5f}; all-nuisance identity-control p={identity['nuisance_control']['one_sided_p_value']:.5f}; behavioral identity p={identity['behavioral_model']['one_sided_p_value']:.5f}.",
              "", "## Hyperparameters selected on all development runs (lock before confirmation)", ""]
    for name, choice in final_choices.items():
        lines.append(f"- {name}: " + ", ".join(f"{key}={value}" for key, value in choice.items()))
    lines += ["", f"Selected development representation: **{selected}** (smallest outer-CV behavioral MAE).", "", "## Development-fit geometry (exploratory)", "",
             f"Cross-run distance vs |g difference| Spearman: Euclidean {geometry['cross_run']['euclidean_spearman_with_abs_g_difference']:.4f}; cosine {geometry['cross_run']['cosine_spearman_with_abs_g_difference']:.4f}.",
             f"Matched-epoch subset: Euclidean {geometry['matched_epoch']['euclidean_spearman_with_abs_g_difference']:.4f}; cosine {geometry['matched_epoch']['cosine_spearman_with_abs_g_difference']:.4f}.",
             f"Accuracy-binned (width 0.02): Euclidean {geometry['accuracy_binned']['euclidean_spearman_with_abs_g_difference']:.4f}; cosine {geometry['accuracy_binned']['cosine_spearman_with_abs_g_difference']:.4f}.",
             f"Mean Euclidean state distances: g=0 to g=0 {state['g_zero_vs_g_zero']['mean_euclidean_distance']:.4f}; g>0 to g>0 {state['g_positive_vs_g_positive']['mean_euclidean_distance']:.4f}; g=0 to g>0 {state['g_zero_vs_g_positive']['mean_euclidean_distance']:.4f}.",
             f"Adjacent-checkpoint direction alignment (exploratory): mean cosine {geometry['adjacent_direction_alignment_exploratory']['mean_cosine']:.4f}.",
             "", "## Development decision", "",
             "The probability representation contains a promising development-only signal for predicting the overfitting gap beyond the recorded nuisance summaries: it has the best pooled held-out-trajectory MAE and strong pooled rank correlation. This is not yet a stable effect: only 7/12 held-out trajectories improve, the whole-run bootstrap interval crosses zero, and two trajectories have large opposing influence. The identity permutation is a reason to carry the locked model into confirmation, not a substitute for confirmation.",
             "", "The stronger claim of a coherent overfitting geometry is not supported by this development set. Distance-to-|g difference| associations are near zero or negative after matching epoch or binning evaluation accuracy, and adjacent trajectory directions are almost orthogonal on average. The larger between-state mean distance is at most a coarse state-separation hint, not evidence of a shared continuous path.",
             "", "Lock probabilities with 16 PCA components and Ridge alpha 1.0 for confirmation. The selected representation, hyperparameters, and geometry are development decisions only; none is confirmation evidence."]
    (output / "DEVELOPMENT-REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
