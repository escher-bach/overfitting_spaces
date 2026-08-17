from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import canonical_sha256

PARTITION_SIZES = {"train": 10_000, "probe": 5_000, "recipe_validation": 5_000, "reserve": 30_000}


def _labels_digest(labels: list[int]) -> str:
    return hashlib.sha256(np.asarray(labels, dtype=np.int64).tobytes()).hexdigest()


def make_split_manifest(labels: list[int], split_seed: int = 1729) -> dict[str, Any]:
    """Create the fixed class-stratified partition without using image pixels."""
    if len(labels) != 50_000:
        raise ValueError(f"expected CIFAR-10 training labels, got {len(labels)}")
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[int(label)].append(index)
    if sorted(grouped) != list(range(10)) or any(len(v) != 5_000 for v in grouped.values()):
        raise ValueError("source labels are not the canonical balanced CIFAR-10 training set")
    randomizer = random.Random(split_seed)
    per_class = {name: size // 10 for name, size in PARTITION_SIZES.items()}
    partitions = {name: [] for name in PARTITION_SIZES}
    for label, indices in sorted(grouped.items()):
        indices = indices.copy()
        randomizer.shuffle(indices)
        cursor = 0
        for name in ("train", "probe", "recipe_validation", "reserve"):
            take = per_class[name]
            partitions[name].extend(indices[cursor:cursor + take])
            cursor += take
    for values in partitions.values():
        values.sort()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "CIFAR10",
        "source": "torchvision.datasets.CIFAR10.train",
        "split_seed": split_seed,
        "source_label_sha256": _labels_digest(labels),
        "normalization": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]},
        "partitions": partitions,
        "class_counts": {name: dict(sorted(Counter(labels[i] for i in indices).items())) for name, indices in partitions.items()},
    }
    manifest["sha256"] = canonical_sha256(manifest)
    return manifest


def write_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = manifest.pop("sha256", None)
    actual = canonical_sha256(manifest)
    manifest["sha256"] = expected
    if expected != actual:
        raise ValueError(f"manifest checksum mismatch: {actual} != {expected}")
    return manifest


def _cifar(root: str | Path, train: bool, download: bool):
    from torchvision.datasets import CIFAR10

    return CIFAR10(root=str(root), train=train, download=download)


def materialize_cifar(root: str | Path, *, train: bool, download: bool, mean: list[float], std: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = _cifar(root, train, download)
    images = torch.from_numpy(np.asarray(dataset.data)).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    images.sub_(torch.tensor(mean).view(1, 3, 1, 1)).div_(torch.tensor(std).view(1, 3, 1, 1))
    return images.contiguous(), torch.tensor(dataset.targets, dtype=torch.long)


def build_loaders(config: dict[str, Any], manifest: dict[str, Any], *, seed: int) -> dict[str, DataLoader]:
    data, training = config["data"], config["training"]
    images, labels = materialize_cifar(data["root"], train=True, download=data.get("download", False), mean=data["normalization_mean"], std=data["normalization_std"])
    if _labels_digest(labels.tolist()) != manifest["source_label_sha256"]:
        raise ValueError("downloaded CIFAR labels do not match split manifest")
    eval_images, eval_labels = materialize_cifar(data["root"], train=False, download=data.get("download", False), mean=data["normalization_mean"], std=data["normalization_std"])
    train_options: dict[str, Any] = {
        "batch_size": training["batch_size"],
        "num_workers": training["loader_workers"],
        "pin_memory": training.get("pin_memory", True),
    }
    if train_options["num_workers"]:
        train_options.update({
            "persistent_workers": training.get("persistent_workers", True),
            "prefetch_factor": training.get("prefetch_factor", 2),
        })
    evaluation_options: dict[str, Any] = {
        "batch_size": training["batch_size"],
        "num_workers": training.get("evaluation_loader_workers", 0),
        "pin_memory": training.get("pin_memory", True),
    }
    if evaluation_options["num_workers"]:
        evaluation_options.update({
            "persistent_workers": training.get("persistent_workers", True),
            "prefetch_factor": training.get("prefetch_factor", 2),
        })
    generator = torch.Generator().manual_seed(seed + 1)
    def worker_init(worker_id: int) -> None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
    indices = manifest["partitions"]
    train = TensorDataset(images[indices["train"]], labels[indices["train"]])
    recipe = TensorDataset(images[indices["recipe_validation"]], labels[indices["recipe_validation"]])
    # This dataset intentionally has no label tensor: extraction APIs cannot
    # accidentally hand probe labels to representations or downstream models.
    probe = TensorDataset(images[indices["probe"]])
    return {
        "train": DataLoader(train, shuffle=True, generator=generator, worker_init_fn=worker_init, **train_options),
        "recipe_validation": DataLoader(recipe, shuffle=False, worker_init_fn=worker_init, **evaluation_options),
        "probe": DataLoader(probe, shuffle=False, worker_init_fn=worker_init, **evaluation_options),
        "evaluation": DataLoader(TensorDataset(eval_images, eval_labels), shuffle=False, worker_init_fn=worker_init, **evaluation_options),
    }
