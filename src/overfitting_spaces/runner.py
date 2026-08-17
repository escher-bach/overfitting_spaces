from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from .artifacts import RunArtifacts, sha256_file, write_json
from .config import canonical_sha256, load_config
from .data import _cifar, build_loaders, load_manifest, make_split_manifest, write_manifest
from .model import cifar_resnet18
from .train import benchmark_loader, configure_performance, evaluate, export_probe_logits, train_epoch


def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def cuda_inventory() -> list[dict[str, Any]]:
    return [{"index": i, "name": torch.cuda.get_device_name(i), "capability": torch.cuda.get_device_capability(i), "memory_bytes": torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]


def validate_hardware(config: dict[str, Any]) -> list[dict[str, Any]]:
    expected, inventory = config["hardware"], cuda_inventory()
    if len(inventory) != expected["expected_cuda_devices"]:
        raise RuntimeError(f"expected {expected['expected_cuda_devices']} CUDA devices, observed {len(inventory)}: {inventory}")
    if expected.get("require_t4") and any(expected["expected_gpu_name"].lower() not in gpu["name"].lower() for gpu in inventory):
        raise RuntimeError(f"expected {expected['expected_gpu_name']} GPUs, observed {inventory}")
    return inventory


def capture_environment(config: dict[str, Any]) -> dict[str, Any]:
    import torchvision

    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
            "devices": cuda_inventory() if torch.cuda.is_available() else [], "config_hash": canonical_sha256(config),
            "command": sys.argv, "git_sha": os.environ.get("OVERFIT_GIT_SHA"),
            "cudnn_benchmark": True, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def phase(run_dir: Path, name: str, state: str, **extra: Any) -> None:
    path = run_dir / "phase_status.json"; values = json.loads(path.read_text()) if path.exists() else {"phases": []}
    values["phases"].append({"phase": name, "state": state, "at": time.time(), **extra}); write_json(path, values)


def nuisance(logits: torch.Tensor) -> dict[str, Any]:
    probability = logits.softmax(-1); predictions = probability.argmax(-1)
    return {"mean_logit_norm": logits.norm(dim=-1).mean().item(), "mean_entropy": (-(probability * probability.clamp_min(1e-12).log()).sum(-1)).mean().item(), "mean_max_probability": probability.max(-1).values.mean().item(), "predicted_class_histogram": torch.bincount(predictions, minlength=logits.shape[-1]).tolist()}


def checkpoint_record(model, loaders, device, manifest, seed_dir: Path, epoch: int, config: dict[str, Any]) -> dict[str, Any]:
    evaluation_key = "recipe_validation" if config["run"]["mode"] == "pilot" else "evaluation"
    logits = export_probe_logits(model, loaders["probe"], device); path = seed_dir / "logits" / f"epoch-{epoch:03d}.safetensors"; path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"logits": logits}, str(path), metadata={"probe_manifest_sha256": manifest["sha256"], "epoch": str(epoch), "seed": str(seed_dir.name)})
    return {"epoch": epoch, "evaluation": evaluate(model, loaders[evaluation_key], device), "probe_logits": {"path": str(path.relative_to(seed_dir)), "sha256": sha256_file(path), "shape": list(logits.shape)}, "nuisance": nuisance(logits)}


def preflight_worker(config: dict[str, Any], manifest: dict[str, Any], run_dir: Path, physical_gpu: int) -> None:
    # CUDA_VISIBLE_DEVICES is set by the parent before this child starts. Each
    # worker sees exactly one T4 as cuda:0 and independently exercises its I/O.
    seed_everything(config["run"]["root_seeds"][physical_gpu]); configure_performance(config); device = torch.device("cuda:0")
    loaders = build_loaders(config, manifest, seed=config["run"]["root_seeds"][physical_gpu])
    model = cifar_resnet18().to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["training"]["learning_rate"], momentum=config["training"]["momentum"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["training"].get("amp", True))
    input_metrics = benchmark_loader(loaders["train"], config["training"].get("benchmark_steps", len(loaders["train"])))
    training_metrics = train_epoch(model, loaders["train"], optimizer, scaler, device, config["training"].get("amp", True))
    input_headroom = input_metrics["images_per_second"] / training_metrics["images_per_second"]
    minimum_headroom = config["training"].get("minimum_input_headroom", 1.25)
    report = {
        "physical_gpu": physical_gpu,
        "visible_device": torch.cuda.get_device_name(0),
        "batch_size": config["training"]["batch_size"],
        "workers_per_seed": config["training"]["loader_workers"],
        "input_pipeline": input_metrics,
        "training": training_metrics,
        "input_headroom": input_headroom,
        "minimum_input_headroom": minimum_headroom,
        "input_gate_passed": input_headroom >= minimum_headroom,
    }
    write_json(run_dir / "analysis" / f"preflight-gpu-{physical_gpu}.json", report)
    if input_headroom < minimum_headroom:
        raise RuntimeError(f"GPU {physical_gpu} input headroom {input_headroom:.2f} is below {minimum_headroom:.2f}")


def preflight(config: dict[str, Any], artifacts: RunArtifacts) -> None:
    inventory = validate_hardware(config); data, manifest_path = config["data"], Path(config["data"]["manifest"])
    if not manifest_path.is_file():
        dataset = _cifar(data["root"], train=True, download=data.get("download", False)); write_manifest(make_split_manifest(dataset.targets), manifest_path)
    _cifar(data["root"], train=False, download=data.get("download", False))
    manifest = load_manifest(manifest_path); write_json(artifacts.run_dir / "data_manifest.json", manifest); phase(artifacts.run_dir, "preflight", "running", devices=inventory)
    children = []
    for gpu in range(len(inventory)):
        children.append(subprocess.Popen([sys.executable, "-m", "overfitting_spaces.runner", "--config", os.environ["OVERFIT_CONFIG_PATH"], "--output-root", str(artifacts.root), "--run-id", artifacts.run_id, "--preflight-worker", str(gpu)], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}))
    codes = [child.wait() for child in children]
    if any(codes): raise RuntimeError(f"preflight device workers failed: {codes}")
    reports = [json.loads(path.read_text()) for path in sorted((artifacts.run_dir / "analysis").glob("preflight-gpu-*.json"))]
    if len(reports) != len(inventory): raise RuntimeError("missing one or more preflight GPU reports")
    write_json(artifacts.run_dir / "analysis" / "preflight_report.json", {"devices": inventory, "per_gpu": reports, "concurrent_seed_processes": len(reports), "no_ddp": True})
    phase(artifacts.run_dir, "preflight", "complete")


def run_seed(config: dict[str, Any], manifest: dict[str, Any], run_dir: Path, seed: int) -> None:
    seed_dir = run_dir / "seeds" / f"seed-{seed}"; seed_dir.mkdir(parents=True, exist_ok=True)
    try:
        seed_everything(seed); configure_performance(config); device = torch.device("cuda:0"); loaders = build_loaders(config, manifest, seed=seed)
        model = cifar_resnet18(config["model"]["classes"]).to(device)
        if config["training"].get("channels_last", True): model = model.to(memory_format=torch.channels_last)
        optimizer = torch.optim.SGD(model.parameters(), lr=config["training"]["learning_rate"], momentum=config["training"]["momentum"], weight_decay=config["training"]["weight_decay"])
        scheduler_name = config["training"].get("scheduler")
        if scheduler_name != "cosine":
            raise ValueError(f"unsupported scheduler {scheduler_name!r}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
        scaler = torch.amp.GradScaler("cuda", enabled=config["training"].get("amp", True)); history = [checkpoint_record(model, loaders, device, manifest, seed_dir, 0, config)]
        for epoch in range(1, config["training"]["epochs"] + 1):
            record = {"epoch": epoch, "train": train_epoch(model, loaders["train"], optimizer, scaler, device, config["training"].get("amp", True)), "learning_rate": optimizer.param_groups[0]["lr"]}
            scheduler.step()
            if epoch % config["training"]["checkpoint_every_epochs"] == 0: record.update(checkpoint_record(model, loaders, device, manifest, seed_dir, epoch, config))
            history.append(record)
        write_json(seed_dir / "metrics" / "history.json", history); write_json(seed_dir / "SUCCESS.json", {"seed": seed})
    except Exception as error:
        write_json(seed_dir / "failure.json", {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}); raise


def run(config_path: Path, output_root: Path, run_id: str | None = None, worker_seed: int | None = None, preflight_gpu: int | None = None) -> None:
    config = load_config(config_path)
    if worker_seed is not None:
        run_seed(config, load_manifest(config["data"]["manifest"]), output_root / run_id, worker_seed); return
    if preflight_gpu is not None:
        preflight_worker(config, load_manifest(config["data"]["manifest"]), output_root / run_id, preflight_gpu); return
    artifacts = RunArtifacts(output_root, run_id or f"run-{uuid.uuid4().hex[:12]}")
    write_json(artifacts.run_dir / "resolved_config.json", config); write_json(artifacts.run_dir / "environment.json", capture_environment(config)); phase(artifacts.run_dir, "capture_environment", "complete")
    run_manifest = {"schema_version": 1, "run_id": artifacts.run_id, "git_sha": os.environ.get("OVERFIT_GIT_SHA"), "config_sha256": canonical_sha256(config), "root_seeds": config["run"]["root_seeds"], "manifest_sha256": None}
    write_json(artifacts.run_dir / "run_manifest.json", run_manifest)
    write_json(artifacts.run_dir / "analysis" / "provenance.json", {**run_manifest, "accelerator_inventory": capture_environment(config)["devices"]})
    try:
        os.environ["OVERFIT_CONFIG_PATH"] = str(config_path.resolve())
        if config["run"]["mode"] == "preflight":
            preflight(config, artifacts)
            manifest = load_manifest(config["data"]["manifest"])
            run_manifest["manifest_sha256"] = manifest["sha256"]
            write_json(artifacts.run_dir / "run_manifest.json", run_manifest)
            provenance = json.loads((artifacts.run_dir / "analysis" / "provenance.json").read_text())
            provenance["manifest_sha256"] = manifest["sha256"]
            write_json(artifacts.run_dir / "analysis" / "provenance.json", provenance)
            artifacts.package(True)
            return
        inventory, manifest, seeds = validate_hardware(config), load_manifest(config["data"]["manifest"]), config["run"]["root_seeds"]
        run_manifest["manifest_sha256"] = manifest["sha256"]
        write_json(artifacts.run_dir / "run_manifest.json", run_manifest)
        provenance = json.loads((artifacts.run_dir / "analysis" / "provenance.json").read_text())
        provenance["manifest_sha256"] = manifest["sha256"]
        write_json(artifacts.run_dir / "analysis" / "provenance.json", provenance)
        phase(artifacts.run_dir, "training", "running", seeds=seeds, scheduling="one independent seed process per GPU; no DDP")
        width = min(config["hardware"]["concurrent_seed_processes"], len(inventory))
        codes: list[int] = []
        for start in range(0, len(seeds), width):
            wave = [subprocess.Popen([sys.executable, "-m", "overfitting_spaces.runner", "--config", str(config_path), "--output-root", str(output_root), "--run-id", artifacts.run_id, "--worker-seed", str(seed)], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}) for gpu, seed in enumerate(seeds[start:start + width])]
            codes.extend(child.wait() for child in wave)
        if any(codes): raise RuntimeError(f"seed worker exit codes: {codes}")
        phase(artifacts.run_dir, "training", "complete"); artifacts.package(True)
    except Exception as error:
        phase(artifacts.run_dir, "failure", "failed", error=type(error).__name__); write_json(artifacts.run_dir / "failure.json", {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}); artifacts.package(False); raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output-root", type=Path, required=True); parser.add_argument("--run-id"); parser.add_argument("--worker-seed", type=int); parser.add_argument("--preflight-worker", type=int); args = parser.parse_args(argv); run(args.config, args.output_root, args.run_id, args.worker_seed, args.preflight_worker)


if __name__ == "__main__": main()
