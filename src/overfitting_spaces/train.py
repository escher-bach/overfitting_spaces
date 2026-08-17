from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn


def configure_performance(config: dict[str, Any]) -> None:
    options = config["training"]
    torch.backends.cuda.matmul.allow_tf32 = bool(options.get("tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(options.get("tf32", True))
    torch.backends.cudnn.benchmark = True


def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device, dtype=torch.long)
    total_count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            total_loss += torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
            total_correct += (logits.argmax(1) == labels).sum()
            total_count += len(labels)
    return {"loss": total_loss.item() / total_count, "accuracy": total_correct.item() / total_count}


def train_epoch(model: nn.Module, loader, optimizer, scaler, device: torch.device, amp: bool) -> dict[str, float]:
    model.train()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    count = 0
    started = time.perf_counter()
    for images, labels in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(images); loss = torch.nn.functional.cross_entropy(logits, labels)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        loss_sum += loss.detach() * len(labels)
        correct += (logits.detach().argmax(1) == labels).sum()
        count += len(labels)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {"loss": loss_sum.item() / count, "accuracy": correct.item() / count, "images_per_second": count / elapsed}


def benchmark_loader(loader, steps: int) -> dict[str, float | int]:
    """Measure the host input path while persistent workers are active."""
    if steps <= 0:
        raise ValueError("benchmark steps must be positive")
    iterator = iter(loader)
    # Exclude one-time worker startup from the steady-state starvation gate.
    try:
        next(iterator)
    except StopIteration:
        iterator = iter(loader)
        next(iterator)
    images = 0
    completed = 0
    started = time.perf_counter()
    while completed < steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        images += len(batch[0])
        completed += 1
    elapsed = time.perf_counter() - started
    return {
        "steps": completed,
        "images": images,
        "elapsed_seconds": elapsed,
        "images_per_second": images / elapsed,
    }


def export_probe_logits(model: nn.Module, loader, device: torch.device) -> torch.Tensor:
    model.eval(); batches = []
    with torch.inference_mode():
        for (images,) in loader:
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            batches.append(model(images).float().cpu())
    return torch.cat(batches).contiguous()
