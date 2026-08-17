import importlib.util
import json
import sys
import tempfile
import tomllib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "kaggle_run.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("kaggle_run", TOOL); module = importlib.util.module_from_spec(spec); assert spec and spec.loader; sys.modules[spec.name] = module; spec.loader.exec_module(module); return module


def load_config(path: Path):
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_registry_and_commit_slug_contract():
    tool = load_tool(); experiment = tool.resolve("preflight")
    assert tool.slug(experiment, "a" * 40) == "overfit-preflight-aaaaaaa"
    assert tool.kernel(experiment, "a" * 40).endswith("/overfit-preflight-aaaaaaa")


def test_rendered_launcher_has_no_training_logic(monkeypatch):
    tool = load_tool(); experiment = tool.resolve("preflight")
    monkeypatch.setattr(tool, "git", lambda *args: "https://example.invalid/repo.git" if args[:3] == ("remote", "get-url", "origin") else (ROOT / "kaggle/launcher-template.ipynb").read_text())
    rendered = json.loads(tool.render("a" * 40, experiment)); source = "".join(line for cell in rendered["cells"] for line in cell.get("source", []))
    assert "class ResNet" not in source and "train_epoch" not in source
    assert "/tmp/overfitting-spaces-runtime" in source and "/kaggle/working/overfitting-results" in source


def test_staged_title_equals_slug(monkeypatch):
    tool = load_tool(); experiment = tool.resolve("preflight")
    monkeypatch.setattr(tool, "render", lambda *_: json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
    with tempfile.TemporaryDirectory() as temp:
        tool.stage("b" * 40, experiment, Path(temp)); metadata = json.loads((Path(temp) / "kernel-metadata.json").read_text())
    assert metadata["title"] == "overfit-preflight-bbbbbbb"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"


def test_pilot_stages_maintained_cifar_dataset(monkeypatch):
    tool = load_tool(); experiment = tool.resolve("pilot-overfit")
    monkeypatch.setattr(tool, "render", lambda *_: json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
    with tempfile.TemporaryDirectory() as temp:
        tool.stage("c" * 40, experiment, Path(temp)); metadata = json.loads((Path(temp) / "kernel-metadata.json").read_text())
    assert metadata["dataset_sources"] == ["pankrzysiu/cifar10-python"]


def test_main_seed_assignment_is_complete_disjoint_and_recipe_frozen():
    tool = load_tool()
    assignment = json.loads((ROOT / "data/manifests/main-seed-assignment-v1.json").read_text())
    digest = bytes.fromhex(assignment["source_manifest_sha256"])
    entropy = [int.from_bytes(digest[index:index + 4], "big") for index in range(0, len(digest), 4)]
    expected = np.random.SeedSequence(entropy=entropy).generate_state(20, dtype=np.uint32).tolist()
    development = [tool.resolve(f"main-development-{batch:02d}") for batch in range(1, 7)]
    confirmation = [tool.resolve(f"main-confirmation-{batch:02d}") for batch in range(1, 5)]
    development_seeds = [seed for experiment in development for seed in load_config(ROOT / experiment["config"])["run"]["root_seeds"]]
    confirmation_seeds = [seed for experiment in confirmation for seed in load_config(ROOT / experiment["config"])["run"]["root_seeds"]]
    pilot = load_config(ROOT / "configs/pilot.toml")
    assert assignment["derivation"]["entropy"] == entropy
    assert assignment["development"] == expected[:12] == development_seeds
    assert assignment["confirmation"] == expected[12:] == confirmation_seeds
    assert not set(development_seeds) & set(confirmation_seeds)
    assert not set(pilot["run"]["root_seeds"]) & set(development_seeds + confirmation_seeds)
    for experiment in development + confirmation:
        config = load_config(ROOT / experiment["config"])
        assert config["run"]["mode"] == "main"
        assert config["training"] == pilot["training"]
        assert config["data"] == pilot["data"]
        assert experiment["dataset_sources"] == ["pankrzysiu/cifar10-python"]


def test_status_parser_uses_exact_worker_status(monkeypatch):
    tool = load_tool()
    monkeypatch.setattr(tool, "kaggle", lambda *_args, **_kwargs: 'status "KernelWorkerStatus.COMPLETE"')
    assert tool.status("owner/kernel/7") == "COMPLETE"


def test_collection_pattern_excludes_recovery_payload():
    import re

    pattern = re.compile(load_tool().ANALYSIS_PATTERN)
    assert pattern.fullmatch("overfitting-results/latest-summary.json")
    assert pattern.fullmatch("overfitting-results/run-1-analysis.tar.gz")
    assert not pattern.fullmatch("overfitting-results/run-1.tar.gz")


def test_collect_retains_failure_diagnostics(monkeypatch, tmp_path):
    import hashlib
    import tarfile

    tool = load_tool(); run_id = "run-failed"
    payload = tmp_path / "payload" / run_id
    payload.mkdir(parents=True)
    for name, contents in (("failure.json", "{}"), ("run_manifest.json", "{}"), ("resolved_config.json", "{}")):
        (payload / name).write_text(contents)
    seed_failure = payload / "seeds" / "seed-1" / "failure.json"
    seed_failure.parent.mkdir(parents=True)
    seed_failure.write_text("{}")
    (payload / "analysis").mkdir(); (payload / "analysis" / "summary.json").write_text("{}")
    archive = tmp_path / f"{run_id}-analysis.tar.gz"
    with tarfile.open(archive, "w:gz") as value:
        value.add(payload, arcname=run_id)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    summary = {"run_id": run_id, "success": False, "git_sha": "a" * 40, "config_sha256": "b", "manifest_sha256": "c", "analysis": {"sha256": digest}}

    def fake_kaggle(*args, **kwargs):
        destination = Path(args[args.index("-p") + 1])
        (destination / "latest-summary.json").write_text(json.dumps(summary))
        (destination / archive.name).write_bytes(archive.read_bytes())
        (destination / f"{run_id}-analysis.sha256").write_text(f"{digest}  {archive.name}")
        return ""

    monkeypatch.setattr(tool, "kaggle", fake_kaggle)
    monkeypatch.setattr(tool, "status", lambda *_: "ERROR")
    monkeypatch.setattr(tool, "AUDIT", tmp_path / "audit")
    reference = {"experiment": "failed", "kernel": "owner/kernel", "exact_version": "owner/kernel/1", "url": "https://example.invalid", "git_sha": "a" * 40, "git_remote_url": "https://example.invalid/repo", "config": "configs/example.toml", "config_sha256": "b", "manifest_sha256": "c"}
    tool.collect(reference)

    audit = next((tmp_path / "audit").iterdir())
    assert all((audit / name).is_file() for name in ("failure.json", "run_manifest.json", "resolved_config.json"))
    assert (audit / "seeds" / "seed-1" / "failure.json").is_file()
