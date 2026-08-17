import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "kaggle_run.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("kaggle_run", TOOL); module = importlib.util.module_from_spec(spec); assert spec and spec.loader; sys.modules[spec.name] = module; spec.loader.exec_module(module); return module


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
