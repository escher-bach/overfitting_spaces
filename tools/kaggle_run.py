"""Thin, explicit control plane around the official Kaggle CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "kaggle" / "experiments.toml"
AUDIT = ROOT / "audit" / "runs"
ANALYSIS_PATTERN = r".*(latest-summary\.json|-analysis\.tar\.gz|-analysis\.sha256)$"
TERMINAL = {"COMPLETE", "ERROR", "CANCEL"}


def now() -> str: return datetime.now(timezone.utc).isoformat()
def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    with path.open("rb") as handle: return tomllib.load(handle)
def git(*args: str) -> str:
    result = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT, text=True, capture_output=True)
    if result.returncode: raise SystemExit(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()
def kaggle(*args: str, check: bool = True, timeout: int = 900) -> str:
    executable = shutil.which("kaggle")
    if not executable: raise SystemExit("official Kaggle CLI is not on PATH")
    result = subprocess.run([executable, *args], text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout, env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    output = (result.stdout or "") + (result.stderr or "")
    if check and result.returncode: raise SystemExit(f"kaggle {' '.join(args)} failed:\n{output}")
    return output
def resolve(name: str, registry: dict[str, Any] | None = None) -> dict[str, Any]:
    registry = registry or load_registry(); entry = registry.get("experiments", {}).get(name)
    if not entry: raise SystemExit(f"unknown experiment {name!r}")
    return {**entry, "name": name, "owner": registry["owner"], "git_remote": registry["git_remote"], "accelerator": entry.get("accelerator", registry["accelerator"]), "launcher_template": registry["launcher_template"], "output_root": registry["output_root"]}
def commit(value: str | None) -> str:
    value = git("rev-parse", value or "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", value): raise SystemExit("resolved commit is not a full SHA")
    return value
def slug(experiment: dict[str, Any], sha: str) -> str: return f"{experiment['slug_prefix']}-{sha[:7]}"
def kernel(experiment: dict[str, Any], sha: str) -> str: return f"{experiment['owner']}/{slug(experiment, sha)}"
def manifest_hash(path: Path) -> str:
    source = json.loads(path.read_text(encoding="utf-8")); return source["sha256"]
def config_hash(path: Path) -> str:
    with path.open("rb") as handle: value = tomllib.load(handle)
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
def assert_remote_reachable(sha: str, remote: str) -> None:
    refs = git("ls-remote", "--heads", "--tags", remote).splitlines()
    for line in refs:
        ref = line.split()[0]
        if subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "merge-base", "--is-ancestor", sha, ref], cwd=ROOT).returncode == 0: return
    raise SystemExit(f"{sha} is not reachable on {remote}; this tool never pushes")
def render(sha: str, experiment: dict[str, Any]) -> str:
    template = git("show", f"{sha}:{experiment['launcher_template']}")
    url = git("remote", "get-url", experiment["git_remote"])
    values = {"__GIT_COMMIT__": sha, "__CONFIG_REL__": experiment["config"], "__REPO_URL__": url}
    for marker, value in values.items():
        if template.count(marker) != 1: raise SystemExit(f"template must contain exactly one {marker}")
        template = template.replace(marker, value)
    json.loads(template); return template
def validate(experiment: dict[str, Any], sha: str) -> None:
    for path in (experiment["config"], experiment["manifest"], experiment["launcher_template"], "kaggle/experiments.toml", "requirements-kaggle.txt", "pyproject.toml", "src", "tools/kaggle_run.py"):
        if git("status", "--porcelain", "--", path): raise SystemExit(f"uncommitted experiment path: {path}")
    config, manifest = ROOT / experiment["config"], ROOT / experiment["manifest"]
    if not config.is_file(): raise SystemExit("experiment config must exist")
    expected = experiment["manifest_sha256"]
    if experiment.get("manifest_bootstrap"):
        if expected != "GENERATED_AT_PREFLIGHT": raise SystemExit("only the explicit preflight bootstrap may omit a committed manifest hash")
    elif not manifest.is_file() or expected.startswith("PENDING") or manifest_hash(manifest) != expected:
        raise SystemExit("registry manifest hash is not the generated manifest hash")
    assert_remote_reachable(sha, experiment["git_remote"]); render(sha, experiment)
def stage(sha: str, experiment: dict[str, Any], directory: Path) -> None:
    name = slug(experiment, sha); (directory / f"{name}.ipynb").write_text(render(sha, experiment), encoding="utf-8")
    metadata = {"id": kernel(experiment, sha), "title": name, "code_file": f"{name}.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": experiment["accelerator"]}
    if experiment.get("dataset_sources"):
        metadata["dataset_sources"] = experiment["dataset_sources"]
    if experiment.get("kernel_sources"):
        metadata["kernel_sources"] = experiment["kernel_sources"]
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
def status(exact_version: str) -> str:
    result = kaggle("kernels", "status", exact_version, check=False); found = re.search(r'status "(?:KernelWorkerStatus\.)?([A-Z_]+)"', result)
    return found.group(1) if found else "UNKNOWN"
def reference_path(experiment: dict[str, Any], sha: str) -> Path: return AUDIT / "submitted" / f"{slug(experiment, sha)}.json"
def launch(experiment: dict[str, Any], sha: str) -> dict[str, Any]:
    validate(experiment, sha)
    requested = kernel(experiment, sha)
    current = status(requested)
    if current not in TERMINAL | {"UNKNOWN"}:
        raise SystemExit(f"{requested} is already {current}; refusing a concurrent mutable-kernel submission")
    with tempfile.TemporaryDirectory(prefix="overfit-kaggle-") as temp:
        stage(sha, experiment, Path(temp)); result = kaggle("kernels", "push", "-p", temp, "--accelerator", experiment["accelerator"])
    version = re.search(r"version\s+(\d+)", result)
    if not version: raise SystemExit(f"could not capture immutable Kaggle version: {result}")
    returned = re.search(r"kaggle\.com/(?:code/)?([\w.-]+/[\w.-]+)", result)
    if returned and returned.group(1) != requested: raise SystemExit("Kaggle title-derived slug differs from requested commit-derived slug")
    reference = {"experiment": experiment["name"], "kernel": requested, "version": int(version.group(1)), "exact_version": f"{requested}/{version.group(1)}", "url": f"https://www.kaggle.com/code/{requested}", "git_sha": sha, "git_remote_url": git("remote", "get-url", experiment["git_remote"]), "config": experiment["config"], "config_sha256": config_hash(ROOT / experiment["config"]), "manifest_sha256": experiment["manifest_sha256"], "submitted_at": now()}
    path = reference_path(experiment, sha); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8"); return reference
def load_reference(experiment: dict[str, Any], sha: str) -> dict[str, Any]:
    path = reference_path(experiment, sha)
    if not path.is_file(): raise SystemExit(f"no saved submission reference at {path}")
    return json.loads(path.read_text(encoding="utf-8"))
def collect(reference: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory(prefix="overfit-collect-") as temp:
        folder = Path(temp); kaggle("kernels", "output", reference["exact_version"], "-p", temp, "--file-pattern", ANALYSIS_PATTERN, "-o", "-q")
        summary_files = list(folder.rglob("latest-summary.json")); archives = list(folder.rglob("*-analysis.tar.gz")); sidecars = list(folder.rglob("*-analysis.sha256"))
        if len(summary_files) != 1 or len(archives) != 1 or len(sidecars) != 1: raise SystemExit("missing compact analysis output; failed runs should still be collected")
        digest = hashlib.sha256(archives[0].read_bytes()).hexdigest(); expected = sidecars[0].read_text().split()[0]; summary = json.loads(summary_files[0].read_text())
        if digest != expected or digest != summary["analysis"]["sha256"]: raise SystemExit("analysis payload checksum mismatch")
        if summary.get("git_sha") != reference["git_sha"]: raise SystemExit("run Git SHA does not match submitted commit")
        if summary.get("config_sha256") != reference["config_sha256"]: raise SystemExit("run config hash does not match submitted config")
        if reference["manifest_sha256"] != "GENERATED_AT_PREFLIGHT" and summary.get("manifest_sha256") != reference["manifest_sha256"]: raise SystemExit("run data manifest does not match the declared manifest")
        audit = AUDIT / summary["run_id"]; audit.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archives[0]) as archive: archive.extractall(folder / "payload", filter="data")
        unpacked = folder / "payload" / summary["run_id"]
        for relative in ("analysis/summary.json", "analysis/provenance.json", "analysis/result-report.json", "phase_status.json", "environment.json", "data_manifest.json"):
            source = unpacked / relative
            if source.is_file(): shutil.copyfile(source, audit / Path(relative).name)
        for source in sorted((unpacked / "analysis").glob("preflight*.json")):
            shutil.copyfile(source, audit / source.name)
        receipt = {"schema_version": 1, "run_id": summary["run_id"], "experiment": reference["experiment"], "kaggle": {"kernel": reference["kernel"], "exact_version": reference["exact_version"], "url": reference["url"], "terminal_status": status(reference["exact_version"])}, "git": {"sha": reference["git_sha"], "remote_url": reference["git_remote_url"]}, "config": {"path": reference["config"], "sha256": reference["config_sha256"], "manifest_sha256": summary.get("manifest_sha256")}, "analysis_artifact": {"name": archives[0].name, "bytes": archives[0].stat().st_size, "sha256": digest}, "recovery_artifact": {**summary.get("recovery", {}), "location": f"Kaggle output of {reference['exact_version']}"}, "success": summary["success"], "collected_at": now()}
        (audit / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"receipt": str(audit / "receipt.json"), "success": summary["success"]}, indent=2))
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "launch", "status", "logs", "collect"):
        item = sub.add_parser(command); item.add_argument("--experiment", required=True); item.add_argument("--commit")
    args = parser.parse_args(argv); experiment = resolve(args.experiment); sha = commit(args.commit)
    if args.command == "validate": validate(experiment, sha); print("validation passed")
    elif args.command == "launch": print(json.dumps(launch(experiment, sha), indent=2))
    else:
        reference = load_reference(experiment, sha)
        if args.command == "status": print(json.dumps({"exact_version": reference["exact_version"], "status": status(reference["exact_version"])}, indent=2))
        elif args.command == "logs": print(kaggle("kernels", "logs", reference["exact_version"], check=False))
        else: collect(reference)
if __name__ == "__main__": main()
