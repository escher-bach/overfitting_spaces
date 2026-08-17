from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


class RunArtifacts:
    def __init__(self, output_root: str | Path, run_id: str) -> None:
        self.root = Path(output_root)
        self.run_id = run_id
        self.run_dir = self.root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def package(self, success: bool) -> dict[str, Any]:
        (self.run_dir / ("SUCCESS" if success else "FAILURE")).write_text("\n", encoding="utf-8")
        write_json(self.run_dir / "analysis" / "summary.json", {"run_id": self.run_id, "success": success})
        analysis = self.root / f"{self.run_id}-analysis.tar.gz"
        recovery = self.root / f"{self.run_id}.tar.gz"
        # Raw trajectory logits and resumable state stay in the remote recovery
        # payload. A downstream Kaggle analysis run can attach that output
        # without routing heavyweight tensors through the operator's device.
        excluded = {"recovery", "checkpoints", "logits"}
        with tarfile.open(analysis, "w:gz") as archive:
            for path in self.run_dir.rglob("*"):
                if path.is_file() and not any(part in excluded for part in path.relative_to(self.run_dir).parts):
                    archive.add(path, arcname=f"{self.run_id}/{path.relative_to(self.run_dir).as_posix()}")
        with tarfile.open(recovery, "w:gz") as archive:
            archive.add(self.run_dir, arcname=self.run_id)
        run_manifest_path = self.run_dir / "run_manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8")) if run_manifest_path.is_file() else {}
        report = {"run_id": self.run_id, "success": success,
                  "git_sha": run_manifest.get("git_sha"),
                  "config_sha256": run_manifest.get("config_sha256"),
                  "manifest_sha256": run_manifest.get("manifest_sha256"),
                  "analysis": {"path": analysis.name, "bytes": analysis.stat().st_size, "sha256": sha256_file(analysis)},
                  "recovery": {"path": recovery.name, "bytes": recovery.stat().st_size, "sha256": sha256_file(recovery)}}
        for path, section in ((analysis, "analysis"), (recovery, "recovery")):
            (path.with_suffix("").with_suffix(".sha256")).write_text(f"{report[section]['sha256']}  {path.name}\n", encoding="utf-8")
        write_json(self.root / "latest-summary.json", report)
        return report
