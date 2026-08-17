import json

from overfitting_spaces.artifacts import RunArtifacts


def test_artifact_round_trip_writes_compact_and_recovery_payloads(tmp_path):
    artifacts = RunArtifacts(tmp_path, "run-1")
    (artifacts.run_dir / "analysis").mkdir(); (artifacts.run_dir / "analysis" / "summary.json").write_text("{}")
    (artifacts.run_dir / "recovery").mkdir(); (artifacts.run_dir / "recovery" / "model.pt").write_bytes(b"weights")
    summary = artifacts.package(True)
    assert summary["analysis"]["bytes"] < summary["recovery"]["bytes"]
    assert json.loads((tmp_path / "latest-summary.json").read_text()) == summary


def test_analysis_payload_excludes_raw_logits(tmp_path):
    import tarfile

    artifacts = RunArtifacts(tmp_path, "run-2")
    logits = artifacts.run_dir / "seeds" / "seed-1" / "logits" / "epoch-000.safetensors"
    logits.parent.mkdir(parents=True)
    logits.write_bytes(b"large trajectory tensor")
    artifacts.package(True)
    with tarfile.open(tmp_path / "run-2-analysis.tar.gz") as archive:
        assert not [name for name in archive.getnames() if "/logits/" in name]
    with tarfile.open(tmp_path / "run-2.tar.gz") as archive:
        assert [name for name in archive.getnames() if "/logits/" in name]
