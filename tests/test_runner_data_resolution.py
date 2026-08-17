from pathlib import Path

from overfitting_spaces import runner


def test_resolve_cifar_root_prefers_verified_configured_root(monkeypatch, tmp_path):
    configured = tmp_path / "configured"
    calls = []

    def fake_cifar(root, train, download):
        calls.append((Path(root), train, download))
        if Path(root) != configured:
            raise RuntimeError("not CIFAR")

    monkeypatch.setattr(runner, "_cifar", fake_cifar)
    result = runner.resolve_cifar_root(configured, kaggle_input_root=tmp_path / "input", fallback_root=tmp_path / "fallback")

    assert result == {"configured_root": str(configured), "resolved_root": str(configured), "source": "configured"}
    assert calls == [(configured, True, False), (configured, False, False)]


def test_resolve_cifar_root_discovers_verified_direct_kaggle_mount(monkeypatch, tmp_path):
    configured, input_root = tmp_path / "missing", tmp_path / "input"
    attached, invalid = input_root / "attached", input_root / "invalid"
    attached.mkdir(parents=True); invalid.mkdir()

    def fake_cifar(root, train, download):
        if Path(root) != attached:
            raise RuntimeError("not CIFAR")

    monkeypatch.setattr(runner, "_cifar", fake_cifar)
    result = runner.resolve_cifar_root(configured, kaggle_input_root=input_root, fallback_root=tmp_path / "fallback")

    assert result["resolved_root"] == str(attached)
    assert result["source"] == "kaggle_input_discovery"


def test_resolve_cifar_root_downloads_once_before_workers(monkeypatch, tmp_path):
    configured, fallback = tmp_path / "missing", tmp_path / "fallback"
    downloaded = set()
    calls = []

    def fake_cifar(root, train, download):
        root = Path(root); calls.append((root, train, download))
        if root == fallback and download:
            downloaded.add(train)
            return object()
        if root == fallback and downloaded == {True, False}:
            return object()
        raise RuntimeError("dataset missing")

    monkeypatch.setattr(runner, "_cifar", fake_cifar)
    result = runner.resolve_cifar_root(configured, kaggle_input_root=tmp_path / "input", fallback_root=fallback)

    assert result == {"configured_root": str(configured), "resolved_root": str(fallback), "source": "tmp_download"}
    assert calls[-4:] == [(fallback, True, True), (fallback, False, True), (fallback, True, False), (fallback, False, False)]
