import hashlib
import importlib
import stat
import tempfile
import zipfile
from pathlib import Path

import pytest

from gsv_tts.Download import download_file, unzip_file


class Response:
    def __init__(self, body):
        self.body = body
        self.headers = {"content-length": str(len(body))}
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_content(self, block_size):
        yield self.body

    def close(self):
        self.closed = True


def test_download_file_rejects_checksum_mismatch(monkeypatch):
    module = importlib.import_module("gsv_tts.Download")
    response = Response(b"bad")
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: response)

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        target = Path(directory) / "model.bin"
        assert not download_file(
            "https://example.test/model.bin",
            target,
            expected_sha256="00" * 32,
        )
        assert not target.exists()
        assert list(target.parent.glob("*.part")) == []
        assert response.closed


def test_download_file_atomically_replaces_existing_target(monkeypatch):
    module = importlib.import_module("gsv_tts.Download")
    body = b"verified-model"
    response = Response(body)
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: response)

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        target = Path(directory) / "model.bin"
        target.write_bytes(b"old-model")

        assert download_file(
            "https://example.test/model.bin",
            target,
            expected_sha256=hashlib.sha256(body).hexdigest(),
        )
        assert target.read_bytes() == body
        assert list(target.parent.glob("*.part")) == []


def test_checksum_failure_preserves_existing_target(monkeypatch):
    module = importlib.import_module("gsv_tts.Download")
    monkeypatch.setattr(
        module.requests,
        "get",
        lambda *args, **kwargs: Response(b"untrusted"),
    )

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        target = Path(directory) / "model.bin"
        target.write_bytes(b"known-good")

        assert not download_file(
            "https://example.test/model.bin",
            target,
            expected_sha256="00" * 32,
        )
        assert target.read_bytes() == b"known-good"


def test_invalid_checksum_is_rejected_before_request(monkeypatch):
    module = importlib.import_module("gsv_tts.Download")

    def unexpected_request(*args, **kwargs):
        raise AssertionError("invalid manifest digest triggered a request")

    monkeypatch.setattr(module.requests, "get", unexpected_request)

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        assert not download_file(
            "https://example.test/model.bin",
            Path(directory) / "model.bin",
            expected_sha256="invalid",
        )


def test_download_size_limit_applies_without_content_length(monkeypatch):
    module = importlib.import_module("gsv_tts.Download")
    response = Response(b"oversized")
    response.headers = {}
    monkeypatch.setattr(module, "_MAX_DOWNLOAD_BYTES", 3)
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: response)

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        target = Path(directory) / "model.bin"
        assert not download_file("https://example.test/model.bin", target)
        assert not target.exists()
        assert list(target.parent.glob("*.part")) == []


def test_zip_file_does_not_reject_valid_sibling_prefix():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        root = Path(directory)
        archive = root / "safe.zip"
        destination = root / "models"
        destination.mkdir()
        with zipfile.ZipFile(archive, "w") as zip_ref:
            zip_ref.writestr("models/file.txt", "safe")

        unzip_file(archive, destination)
        assert (destination / "models" / "file.txt").read_text() == "safe"


def test_zip_file_rejects_path_traversal():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        root = Path(directory)
        archive = root / "malicious.zip"
        destination = root / "models"
        with zipfile.ZipFile(archive, "w") as zip_ref:
            zip_ref.writestr("../models-evil/outside.txt", "untrusted")

        with pytest.raises(ValueError, match="path traversal"):
            unzip_file(archive, destination)
        assert not (root / "models-evil" / "outside.txt").exists()


def test_zip_file_rejects_symbolic_links():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        root = Path(directory)
        archive = root / "symlink.zip"
        destination = root / "models"
        link = zipfile.ZipInfo("link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(archive, "w") as zip_ref:
            zip_ref.writestr(link, "../outside.txt")

        with pytest.raises(ValueError, match="symbolic links"):
            unzip_file(archive, destination)


def test_zip_file_limits_total_extracted_size(monkeypatch):
    module = importlib.import_module("gsv_tts.Download")
    monkeypatch.setattr(module, "_MAX_EXTRACTED_BYTES", 3)
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        root = Path(directory)
        archive = root / "large.zip"
        with zipfile.ZipFile(archive, "w") as zip_ref:
            zip_ref.writestr("first.bin", b"12")
            zip_ref.writestr("second.bin", b"34")

        with pytest.raises(ValueError, match="maximum size"):
            unzip_file(archive, root / "models")


def test_automatic_model_downloads_use_pinned_sources_and_hashes(monkeypatch):
    module = importlib.import_module("gsv_tts.Download")
    calls = []

    def download(url, target, **kwargs):
        calls.append((url, Path(target).name, kwargs.get("expected_sha256")))
        return True

    monkeypatch.setattr(module, "download_file", download)
    monkeypatch.setattr(module, "get_base_url", lambda: module.modelscope_base_url)
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        module.ensure_default_models(directory)

    assert [name for _, name, _ in calls] == module._DEFAULT_MODEL_FILES
    assert all(module._UPSTREAM_MODEL_REVISION in url for url, _, _ in calls)
    assert [digest for _, _, digest in calls] == [
        module._DEFAULT_MODEL_SHA256[name]
        for name in module._DEFAULT_MODEL_FILES
    ]


def test_download_model_requires_a_trusted_hash():
    module = importlib.import_module("gsv_tts.Download")
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        with pytest.raises(ValueError, match="No trusted SHA-256"):
            module.download_model(
                "custom.zip",
                directory,
                download_url="https://example.test/%s",
            )


def test_cnroberta_downloads_verify_existing_files_and_manifest(monkeypatch):
    module = importlib.import_module("gsv_tts.Download")
    calls = []

    def download(url, target, **kwargs):
        calls.append((Path(target).name, kwargs.get("expected_sha256")))
        return True

    monkeypatch.setattr(module, "download_file", download)
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        corrupt = Path(directory) / "config.json"
        corrupt.write_bytes(b"corrupt")
        module.download_cnroberta_int8(
            directory,
            download_url=module.cnroberta_int8_huggingface_base_url,
        )

    assert dict(calls) == module._CNROBERTA_INT8_SHA256
