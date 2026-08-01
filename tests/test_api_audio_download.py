import asyncio
import socket
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from API import audio_download


def _record(address, port=443):
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    return (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port))


def test_resolution_rejects_credentials_and_any_non_public_address():
    with pytest.raises(ValueError, match="credentials"):
        audio_download.resolve_public_addresses(
            "https://user:secret@example.test/audio.wav",
            getaddrinfo=lambda *args, **kwargs: [_record("8.8.8.8")],
        )

    with pytest.raises(ValueError, match="only to public"):
        audio_download.resolve_public_addresses(
            "https://example.test/audio.wav",
            getaddrinfo=lambda *args, **kwargs: [
                _record("8.8.8.8"),
                _record("127.0.0.1"),
            ],
        )


def test_pinned_resolver_returns_only_validated_addresses():
    resolver = audio_download.PinnedResolver(
        "example.test",
        ("8.8.8.8", "2606:4700:4700::1111"),
    )
    results = asyncio.run(resolver.resolve("example.test", 443))

    assert [result["host"] for result in results] == [
        "8.8.8.8",
        "2606:4700:4700::1111",
    ]
    with pytest.raises(OSError, match="redirected host"):
        asyncio.run(resolver.resolve("other.test", 443))


def test_installed_aiohttp_accepts_pinned_resolver():
    aiohttp = pytest.importorskip("aiohttp")

    async def verify():
        resolver = audio_download.PinnedResolver("example.test", ("8.8.8.8",))
        connector = aiohttp.TCPConnector(resolver=resolver, use_dns_cache=False)
        try:
            assert (await resolver.resolve("example.test", 443))[0]["host"] == "8.8.8.8"
        finally:
            await connector.close()

    asyncio.run(verify())


class FakeContent:
    def __init__(self, chunks):
        self.chunks = chunks

    async def iter_chunked(self, chunk_size):
        for chunk in self.chunks:
            yield chunk


class FakeResponse:
    def __init__(self, chunks, headers=None, status=200):
        self.content = FakeContent(chunks)
        self.headers = headers or {}
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


def _fake_aiohttp(response, captured):
    class Connector:
        def __init__(self, **kwargs):
            captured["connector"] = kwargs

    class Timeout:
        def __init__(self, **kwargs):
            captured["timeout"] = kwargs

    class Session:
        def __init__(self, **kwargs):
            captured["session"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return None

        def get(self, url, **kwargs):
            captured["request"] = (url, kwargs)
            return response

    return SimpleNamespace(
        TCPConnector=Connector,
        ClientTimeout=Timeout,
        ClientSession=Session,
    )


def test_download_uses_pinned_resolver_and_disables_redirects_and_proxy(
    monkeypatch,
):
    captured = {}
    response = FakeResponse(
        [b"audio"],
        headers={"content-length": "5", "content-type": "audio/mpeg"},
    )
    monkeypatch.setattr(
        audio_download,
        "resolve_public_addresses",
        lambda url: ("example.test", 443, ("8.8.8.8",)),
    )
    monkeypatch.setitem(sys.modules, "aiohttp", _fake_aiohttp(response, captured))

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        result = asyncio.run(
            audio_download.download_remote_audio(
                "https://example.test/audio",
                directory,
            )
        )

        assert Path(result).read_bytes() == b"audio"
        assert Path(result).suffix == ".mp3"
    assert captured["session"]["trust_env"] is False
    assert isinstance(
        captured["connector"]["resolver"],
        audio_download.PinnedResolver,
    )
    assert captured["request"][1]["allow_redirects"] is False


def test_oversized_stream_leaves_no_temporary_file(monkeypatch):
    captured = {}
    response = FakeResponse([b"12", b"34"])
    monkeypatch.setattr(
        audio_download,
        "resolve_public_addresses",
        lambda url: ("example.test", 443, ("8.8.8.8",)),
    )
    monkeypatch.setitem(sys.modules, "aiohttp", _fake_aiohttp(response, captured))

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        with pytest.raises(ValueError, match="size limit"):
            asyncio.run(
                audio_download.download_remote_audio(
                    "https://example.test/audio.wav",
                    directory,
                    max_bytes=3,
                )
            )
        assert list(Path(directory).iterdir()) == []


def test_cleanup_removes_only_downloader_files():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        root = Path(directory)
        downloaded = root / "download_audio.wav"
        unrelated = root / "reference.wav"
        downloaded.write_bytes(b"audio")
        unrelated.write_bytes(b"keep")

        audio_download.cleanup_downloaded_audio(
            [downloaded, unrelated, root.parent / "download_outside.wav"],
            root,
        )

        assert not downloaded.exists()
        assert unrelated.exists()


def test_temporary_audio_registry_releases_owned_files():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        root = Path(directory)
        first = root / "download_first.wav"
        second = root / "download_second.wav"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        registry = audio_download.TemporaryAudioRegistry(root)

        registry.adopt("alice", [first])
        registry.adopt("bob", [second])
        registry.release("alice")

        assert not first.exists()
        assert second.exists()
        registry.clear()
        assert not second.exists()


def test_stream_cleanup_waits_for_inference_future():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        downloaded = Path(directory) / "download_stream.wav"
        downloaded.write_bytes(b"audio")

        async def verify():
            future = asyncio.get_running_loop().create_future()
            await audio_download.cleanup_downloaded_audio_after(
                future,
                [downloaded],
                directory,
            )
            assert downloaded.exists()
            future.set_result(None)
            await asyncio.sleep(0)
            assert not downloaded.exists()

        asyncio.run(verify())
