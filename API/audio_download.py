"""Safe remote-audio downloads shared by the API servers."""

from __future__ import annotations

import ipaddress
import socket
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse


DEFAULT_MAX_AUDIO_BYTES = 50 * 1024 * 1024
_CHUNK_SIZE = 64 * 1024
_EXTENSIONS_BY_MIME = {
    "audio/flac": ".flac",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-wav": ".wav",
}
_ALLOWED_EXTENSIONS = frozenset(_EXTENSIONS_BY_MIME.values())


def _normalize_hostname(hostname: str) -> str:
    return hostname.rstrip(".").encode("idna").decode("ascii").lower()


def resolve_public_addresses(
    url: str,
    getaddrinfo=socket.getaddrinfo,
) -> tuple[str, int, tuple[str, ...]]:
    """Resolve an HTTP(S) URL and return only globally routable addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("audio URL must use HTTP or HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("audio URL credentials are not allowed")

    hostname = _normalize_hostname(parsed.hostname)
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise ValueError("audio URL contains an invalid port") from exc

    try:
        records = getaddrinfo(
            hostname,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise ValueError("audio URL host could not be resolved") from exc

    addresses = tuple(dict.fromkeys(record[4][0] for record in records))
    if not addresses:
        raise ValueError("audio URL host did not resolve to an address")
    if any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError("audio URL must resolve only to public addresses")
    return hostname, port, addresses


class PinnedResolver:
    """aiohttp-compatible resolver that never performs another DNS lookup."""

    def __init__(self, hostname: str, addresses: tuple[str, ...]):
        self.hostname = _normalize_hostname(hostname)
        self.addresses = addresses

    async def resolve(self, host, port=0, family=socket.AF_UNSPEC):
        if _normalize_hostname(host) != self.hostname:
            raise OSError("redirected host is not allowed")
        results = []
        for address in self.addresses:
            ip = ipaddress.ip_address(address)
            address_family = socket.AF_INET6 if ip.version == 6 else socket.AF_INET
            if family not in {socket.AF_UNSPEC, address_family}:
                continue
            results.append(
                {
                    "hostname": host,
                    "host": address,
                    "port": port,
                    "family": address_family,
                    "proto": socket.IPPROTO_TCP,
                    "flags": socket.AI_NUMERICHOST,
                }
            )
        if not results:
            raise OSError("no pinned address matches the requested family")
        return results

    async def close(self):
        return None


def _audio_extension(url: str, content_type: str) -> str:
    mime = content_type.partition(";")[0].strip().lower()
    if mime in _EXTENSIONS_BY_MIME:
        return _EXTENSIONS_BY_MIME[mime]
    suffix = Path(unquote(urlparse(url).path)).suffix.lower()
    return suffix if suffix in _ALLOWED_EXTENSIONS else ".wav"


async def download_remote_audio(
    url: str,
    destination_dir,
    max_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
) -> str:
    """Download a bounded audio file while pinning the validated DNS result."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    hostname, _, addresses = resolve_public_addresses(url)

    try:
        import aiohttp
    except ImportError as exc:
        raise RuntimeError(
            "Remote audio URLs require API dependencies: "
            "pip install -r API/requirements.txt"
        ) from exc

    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    resolver = PinnedResolver(hostname, addresses)
    connector = aiohttp.TCPConnector(resolver=resolver, use_dns_cache=False)
    timeout = aiohttp.ClientTimeout(total=60, connect=10, sock_read=30)
    temp_path = None

    try:
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            trust_env=False,
        ) as session:
            async with session.get(url, allow_redirects=False) as response:
                if 300 <= response.status < 400:
                    raise ValueError("audio URL redirects are not allowed")
                response.raise_for_status()
                try:
                    content_length = int(response.headers.get("content-length", 0))
                except (TypeError, ValueError) as exc:
                    raise ValueError("invalid Content-Length header") from exc
                if content_length < 0 or content_length > max_bytes:
                    raise ValueError("audio download exceeds the size limit")

                extension = _audio_extension(
                    url,
                    response.headers.get("content-type", ""),
                )
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=destination,
                    prefix="download_",
                    suffix=extension,
                    delete=False,
                ) as file:
                    temp_path = Path(file.name)
                    downloaded = 0
                    async for chunk in response.content.iter_chunked(_CHUNK_SIZE):
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise ValueError("audio download exceeds the size limit")
                        file.write(chunk)
                if content_length and downloaded != content_length:
                    raise ValueError("audio download ended before Content-Length")

        return str(temp_path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def cleanup_downloaded_audio(paths, destination_dir) -> None:
    """Delete only files created by this downloader in its dedicated directory."""
    destination = Path(destination_dir).resolve()
    for path in paths:
        if not path:
            continue
        candidate = Path(path).resolve()
        if candidate.parent != destination or not candidate.name.startswith("download_"):
            continue
        candidate.unlink(missing_ok=True)


async def cleanup_downloaded_audio_after(future, paths, destination_dir) -> None:
    """Clean now, or attach cleanup to an inference future still in flight."""
    if future is None or future.done():
        cleanup_downloaded_audio(paths, destination_dir)
        return

    future.add_done_callback(
        lambda _: cleanup_downloaded_audio(paths, destination_dir)
    )


class TemporaryAudioRegistry:
    """Track downloaded files whose lifetime belongs to a named resource."""

    def __init__(self, destination_dir):
        self.destination_dir = Path(destination_dir)
        self._paths: dict[str, list[str]] = {}

    def adopt(self, owner: str, paths) -> None:
        self.release(owner)
        self._paths[owner] = list(paths)

    def release(self, owner: str) -> None:
        cleanup_downloaded_audio(
            self._paths.pop(owner, []),
            self.destination_dir,
        )

    def clear(self) -> None:
        for owner in list(self._paths):
            self.release(owner)
