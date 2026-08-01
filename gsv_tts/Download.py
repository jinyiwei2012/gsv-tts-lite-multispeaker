import os
import time
import logging
import requests
import stat
import tempfile
import zipfile
from hashlib import sha256
from tqdm import tqdm
from pathlib import Path


# Mirror selection — allow override via GSV_MIRROR env var
#   "modelscope" → ModelScope (best for China)
#   "huggingface" → Hugging Face (international)
#   "hf-mirror" → hf-mirror.com (HF proxy for China)
_MIRROR_OVERRIDE = os.environ.get("GSV_MIRROR", "")
_HF_RUNTIME_REVISION = "0978701405063b68206b7b5784fe628b84637a6d"
_UPSTREAM_MODEL_REVISION = "336b2ec4e8d4ac74740798dd40af44e74659ecaf"
modelscope_base_url = "https://modelscope.cn/models/chinokiki/GPTSoVITS-RT/resolve/master/%s"
huggingface_base_url = (
    "https://huggingface.co/cnmds/GPTSoVITS-RT/resolve/"
    f"{_HF_RUNTIME_REVISION}/%s?download=true"
)
hf_mirror_base_url = (
    "https://hf-mirror.com/cnmds/GPTSoVITS-RT/resolve/"
    f"{_HF_RUNTIME_REVISION}/%s?download=true"
)
upstream_model_base_url = (
    "https://huggingface.co/lj1995/GPT-SoVITS/resolve/"
    f"{_UPSTREAM_MODEL_REVISION}/%s?download=true"
)
upstream_model_mirror_url = (
    "https://hf-mirror.com/lj1995/GPT-SoVITS/resolve/"
    f"{_UPSTREAM_MODEL_REVISION}/%s?download=true"
)
g2p_release_base_url = (
    "https://github.com/chinokikiss/GSV-TTS-Lite/"
    "releases/download/g2p/%s"
)

base_url = None

_DEFAULT_TIMEOUT = 30  # seconds
_MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024 * 1024
_MAX_EXTRACTED_BYTES = 10 * 1024 * 1024 * 1024

# Default GPT/SoVITS model files (not in pretrained_models zip)
_DEFAULT_MODEL_FILES = [
    "s1v3.ckpt",
    "s2Gv2ProPlus.pth",
]

_DEFAULT_MODEL_PATHS = {
    "s1v3.ckpt": "s1v3.ckpt",
    "s2Gv2ProPlus.pth": "v2Pro/s2Gv2ProPlus.pth",
}

# SHA-256 values published by the backing repositories (HF LFS OIDs and
# ModelScope/GitHub release digests). Mirror-specific archives are different
# byte streams and therefore intentionally have separate entries.
_DOWNLOAD_SHA256 = {
    (modelscope_base_url, "pretrained_models5.zip"): (
        "534d4fc57fde79e83dcd7af311a47f58530861665bcbd75c6c4c8da0b677648c"
    ),
    (huggingface_base_url, "pretrained_models6.zip"): (
        "640ab803939912c3b96bee1aa7271100225dbea16341075a9f9c6079c0be097d"
    ),
    (hf_mirror_base_url, "pretrained_models6.zip"): (
        "640ab803939912c3b96bee1aa7271100225dbea16341075a9f9c6079c0be097d"
    ),
    (g2p_release_base_url, "g2p.zip"): (
        "8bb1d58798c49c7913f24ca53ebe1ed2f69d0fda5c7c6e158b7a36d4a160e148"
    ),
}

_DEFAULT_MODEL_SHA256 = {
    "s1v3.ckpt": (
        "87133414860ea14ff6620c483a3db5ed07b44be42e2c3fcdad65523a729a745a"
    ),
    "s2Gv2ProPlus.pth": (
        "d42a22bbbf65fb2bbdd45ad6a66841156977db45c7aabe0a6992ff378d9c7d3b"
    ),
}


def _file_matches_sha256(path, expected_sha256):
    digest = sha256()
    try:
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return False
    return digest.hexdigest().lower() == expected_sha256.lower()


def download_file(url, filename, timeout=_DEFAULT_TIMEOUT, expected_sha256=None):
    """Download, verify, and atomically replace a file."""
    logging.info(f"Downloading model from {url}")
    target = Path(filename)
    partial_path = None
    response = None
    try:
        if expected_sha256 is not None:
            expected_sha256 = expected_sha256.lower()
            if len(expected_sha256) != 64 or any(
                char not in "0123456789abcdef" for char in expected_sha256
            ):
                raise ValueError("expected_sha256 must be a 64-character hex digest")

        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        try:
            total_size_in_bytes = int(response.headers.get("content-length", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid Content-Length header") from exc
        if total_size_in_bytes < 0 or total_size_in_bytes > _MAX_DOWNLOAD_BYTES:
            raise ValueError(
                f"download size is outside the allowed range: "
                f"{total_size_in_bytes} bytes"
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256()
        downloaded_size = 0
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".part",
            delete=False,
        ) as file:
            partial_path = Path(file.name)
            with tqdm(
                total=total_size_in_bytes,
                unit="iB",
                unit_scale=True,
            ) as progress_bar:
                for data in response.iter_content(64 * 1024):
                    if not data:
                        continue
                    downloaded_size += len(data)
                    if downloaded_size > _MAX_DOWNLOAD_BYTES:
                        raise ValueError("download exceeds maximum size")
                    digest.update(data)
                    progress_bar.update(len(data))
                    file.write(data)

        if total_size_in_bytes and downloaded_size != total_size_in_bytes:
            raise ValueError(
                f"incomplete download: {downloaded_size}/"
                f"{total_size_in_bytes} bytes"
            )
        if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
            raise ValueError(f"checksum mismatch for {target}")

        os.replace(partial_path, target)
        partial_path = None
        logging.info(f"Download complete: {target}")
        return True
    except Exception as e:
        logging.error(f"Download failed for {url}: {e}")
        return False
    finally:
        if response is not None:
            response.close()
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)


def unzip_file(zip_filepath, extract_to):
    """安全解压 ZIP 文件，防止路径遍历攻击"""
    logging.info(f"Extracting {zip_filepath}...")
    extract_to = Path(extract_to).resolve()
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        extracted_size = 0
        for member in zip_ref.infolist():
            member_path = (extract_to / member.filename).resolve()
            # 确保解压路径在目标目录内
            try:
                member_path.relative_to(extract_to)
            except ValueError:
                raise ValueError(
                    f"Security: attempted path traversal in zip: {member.filename}"
                )
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(
                    f"Security: symbolic links are not allowed in zip: "
                    f"{member.filename}"
                )
            # 检查解压后文件总大小（防止 zip bomb）
            extracted_size += member.file_size
            if extracted_size > _MAX_EXTRACTED_BYTES:
                raise ValueError("Security: extracted zip exceeds maximum size")
        zip_ref.extractall(extract_to)
    logging.info(f"Extraction complete, files located at: {extract_to}")


def check_latency(url, timeout=3):
    """Check if a URL is reachable and measure latency in ms."""
    try:
        start_time = time.time()
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        
        if response.status_code == 405:
            response = requests.get(url, timeout=timeout, stream=True)
            response.close()
            
        end_time = time.time()
        
        if 200 <= response.status_code < 400:
            latency = (end_time - start_time) * 1000
            return True, latency
        else:
            return False, float('inf')
            
    except requests.RequestException:
        return False, float('inf')
    finally:
        # Ensure stream connections are closed
        try:
            if 'response' in locals():
                response.close()
        except Exception:
            pass


def get_base_url(force_refresh=False):
    """Select best mirror (cached).  Respects GSV_MIRROR env var."""
    global base_url
    if base_url is not None and not force_refresh:
        return base_url

    # Env var override
    if _MIRROR_OVERRIDE:
        mapping = {
            "modelscope": modelscope_base_url,
            "huggingface": huggingface_base_url,
            "hf-mirror": hf_mirror_base_url,
        }
        if _MIRROR_OVERRIDE in mapping:
            base_url = mapping[_MIRROR_OVERRIDE]
            logging.info(f"Using mirror (env GSV_MIRROR): {_MIRROR_OVERRIDE}")
            return base_url
        logging.warning(f"Unknown GSV_MIRROR value '{_MIRROR_OVERRIDE}', auto-detecting")

    ms_url = "https://www.modelscope.cn"
    hf_url = "https://huggingface.co"
    hfm_url = "https://hf-mirror.com"

    ms_ok, ms_latency = check_latency(ms_url, timeout=5)
    hf_ok, hf_latency = check_latency(hf_url, timeout=5)
    hfm_ok, hfm_latency = check_latency(hfm_url, timeout=5)

    # ModelScope is best for China — prefer it if available
    if ms_ok and (not hf_ok or ms_latency < hf_latency):
        logging.info("Selected ModelScope.")
        base_url = modelscope_base_url
        return base_url

    # hf-mirror is fast in China, slower than ModelScope but faster than raw HF
    if hfm_ok and (not hf_ok or hfm_latency < hf_latency):
        logging.info("Selected HF-Mirror (hf-mirror.com).")
        base_url = hf_mirror_base_url
        return base_url

    if hf_ok:
        logging.info("Selected Hugging Face.")
        base_url = huggingface_base_url
        return base_url

    logging.error("All sources unreachable. Defaulting to HF-Mirror.")
    base_url = hf_mirror_base_url
    return base_url


def download_model(filename, dir, download_url=None, expected_sha256=None):
    if download_url is None:
        download_url = get_base_url()
        
    url = download_url % (filename)
    zip_filename = Path(dir) / filename
    expected_sha256 = expected_sha256 or _DOWNLOAD_SHA256.get(
        (download_url, filename)
    )
    if expected_sha256 is None:
        raise ValueError(f"No trusted SHA-256 configured for {filename}")

    if not download_file(
        url,
        zip_filename,
        expected_sha256=expected_sha256,
    ):
        raise RuntimeError(f"Download of {filename} failed after exhausting retries")

    unzip_file(zip_filename, os.path.dirname(zip_filename))
    zip_filename.unlink(missing_ok=True)


def check_pretrained_models(models_dir):
    model_list = [
        Path(models_dir) / "chinese-hubert-base",
        Path(models_dir) / "g2p",
        Path(models_dir) / "sv",
    ]

    is_download = False
    for model_path in model_list:
        if not os.path.exists(model_path):
            is_download = True
            break
    
    if is_download:
        base = get_base_url()

        os.makedirs(models_dir, exist_ok=True)

        if base == modelscope_base_url:
            download_model(
                download_url=base,
                filename="pretrained_models5.zip",
                dir=models_dir,
            )

        elif base == huggingface_base_url:
            download_model(
                download_url=base,
                filename="pretrained_models6.zip",
                dir=models_dir,
            )

            download_model(
                download_url=g2p_release_base_url,
                filename="g2p.zip",
                dir=models_dir,
            )

        else:
            # hf-mirror or any other: download same zip as HF
            download_model(
                download_url=base,
                filename="pretrained_models6.zip",
                dir=models_dir,
            )

            download_model(
                download_url=g2p_release_base_url,
                filename="g2p.zip",
                dir=models_dir,
            )


def ensure_default_models(models_dir):
    """Download default GPT and SoVITS model files if not present."""
    base = get_base_url()

    os.makedirs(models_dir, exist_ok=True)

    for filename in _DEFAULT_MODEL_FILES:
        filepath = Path(models_dir) / filename
        expected_sha256 = _DEFAULT_MODEL_SHA256[filename]
        if filepath.exists() and _file_matches_sha256(filepath, expected_sha256):
            logging.info(f"Default model already exists: {filename}")
            continue
        if filepath.exists():
            logging.warning(f"Default model checksum mismatch: {filepath}")

        remote_path = _DEFAULT_MODEL_PATHS[filename]
        sources = [upstream_model_base_url, upstream_model_mirror_url]
        if base in {modelscope_base_url, hf_mirror_base_url}:
            sources.reverse()
        logging.info(f"Downloading default model: {filename}")
        for source in sources:
            if download_file(
                source % remote_path,
                filepath,
                expected_sha256=expected_sha256,
            ):
                break
        else:
            raise RuntimeError(
                f"Failed to download {filename} from all sources. "
                f"Please download manually and place it in {models_dir}"
            )


cnroberta_int8_modelscope_base_url = "https://modelscope.cn/models/ltyytn/cnroberta_int8_dynamic/resolve/master/%s"
cnroberta_int8_huggingface_base_url = (
    "https://huggingface.co/cnmds/GPTSoVITS-RT/resolve/"
    f"{_HF_RUNTIME_REVISION}/int8/cnroberta/%s?download=true"
)

_CNROBERTA_INT8_SHA256 = {
    "config.json": (
        "3d57de2fd7e80d0e5c8ff194f0bbb6baa10df7e43fc262a0cc71298a78b0a3e5"
    ),
    "tokenizer.json": (
        "173796956820ea27bd14f76bf28162607ff4254807e2948253eb5b46f5bb643b"
    ),
    "cnroberta_int8_dynamic.onnx": (
        "24c36d383779213cad628a3f930941c451ceb7c85763cb6579fc12e6fa3b9284"
    ),
}


def download_cnroberta_int8(dir, download_url=None):
    """下载 CNRoberta INT8 Dynamic ONNX 模型"""
    if download_url is None:
        base = get_base_url()
        
        if base == modelscope_base_url:
            download_url = cnroberta_int8_modelscope_base_url
        else:
            download_url = cnroberta_int8_huggingface_base_url
    
    os.makedirs(dir, exist_ok=True)
    
    files_to_download = [
        "config.json",
        "tokenizer.json",
        "cnroberta_int8_dynamic.onnx",
    ]
    
    for filename in files_to_download:
        url = download_url % filename
        filepath = Path(dir) / filename

        expected_sha256 = _CNROBERTA_INT8_SHA256[filename]
        if filepath.exists() and _file_matches_sha256(filepath, expected_sha256):
            logging.info(f"文件已存在，跳过: {filepath}")
            continue
        if filepath.exists():
            logging.warning(f"文件校验失败，重新下载: {filepath}")
        
        logging.info(f"正在下载: {filename}")
        if not download_file(
            url,
            filepath,
            expected_sha256=expected_sha256,
        ):
            raise RuntimeError(f"Failed to download {filename}")
    
    logging.info(f"CNRoberta INT8 ONNX 模型下载完成: {dir}")
