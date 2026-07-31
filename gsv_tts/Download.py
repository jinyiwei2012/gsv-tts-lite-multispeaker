import os
import time
import logging
import requests
import zipfile
from tqdm import tqdm
from pathlib import Path


# Mirror selection — allow override via GSV_MIRROR env var
#   "modelscope" → ModelScope (best for China)
#   "huggingface" → Hugging Face (international)
#   "hf-mirror" → hf-mirror.com (HF proxy for China)
_MIRROR_OVERRIDE = os.environ.get("GSV_MIRROR", "")
modelscope_base_url = "https://modelscope.cn/models/chinokiki/GPTSoVITS-RT/resolve/master/%s"
huggingface_base_url = "https://huggingface.co/cnmds/GPTSoVITS-RT/resolve/main/%s?download=true"
hf_mirror_base_url = "https://hf-mirror.com/cnmds/GPTSoVITS-RT/resolve/main/%s?download=true"

base_url = None

_DEFAULT_TIMEOUT = 30  # seconds

# Default GPT/SoVITS model files (not in pretrained_models zip)
_DEFAULT_MODEL_FILES = [
    "s1v3.ckpt",
    "s2Gv2ProPlus.pth",
]


def download_file(url, filename, timeout=_DEFAULT_TIMEOUT):
    """Download a file with timeout, progress bar, and integrity check."""
    logging.info(f"Downloading model from {url}")

    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Download request failed for {url}: {e}")
        return False

    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024

    try:
        with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True) as progress_bar:
            with open(filename, 'wb') as file:
                for data in response.iter_content(block_size):
                    if data:
                        progress_bar.update(len(data))
                        file.write(data)
    except Exception as e:
        logging.error(f"Download interrupted: {e}")
        # Remove partial file
        try:
            Path(filename).unlink(missing_ok=True)
        except Exception:
            pass
        return False

    downloaded_size = Path(filename).stat().st_size
    if total_size_in_bytes != 0 and downloaded_size != total_size_in_bytes:
        logging.error(
            f"Incomplete download: {downloaded_size}/{total_size_in_bytes} bytes. "
            f"Removing partial file."
        )
        Path(filename).unlink(missing_ok=True)
        return False

    logging.info(f"Download complete: {filename}")
    return True


def unzip_file(zip_filepath, extract_to):
    """安全解压 ZIP 文件，防止路径遍历攻击"""
    logging.info(f"Extracting {zip_filepath}...")
    extract_to = Path(extract_to).resolve()
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        for member in zip_ref.infolist():
            member_path = (extract_to / member.filename).resolve()
            # 确保解压路径在目标目录内
            if not str(member_path).startswith(str(extract_to)):
                raise ValueError(
                    f"Security: attempted path traversal in zip: {member.filename}"
                )
            # 检查解压后文件总大小（防止 zip bomb）
            if member.file_size > 10 * 1024 * 1024 * 1024:  # 10GB
                raise ValueError(
                    f"Security: file too large in zip: {member.filename} ({member.file_size} bytes)"
                )
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


def download_model(filename, dir, download_url=None):
    if download_url is None:
        download_url = get_base_url()
        
    url = download_url % (filename)
    zip_filename = Path(dir) / filename

    if not download_file(url, zip_filename):
        raise RuntimeError(f"Download of {filename} failed after exhausting retries")

    unzip_file(zip_filename, os.path.dirname(zip_filename))
    zip_filename.unlink(missing_ok=True)


def _download_zip_with_fallback(dir, candidates):
    """Download and extract a zip from the first working candidate.

    Args:
        dir: Destination directory (created if missing).
        candidates: Iterable of (url_template, filename) tried in order.

    Returns:
        The (url, filename) that succeeded, or None if all failed.
    """
    os.makedirs(dir, exist_ok=True)
    for url, filename in candidates:
        zip_path = Path(dir) / filename
        if zip_path.exists():
            # 残留 zip：先尝试直接解压；损坏则删除并重新下载
            try:
                unzip_file(zip_path, dir)
                zip_path.unlink(missing_ok=True)
                return (url, filename)
            except Exception as e:
                logging.warning(f"Existing zip is corrupt ({zip_path}): {e}")
                zip_path.unlink(missing_ok=True)
        try:
            if download_file(url % filename, zip_path):
                unzip_file(zip_path, dir)
                zip_path.unlink(missing_ok=True)
                return (url, filename)
        except Exception as e:
            logging.warning(f"Download failed from {url} ({filename}): {e}")
    return None


def check_pretrained_models(models_dir):
    model_list = [
        Path(models_dir) / "chinese-hubert-base",
        Path(models_dir) / "g2p",
        Path(models_dir) / "sv",
    ]

    if all(os.path.exists(model_path) for model_path in model_list):
        return

    base = get_base_url()
    os.makedirs(models_dir, exist_ok=True)

    def zip_name_for(url):
        # ModelScope 的 5.zip 内置 g2p；HF 系的 6.zip 需要单独下载 g2p.zip
        return "pretrained_models5.zip" if url == modelscope_base_url else "pretrained_models6.zip"

    # 主 zip 带镜像 fallback 链：主源失败后依次尝试其他镜像（换对应文件名）
    candidates = [(base, zip_name_for(base))]
    for _, url in (("HuggingFace", huggingface_base_url), ("HF-Mirror", hf_mirror_base_url), ("ModelScope", modelscope_base_url)):
        if url != base:
            candidates.append((url, zip_name_for(url)))

    used = _download_zip_with_fallback(models_dir, candidates)
    if used is None:
        raise RuntimeError(
            "Failed to download pretrained models from all mirrors. "
            f"Please download manually and place the files under {models_dir}"
        )

    # pretrained_models5.zip (ModelScope) 已内置 g2p；6.zip 需要单独从 GitHub 下载
    used_modelscope_zip = used[0] == modelscope_base_url
    if not used_modelscope_zip and not os.path.exists(Path(models_dir) / "g2p"):
        g2p_url = "https://github.com/chinokikiss/GSV-TTS-Lite/releases/download/g2p/%s"
        if _download_zip_with_fallback(models_dir, [(g2p_url, "g2p.zip")]) is None:
            raise RuntimeError(
                "Failed to download g2p.zip. "
                f"Please download manually and place it under {models_dir}"
            )


def ensure_default_models(models_dir):
    """Download default GPT and SoVITS model files if not present."""
    base = get_base_url()

    os.makedirs(models_dir, exist_ok=True)

    for filename in _DEFAULT_MODEL_FILES:
        filepath = Path(models_dir) / filename
        if filepath.exists():
            logging.info(f"Default model already exists: {filename}")
            continue

        url = base % filename
        logging.info(f"Downloading default model: {filename}")
        try:
            if not download_file(url, filepath):
                raise RuntimeError(f"Download incomplete: {filename}")
        except Exception as e:
            logging.warning(f"Primary download failed for {filename}: {e}")

            # Fallback chain
            fallbacks = []
            if base != hf_mirror_base_url:
                fallbacks.append(("hf-mirror.com", hf_mirror_base_url))
            if base != huggingface_base_url:
                fallbacks.append(("Hugging Face", huggingface_base_url))
            if base != modelscope_base_url:
                fallbacks.append(("ModelScope", modelscope_base_url))

            for name, fb_url in fallbacks:
                try:
                    logging.info(f"Trying {name} fallback: {filename}")
                    if download_file(fb_url % filename, filepath):
                        logging.info(f"Downloaded via {name}")
                        break
                except Exception as e2:
                    logging.warning(f"{name} fallback failed: {e2}")
            else:
                raise RuntimeError(
                    f"Failed to download {filename} from all sources. "
                    f"Please download manually and place it in {models_dir}"
                )


cnroberta_int8_modelscope_base_url = "https://modelscope.cn/models/ltyytn/cnroberta_int8_dynamic/resolve/master/%s"
cnroberta_int8_huggingface_base_url = "https://huggingface.co/cnmds/GPTSoVITS-RT/resolve/main/int8/cnroberta/%s?download=true"

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
        
        if os.path.exists(filepath):
            logging.info(f"文件已存在，跳过: {filepath}")
            continue
        
        logging.info(f"正在下载: {filename}")
        if not download_file(url, filepath):
            raise RuntimeError(f"Failed to download {filename}")
    
    logging.info(f"CNRoberta INT8 ONNX 模型下载完成: {dir}")
