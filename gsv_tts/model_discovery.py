"""Model file discovery & pairing helpers (shared by benchmark & WebUI).

Pairs GPT (``*.ckpt``) with SoVITS (``*.pth``) model files by normalized
filename prefix, e.g. ``CyreneV3.7-e25.ckpt`` + ``CyreneV3.7_e16_s1392.pth``.
"""

import os
import re
from pathlib import Path

_SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    "site-packages", ".idea", ".vscode",
}


def normalize(name: str) -> str:
    """Lowercase alphanumeric-only form used for filename comparison."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def auto_name(gpt: Path, sovits: Path) -> str:
    """Derive a speaker name from the common prefix of two model filenames."""
    common = os.path.commonprefix([gpt.stem, sovits.stem]).rstrip("-_ .")
    return common or gpt.stem


def discover_models(dirs: list[Path], min_prefix: int = 4):
    """Pair *.ckpt (GPT) with *.pth (SoVITS) by normalized filename prefix.

    Greedy: for each .pth, take the not-yet-paired .ckpt with the longest
    common prefix (>= min_prefix). Returns list of (gpt_path, sovits_path,
    speaker_name).
    """
    gpt_files, sovits_files = [], []
    for d in dirs:
        for root, dirnames, filenames in os.walk(d):
            dirnames[:] = [x for x in dirnames if x not in _SKIP_DIRS]
            for fn in filenames:
                p = Path(root) / fn
                if fn.endswith(".ckpt"):
                    gpt_files.append(p)
                elif fn.endswith(".pth"):
                    sovits_files.append(p)

    pairs = []
    unmatched = sorted(gpt_files)
    for sovits in sorted(sovits_files):
        sn = normalize(sovits.stem)
        best, best_score = None, 0
        for gpt in unmatched:
            score = common_prefix_len(normalize(gpt.stem), sn)
            if score > best_score:
                best, best_score = gpt, score
        if best is not None and best_score >= min_prefix:
            unmatched.remove(best)
            pairs.append((best, sovits, auto_name(best, sovits)))
    return pairs
