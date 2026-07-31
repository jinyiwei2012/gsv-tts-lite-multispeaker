"""Unit tests for model file discovery & pairing (no model loading)."""

import tempfile
from pathlib import Path

from gsv_tts.model_discovery import auto_name, discover_models, normalize


def _touch(d: Path, *names: str):
    for n in names:
        (d / n).touch()


def test_discover_typical_pairs():
    with tempfile.TemporaryDirectory() as d:
        _touch(
            Path(d),
            "CyreneV3.7-e25.ckpt", "CyreneV3.7_e16_s1392.pth",
            "shouanren-e20.ckpt", "shouanren_e24_s1584.pth",
            "lty-tts_gpt_model.ckpt", "lty-tts_sovits_model.pth",
        )
        pairs = discover_models([Path(d)])
        assert len(pairs) == 3
        assert {p[2] for p in pairs} == {"CyreneV3.7", "shouanren", "lty-tts"}


def test_discover_ignores_default_models_and_orphans():
    with tempfile.TemporaryDirectory() as d:
        _touch(
            Path(d),
            "s1v3.ckpt", "s2Gv2ProPlus.pth",  # default models, no common prefix
            "unrelated.ckpt",                  # orphan
        )
        pairs = discover_models([Path(d)])
        assert pairs == []


def test_discover_skips_vcs_dirs():
    with tempfile.TemporaryDirectory() as d:
        git = Path(d) / ".git"
        git.mkdir()
        _touch(git, "a.ckpt", "a_sovits.pth")
        pairs = discover_models([Path(d)])
        assert pairs == []


def test_discover_multiple_dirs_and_min_prefix():
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        _touch(Path(d1), "alice-gpt.ckpt", "alice-sovits.pth")
        _touch(Path(d2), "bob_gpt.ckpt", "bob_sovits.pth")
        pairs = discover_models([Path(d1), Path(d2)])
        assert len(pairs) == 2
        # 提升阈值后前缀分不足，但特征词剥离相等仍可强配对
        assert len(discover_models([Path(d1), Path(d2)], min_prefix=99)) == 2


def test_discover_feature_word_pairing():
    # 短前缀 + gpt/sovits 后缀命名也能配对
    with tempfile.TemporaryDirectory() as d:
        _touch(Path(d), "bob_gpt.ckpt", "bob_sovits.pth")
        pairs = discover_models([Path(d)])
        assert len(pairs) == 1 and pairs[0][2] == "bob"


def test_auto_name_and_normalize():
    assert normalize("CyreneV3.7-e25.ckpt") == "cyrenev37e25ckpt"
    assert auto_name(Path("lty-tts_gpt_model.ckpt"), Path("lty-tts_sovits_model.pth")) == "lty-tts"
    assert auto_name(Path("a.ckpt"), Path("b.pth")) == "a"
