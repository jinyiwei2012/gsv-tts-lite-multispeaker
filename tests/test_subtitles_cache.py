"""Unit tests for subtitle export (SRT/ASS) and audio cache fingerprinting."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from gsv_tts import AudioClip
from gsv_tts.TTS import TTS


def _clip(subtitles):
    return AudioClip(None, np.zeros(32000, dtype=np.float32), 32000, 1.0, subtitles, "text")


def test_export_srt():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "out.srt"
        _clip([
            {"text": "你好", "start_s": 0.0, "end_s": 0.5},
            {"text": "世界", "start_s": 0.6, "end_s": 1.0},
        ]).export_subtitles(str(path))
        content = path.read_text(encoding="utf-8")
        assert "00:00:00,000 --> 00:00:00,500" in content
        assert "00:00:00,600 --> 00:00:01,000" in content
        assert content.count("-->") == 2


def test_export_ass():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "out.ass"
        _clip([
            {"text": "多行\n文本", "start_s": 2.0, "end_s": 2.4},
        ]).export_subtitles(str(path), fmt="ass")
        content = path.read_text(encoding="utf-8")
        assert "Dialogue: 0,0:00:02.00,0:00:02.40,Default,,0,0,0,,多行\\N文本" in content


def test_export_subtitles_requires_subtitles():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError, match="return_subtitles"):
            _clip(None).export_subtitles(str(Path(d) / "x.srt"))


def test_export_subtitles_unknown_format():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError, match="Unsupported"):
            _clip([{"text": "a", "start_s": 0.0, "end_s": 0.1}]).export_subtitles(
                str(Path(d) / "x.vtt"), fmt="vtt"
            )


def test_file_stat_fingerprint_changes():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.wav"
        p.write_bytes(b"x" * 100)
        s1 = TTS._file_stat(str(p))
        assert s1 is not None and s1[1] == 100
        p.write_bytes(b"y" * 200)
        s2 = TTS._file_stat(str(p))
        assert s1 != s2


def test_file_stat_none_for_missing_path():
    assert TTS._file_stat("C:/definitely/not/exist.wav") is None
