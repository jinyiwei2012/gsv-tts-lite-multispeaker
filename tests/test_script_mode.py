"""Unit tests for dialogue-script parsing and speaker-text splitting."""

import pytest

from gsv_tts.MultiSpeaker import parse_script, split_speaker_text


def test_parse_basic_lines():
    r = parse_script("alice: こんにちは！\nbob: よろしくお願いします。")
    assert r == [("alice", "こんにちは！"), ("bob", "よろしくお願いします。")]


def test_parse_fullwidth_colon():
    assert parse_script("alice：你好") == [("alice", "你好")]


def test_parse_inline_tags():
    assert parse_script("<speaker:bob>こんにちは</speaker>") == [("bob", "こんにちは")]


def test_parse_skips_blank_and_empty_lines():
    r = parse_script("alice: a\n\n\nbob: b\nalice:  \n")
    assert r == [("alice", "a"), ("bob", "b")]


def test_parse_raises_on_unknown_line():
    with pytest.raises(ValueError, match="Cannot parse"):
        parse_script("this line has no speaker")


def test_split_speaker_text_plain():
    assert split_speaker_text("你好世界", "alice") == [("alice", "你好世界")]


def test_split_speaker_text_tags():
    assert split_speaker_text("<speaker:bob>こんにちは</speaker>", "alice") == [("bob", "こんにちは")]


def test_split_speaker_text_mixed():
    r = split_speaker_text("开头 <speaker:bob>中间</speaker> 结尾", "alice")
    assert r == [("alice", "开头"), ("bob", "中间"), ("alice", "结尾")]


def test_split_speaker_text_multiple_tags():
    r = split_speaker_text("<speaker:a>甲</speaker><speaker:b>乙</speaker>", "x")
    assert r == [("a", "甲"), ("b", "乙")]


def test_split_speaker_text_drops_empty_segments():
    assert split_speaker_text("<speaker:bob>   </speaker>", "alice") == []
    assert split_speaker_text("  a  b  ", "alice") == [("alice", "a  b")]
