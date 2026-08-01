from gsv_tts import TTS


def test_audio_cache_deletion_clears_file_fingerprints():
    engine = object.__new__(TTS)
    engine.spk_audio_cache = {"speaker.wav": {"ge": {}}}
    engine.prompt_audio_cache = {"prompt.wav": {"prompt": object()}}
    engine._spk_audio_stat = {
        "speaker.wav": (1, 2),
        "stale-speaker.wav": (3, 4),
    }
    engine._prompt_audio_stat = {
        "prompt.wav": (5, 6),
        "stale-prompt.wav": (7, 8),
    }

    engine.del_spk_audio("speaker.wav", "stale-speaker.wav")
    engine.del_prompt_audio("prompt.wav", "stale-prompt.wav")

    assert engine.spk_audio_cache == {}
    assert engine.prompt_audio_cache == {}
    assert engine._spk_audio_stat == {}
    assert engine._prompt_audio_stat == {}
