import threading
from types import SimpleNamespace

import pytest

from gsv_tts import TTS


class _BertRequested(Exception):
    pass


def _engine_stopping_at_bert_check():
    engine = object.__new__(TTS)
    engine._infer_lock = threading.RLock()
    engine._empty_cache = lambda: None

    def stop(*args, **kwargs):
        raise _BertRequested

    engine._ensure_bert_for_texts = stop
    return engine


def test_infer_checks_target_and_prompt_for_lazy_bert():
    engine = _engine_stopping_at_bert_check()

    with pytest.raises(_BertRequested):
        engine.infer("speaker.wav", "prompt.wav", "提示", "hello")


def test_infer_stream_checks_target_and_prompt_for_lazy_bert():
    engine = _engine_stopping_at_bert_check()

    stream = engine.infer_stream("speaker.wav", "prompt.wav", "提示", "hello")
    with pytest.raises(_BertRequested):
        next(stream)


def test_infer_batched_checks_target_and_prompt_for_lazy_bert():
    engine = _engine_stopping_at_bert_check()
    engine._check_pause = lambda text: True

    with pytest.raises(_BertRequested):
        engine._prepare_batched_inputs(
            texts=["hello"],
            spk_audio_paths="speaker.wav",
            prompt_audio_paths="prompt.wav",
            prompt_audio_texts="中文",
            return_subtitles=False,
            is_cut_text=True,
            cut_minlen=10,
            cut_mute=0.4,
            cut_mute_scale_map={},
            speed=1.0,
            bert_batch_size=20,
            gpt_model=None,
            sovits_model=None,
            text_languages="en",
            prompt_languages="auto",
        )


def test_bert_language_policy_respects_explicit_languages():
    engine = object.__new__(TTS)
    calls = []
    engine._contains_chinese = lambda text: text == "中文"
    engine._ensure_bert_loaded = lambda: calls.append(True)

    engine._ensure_bert_for_texts(["中文", "forced"], ["en", "zh"])

    assert calls == [True]


def test_cache_prompt_audio_checks_prompt_text_for_lazy_bert():
    engine = _engine_stopping_at_bert_check()
    engine.sovits_models = {"model": SimpleNamespace()}
    engine.cnhubert_model = SimpleNamespace()

    with pytest.raises(_BertRequested):
        engine.cache_prompt_audio("prompt.wav", "中文")
