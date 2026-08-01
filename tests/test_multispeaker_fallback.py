import threading
from types import SimpleNamespace

from gsv_tts import MultiSpeakerTTS


def test_full_model_batch_uses_sequential_full_model_inference():
    engine = object.__new__(MultiSpeakerTTS)
    engine._tts = SimpleNamespace(_infer_lock=threading.RLock())
    engine._speakers = {
        "legacy": SimpleNamespace(is_full_model=True),
    }
    calls = []

    def infer(**kwargs):
        calls.append(kwargs)
        return f"audio:{kwargs['text']}"

    engine.infer = infer
    engine._activate_shared_models = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("full-model speaker used shared backbone")
    )

    results = engine.infer_batched(
        [("legacy", "first"), ("legacy", "second")],
        text_languages=["en", "ja"],
    )

    assert results == ["audio:first", "audio:second"]
    assert [call["text_language"] for call in calls] == ["en", "ja"]
