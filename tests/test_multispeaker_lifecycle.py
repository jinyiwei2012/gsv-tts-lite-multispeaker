import threading
from contextlib import contextmanager
from types import SimpleNamespace

import gsv_tts.MultiSpeaker as multispeaker_module
from gsv_tts import MultiSpeakerTTS


class TrackingRLock:
    def __init__(self):
        self._lock = threading.RLock()
        self.depth = 0
        self.entries = 0

    def __enter__(self):
        self._lock.acquire()
        self.depth += 1
        self.entries += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.depth -= 1
        self._lock.release()


class FakeTTS:
    def __init__(self):
        self._infer_lock = TrackingRLock()
        self.unloaded_gpt = []
        self.unloaded_sovits = []
        self.spk_audio_cache = {}
        self.prompt_audio_cache = {}

    def unload_gpt_model(self, key):
        self.unloaded_gpt.append(key)

    def unload_sovits_model(self, key):
        self.unloaded_sovits.append(key)

    def del_spk_audio(self, key):
        self.spk_audio_cache.pop(key, None)

    def del_prompt_audio(self, key):
        self.prompt_audio_cache.pop(key, None)


def _full_speaker(name):
    return SimpleNamespace(
        name=name,
        is_full_model=True,
        gpt_model_key="shared.ckpt",
        sovits_model_key="shared.pth",
        spk_audio_path="voice.wav",
        prompt_audio_path="prompt.wav",
    )


def test_add_speaker_uses_inference_lock():
    engine = object.__new__(MultiSpeakerTTS)
    engine._tts = FakeTTS()
    engine._speakers = {}

    def add(speaker):
        assert engine._tts._infer_lock.depth == 1
        engine._speakers[speaker.name] = speaker

    engine._add_speaker = add
    engine.add_speaker(SimpleNamespace(name="alice"))

    assert list(engine._speakers) == ["alice"]


def test_remove_speaker_releases_only_the_last_shared_reference():
    engine = object.__new__(MultiSpeakerTTS)
    engine._tts = FakeTTS()
    engine._speakers = {
        "alice": _full_speaker("alice"),
        "bob": _full_speaker("bob"),
    }
    engine._active_speaker = "alice"
    for name in engine._speakers:
        engine._tts.spk_audio_cache[engine._spk_cache_key(name)] = object()
        engine._tts.prompt_audio_cache[engine._prompt_cache_key(name)] = object()
    engine._tts.spk_audio_cache["voice.wav"] = object()
    engine._tts.prompt_audio_cache["prompt.wav"] = object()

    engine.remove_speaker("alice")

    assert engine._tts.unloaded_gpt == []
    assert engine._tts.unloaded_sovits == []
    assert "voice.wav" in engine._tts.spk_audio_cache
    assert "prompt.wav" in engine._tts.prompt_audio_cache
    assert engine._spk_cache_key("alice") not in engine._tts.spk_audio_cache
    assert engine._active_speaker is None

    engine.remove_speaker("bob")

    assert engine._tts.unloaded_gpt == ["shared.ckpt"]
    assert engine._tts.unloaded_sovits == ["shared.pth"]
    assert engine._tts.spk_audio_cache == {}
    assert engine._tts.prompt_audio_cache == {}
    assert engine._tts._infer_lock.entries == 2


def test_infer_auto_keeps_lock_through_routed_inference(monkeypatch):
    engine = object.__new__(MultiSpeakerTTS)
    lock = TrackingRLock()
    embedding = object()
    engine._tts = SimpleNamespace(
        _infer_lock=lock,
        sovits_models={},
        spk_audio_cache={},
        sv_model=SimpleNamespace(compute_embedding3=lambda audio: embedding),
        sv_path="unused",
        tts_config=SimpleNamespace(),
        _get_spec=lambda hps, path: (None, object()),
    )
    engine._shared_sovits = SimpleNamespace(hps=SimpleNamespace())
    engine._speakers = {
        "alice": SimpleNamespace(spk_audio_path="alice.wav"),
    }

    def cache_spk_audio(path, sovits_model):
        engine._tts.spk_audio_cache[path] = {"sv_emb": embedding}

    engine._tts.cache_spk_audio = cache_spk_audio

    @contextmanager
    def activate(*args, **kwargs):
        yield None

    engine._activate_shared_models = activate

    def infer(name, text, **kwargs):
        assert lock.depth == 1
        return "audio"

    engine.infer = infer
    monkeypatch.setattr(
        multispeaker_module.torch,
        "cosine_similarity",
        lambda *args, **kwargs: SimpleNamespace(item=lambda: 0.9),
    )

    assert engine.infer_auto("query.wav", "hello") == ("alice", "audio")
