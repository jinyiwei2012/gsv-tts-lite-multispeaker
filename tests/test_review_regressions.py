import importlib
from types import SimpleNamespace

from gsv_tts import MultiSpeakerTTS


def test_multispeaker_can_be_initialized_before_adding_speakers(monkeypatch):
    module = importlib.import_module("gsv_tts.MultiSpeaker")

    class FakeTTS:
        def __init__(self, **kwargs):
            self.models_dir = "models"
            self.tts_config = SimpleNamespace()
            self.audio_queue = SimpleNamespace()
            self.samplerate = 32000

    monkeypatch.setattr(module, "TTS", FakeTTS)
    monkeypatch.setattr(module, "load_shared_gpt", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "load_shared_sovits", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "_load_gpt_state_dict", lambda path: ({}, {}))
    monkeypatch.setattr(module, "_load_sovits_state_dict", lambda path: ({}, {}))

    engine = MultiSpeakerTTS(speakers=[])

    assert engine.speaker_names == []
