import importlib
from types import SimpleNamespace

import pytest

from gsv_tts import MultiSpeakerTTS


def test_multispeaker_can_be_initialized_before_adding_speakers(monkeypatch):
    module = importlib.import_module("gsv_tts.MultiSpeaker")

    class FakeTTS:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
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
    assert engine._tts.init_kwargs["ensure_default_model_files"] is True


def test_multispeaker_skips_default_checkpoints_with_explicit_backbones(monkeypatch):
    module = importlib.import_module("gsv_tts.MultiSpeaker")

    class FakeTTS:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.models_dir = "models"
            self.tts_config = SimpleNamespace()
            self.audio_queue = SimpleNamespace()
            self.samplerate = 32000

    monkeypatch.setattr(module, "TTS", FakeTTS)
    monkeypatch.setattr(module, "load_shared_gpt", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "load_shared_sovits", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "_load_gpt_state_dict", lambda path: ({}, {}))
    monkeypatch.setattr(module, "_load_sovits_state_dict", lambda path: ({}, {}))

    engine = MultiSpeakerTTS(
        speakers=[],
        base_gpt_path="base-gpt",
        base_sovits_path="base-sovits",
    )

    assert engine._tts.init_kwargs["ensure_default_model_files"] is False


@pytest.mark.parametrize(
    ("ensure_default_model_files", "expected_calls"),
    ((True, 1), (False, 0)),
)
def test_tts_default_checkpoint_download_is_configurable(
    monkeypatch,
    tmp_path,
    ensure_default_model_files,
    expected_calls,
):
    module = importlib.import_module("gsv_tts.TTS")
    calls = []

    monkeypatch.setattr(module, "check_pretrained_models", lambda _path: None)
    monkeypatch.setattr(
        module,
        "ensure_default_models",
        lambda path: calls.append(path),
    )
    monkeypatch.setattr(module, "AudioQueue", lambda _samplerate: SimpleNamespace())

    module.TTS(
        models_dir=tmp_path,
        device="cpu",
        ensure_default_model_files=ensure_default_model_files,
    )

    assert len(calls) == expected_calls
