# AGENTS.md

GPT-SoVITS TTS inference engine. Standalone fork of [GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite) (`multi-speaker-inference` branch) focused on **MultiSpeakerTTS shared-backbone inference**. The MultiSpeaker features are NOT in the PyPI `gsv-tts-lite` package — this repo is the only source. Keep upstream bug fixes in sync via cherry-pick/merge (per README).

## Commands

- No CI, no linter/formatter/typecheck config, no pytest config. Tests are plain scripts:
  - `python tests/test_sovits_sharing.py` — self-consistency test (shared backbone vs full model, MCD metric). Needs `librosa` for MCD (optional, warns otherwise).
  - `python benchmarks/bench_multi_speaker.py` — memory/latency benchmark. ⚠️ Contains **hardcoded absolute paths** to external fine-tuned models (`D:\Agent-LuoTianyi\...`); will not run as-is.
- WebUI: `pip install -r WebUI/requirements.txt` then `python WebUI/web.py`
- API servers: `pip install -r API/requirements.txt` then run `API/personal_api.py` / `API/realtime_api.py`
- Install package locally: `pip install -e .` (pure setuptools, no build step)

## Architecture

- `gsv_tts/` — the package. Entry points: `TTS` (single-speaker engine) and `MultiSpeakerTTS` (shared backbone), exported from `gsv_tts/__init__.py` along with `SpeakerConfig`, `SpeakerWeights`, `AudioClip`, `cut_text`, `ConfigMismatchError`.
- `gsv_tts/MultiSpeaker.py` — loads one base GPT+SoVITS backbone and injects per-speaker weights (~25 GPT keys + ~37 SoVITS keys). Models whose architecture mismatches the base (e.g. v2 with `upsample_initial_channel=512`) **auto-degrade to full model loading** — do not assume all speakers share the backbone.
- `gsv_tts/Loader.py` — weight loading; `gsv_tts/Download.py` — auto-download with mirror selection; `gsv_tts/TTS.py` — inference engine (infer / infer_stream / infer_batched / infer_vc).
- `WebUI/` and `API/` are standalone apps with their own `requirements.txt`, importing `gsv_tts` from repo root.

## Critical quirks

- **Model paths**: default `~/.cache/gsv` (override via `TTS(models_dir=...)`). Default files: `s1v3.ckpt` (GPT), `s2Gv2ProPlus.pth` (SoVITS). Constructing `TTS` triggers `check_pretrained_models()` + `ensure_default_models()` — **first run downloads several GB** (cnhubert/g2p/sv + GPT/SoVITS) and requires network.
- **Download mirrors**: auto-selected by latency test (ModelScope → hf-mirror → HuggingFace); force with env var `GSV_MIRROR=modelscope|huggingface|hf-mirror`. Changing mirror logic affects both `Download.py` and tests.
- **Do not remove the `sys.modules['utils'] = utils` monkey-patch at the top of `Loader.py`** — legacy GPT-SoVITS checkpoints pickle-require it.
- **`torch.load(..., weights_only=False)`** is used deliberately for legacy checkpoint compat. Checkpoints are untrusted input — prefer safetensors directory format (`hps.json` + `model.safetensors`), supported by both `TTS` and `MultiSpeakerTTS`. `tts.to_safetensors()` converts.
- **SoVITS version sniffing** in `Loader.py`: file header bytes (`01`=v2, `05`=v2Pro, `06`=v2ProPlus), else MD5 of known pretrained files, else warns and defaults to v2.
- Device: MPS/CPU forces `float32` and clears `sovits_cache`. CPU BERT uses INT8 ONNX (`cnroberta_int8_dynamic.onnx`), GPU uses PyTorch `chinese-roberta-wwm-ext-large`.
- Inference is serialized by `_infer_lock` (RLock) with `_empty_cache()` after each call — model loading is lazy (on-demand at first inference), not at init.

## Conventions

- Python 3.10+, type hints throughout (`Literal`, `str | None`). No formatter enforced — match existing style.
- Logging via `logging` module; `tqdm` for download progress. Chinese comments exist in `Download.py` and READMEs — keep bilingual comments as-is.
- `gpt_cache` / `sovits_cache` params control CUDA-graph static cache sizes; wrong values cause CUDA graph errors — don't change defaults casually.
- Speaker/reference audio caches (`spk_audio_cache`, `prompt_audio_cache`) are keyed by path — reusing the same path with different content yields stale cached features.
