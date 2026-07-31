<div align="center">

> [!IMPORTANT]
> ### 🔀 Standalone MultiSpeaker Development Fork
> This repository originates from a fork of [GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite): the upstream repo has no `multi-speaker-inference` branch — it was created within the fork and later spun off into this standalone repository, focused on independently developing and optimizing **multi-speaker (MultiSpeakerTTS) shared-backbone inference**.
>
> - **Upstream repo**: [chinokikiss/GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite) (published on PyPI as `gsv-tts-lite`)
> - The multi-speaker features are **NOT yet published to PyPI** — this repo is the only source; upstream bug fixes can be synced via cherry-pick / merge

</div>

<div align="center">
  <h1>GSV-TTS-Lite · MultiSpeaker</h1>

  <p>
    Multi-speaker shared-backbone inference engine for GPT-SoVITS (MultiSpeakerTTS)
  </p>

  <p>
    <a href="README_EN.md">
      <img src="https://img.shields.io/badge/English-66ccff?style=flat-square&logo=github&logoColor=white" alt="English">
    </a>
    &nbsp;
    <a href="README.md">
      <img src="https://img.shields.io/badge/简体中文-ff99cc?style=flat-square&logo=github&logoColor=white" alt="Chinese">
    </a>
    &nbsp;
    <a href="README_JA.md">
      <img src="https://img.shields.io/badge/日本語-ffcc66?style=flat-square&logo=github&logoColor=white" alt="Japanese">
    </a>
  </p>
</div>

## About

This repository is a **standalone multi-speaker development fork of GSV-TTS-Lite**, centered on **MultiSpeakerTTS multi-speaker shared-backbone inference**:

The traditional approach loads a full model per speaker, so VRAM/memory grows linearly with the speaker count. This repo loads only **one shared GPT+SoVITS backbone**, and each speaker injects just ~5-15% lightweight per-speaker weights (~25 GPT keys + 37 SoVITS keys), **dynamically switched by speaker name at zero cost** — in multi-speaker scenarios this saves **40%~75%** VRAM/memory, and **the more speakers, the more you save**.

Full single-speaker inference (`TTS`) is also available: token-level streaming, batched inference, character-level timestamps, zero-shot voice conversion, and speaker verification, plus **WebUI** and **API** servers.

Supported languages: **Chinese, Japanese, English**; supported models: **V2**, **V2Pro**, **V2ProPlus**.

## ✨ Features

- 🎭 **Multi-speaker shared backbone**: one GPT+SoVITS backbone serves 10+ speakers, with only ~25 GPT keys + 37 SoVITS keys of per-speaker weights
- 🔀 **Zero-cost speaker switching**: per-speaker weights are dynamically injected on demand — no extra inference cost when switching
- 🔌 **Auto compatibility check**: speakers incompatible with the backbone architecture auto-degrade to full loading without affecting others
- ⚡ **All inference modes**: single-speaker `infer`, token-level streaming `infer_stream`, same-speaker GPU-parallel `infer_batched`
- 🎵 **Timbre/style decoupling**: timbre (Speaker) and style (Prompt) controlled independently, with per-call style overrides
- 🖥️ **WebUI / API support**: `<speaker:name>` tag mixed synthesis, 6+1 MultiSpeaker API endpoints
- ⏱️ **Character-level timestamps**: per-character timestamps for subtitle sync
- 🌐 **Trilingual support**: auto language detection for Chinese/Japanese/English (`auto` / `ja` / `zh` / `en`)

## 🎭 MultiSpeakerTTS Shared-Backbone Inference (Core Feature)

### How It Works

Unlike the traditional "one full model per speaker" approach, `MultiSpeakerTTS` first loads **one shared GPT+SoVITS backbone**, then stores each speaker's fine-tuned weight deltas (~25 GPT keys + 37 SoVITS keys) separately; at inference time, the corresponding weights are **dynamically injected** by speaker name.

Memory/VRAM usage ≈ **1 backbone + 1 speaker's weights**, not "speaker count × full model". On GPU, weight injection adds almost no VRAM overhead, and the benefit grows linearly with the number of speakers.

### Benchmarks (Shared Backbone vs Full Loading)

> [!NOTE]
> **Test environment**: CPU reference environment (no GPU), using real fine-tuned models (CyreneV3.7 / shouanren / LuoTianyi, v2ProPlus-compatible architecture), average of short-text inference runs.

| Metric | Shared Backbone | Full Loading | Notes |
| :--- | :---: | :---: | :--- |
| Per-speaker avg inference latency | 0.7~0.9s | 0.8~0.9s | ⚖️ No performance loss |
| Peak memory (RAM) | **2.77 GB** | 4.65 GB | 💾 **-40%** (CPU measured) |
| 3-speaker init time | 30.0s | 16.2s | One-time weight extraction; zero-cost speaker switching afterwards |

> [!IMPORTANT]
> **Architecture compatibility validation** (real models):
> - ✅ CyreneV3.7, shouanren, LuoTianyi (Agent-LuoTianyi project model) → shared-backbone mode
> - ⚠️ aimisi (v2 architecture, `upsample_initial_channel=512` vs base `768`) → **auto-degrades** to full model loading without affecting other speakers
>
> Memory savings grow with the number of shared speakers (-17% with 2 → -40% with 3); GPU VRAM savings are far higher than the CPU figures above (weight injection is not bandwidth-bound).

| Approach | 1 Speaker | 3 Speakers | 5 Speakers | 10 Speakers |
|------|--------|--------|--------|---------|
| Full loading | ~800MB | ~2.4GB | ~4.0GB | ~8.0GB |
| **MultiSpeakerTTS** | ~800MB | **~1.2GB** | **~1.4GB** | **~2.0GB** |
| VRAM saved | — | **51%** | **65%** | **75%** |

### Usage

```python
from gsv_tts import MultiSpeakerTTS, SpeakerConfig

# Define multiple speakers (model paths also support safetensors directory format)
speakers = [
    SpeakerConfig(
        name="alice",
        gpt_model_path="models/alice_gpt.ckpt",
        sovits_model_path="models/alice_sovits.pth",
        spk_audio_path="audio/alice_ref.wav",
        prompt_audio_path="audio/alice_prompt.ogg",  # Optional, falls back to spk_audio_path
        prompt_audio_text="Hello, I'm Alice.",
    ),
    SpeakerConfig(
        name="bob",
        gpt_model_path="models/bob_gpt.ckpt",
        sovits_model_path="models/bob_sovits.pth",
        spk_audio_path="audio/bob_ref.wav",
        prompt_audio_path="audio/bob_prompt.ogg",
        prompt_audio_text="Hello, I'm Bob.",
    ),
]

# Load all speakers at once (shared backbone + per-speaker weights)
tts = MultiSpeakerTTS(speakers=speakers, use_bert=True)

# Single-speaker inference — auto-routed by speaker name, supports language params and per-call prompt overrides
audio = tts.infer(
    "alice",
    "Good morning!",
    text_language="en",       # "auto" / "ja" / "zh" / "en"
    prompt_language="en",     # "auto" / "ja" / "zh" / "en"
    # prompt_audio_path="other_style.ogg",   # Optional: override the style reference audio for this call
    # prompt_audio_text="Text for the other style.",  # Must be provided when overriding
)
audio.play()

# Streaming inference — token-level streaming with low-latency real-time feedback
for chunk in tts.infer_stream(
    "alice",
    "What a wonderful day it is!",
    text_language="en",
    stream_chunk=25,
    overlap_len=5,
    return_subtitles=True,
):
    chunk.play()

tts.audio_queue.wait()

# Batch inference — true GPU parallelism for same-speaker texts, supports per-sentence language lists
audios = tts.infer_batched(
    [
        ("alice", "Hello"),
        ("alice", "How are you?"),
        ("bob",   "Nice to meet you"),
    ],
    text_languages=["en", "en", "en"],  # or just pass "auto"
)

# Runtime management: add/remove speakers dynamically, no restart needed
tts.add_speaker(SpeakerConfig(name="charlie", ...))
tts.remove_speaker("bob")
print(tts.speaker_names)  # ["alice", "charlie"]
```

> [!TIP]
> **Auto compatibility check**: Architecture parameters (`vocab_size`, `n_layer`, `gin_channels`, `upsample_initial_channel`, etc.) are validated on load. Incompatible speakers **auto-degrade** to full model loading — no user intervention needed.

## 🚀 Quick Start

### Prerequisites

- Python **>= 3.10** (virtual environment recommended)
- Inference backends: **CUDA**, **MPS (Apple Silicon)**, or **CPU**

```bash
# NVIDIA GPU (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Apple Silicon (MPS) or Linux/Windows (CPU Only)
pip install torch torchvision torchaudio
```

### Install GSV-TTS-Lite

> [!WARNING]
> **The multi-speaker (MultiSpeakerTTS) feature is NOT published on PyPI** — the PyPI `gsv-tts-lite` package only supports single-speaker inference. To use multi-speaker shared backbone, you MUST install from this repository:

```bash
git clone https://github.com/jinyiwei2012/gsv-tts-lite-multispeaker.git
cd gsv-tts-lite-multispeaker
pip install -e .
```

### First Run: Automatic Model Download

> [!NOTE]
> On the first `TTS` / `MultiSpeakerTTS` construction, the required pretrained models (several GB) are downloaded automatically to the local cache directory **`~/.cache/gsv`** (configurable via `TTS(models_dir=...)`):
> - GPT model: `s1v3.ckpt`; SoVITS model: `s2Gv2ProPlus.pth`
> - Pretrained components: CNHubert, G2P, speaker verification model, CNRoBERTa (BERT)
>
> The download source is auto-selected by latency: **ModelScope → hf-mirror → HuggingFace**. ModelScope is usually chosen automatically in China; you can also force it via the environment variable:
>
> ```bash
> # Optional: force download mirror modelscope / huggingface / hf-mirror
> export GSV_MIRROR=modelscope
> ```

### Single-Speaker Basic Inference

> [!NOTE]
> For single-speaker scenarios, you can directly use `TTS` (this repo also provides full single-speaker inference capability, see the [Single-Speaker Inference](#-single-speaker-inference-tts) section).

```python
from gsv_tts import TTS

tts = TTS(use_bert=True)
# tts = TTS(use_flash_attn=True) # Recommended if Flash Attention is installed

# Load GPT / SoVITS model weights from the specified path into memory; loads the default model here.
tts.load_gpt_model()
tts.load_sovits_model()

# infer is the most rudimentary inference method, suitable only for short text. It is generally recommended to use infer_batched instead.
audio = tts.infer(
    spk_audio_path="examples\laffey.mp3", # Voice reference audio (Timbre)
    prompt_audio_path="examples\AnAn.ogg", # Style reference audio (Prompt)
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。", # The corresponding text for the style reference audio
    text="へぇー、ここまでしてくれるんですね。", # Target text to be generated
    text_language="auto", # Language of the target text: "auto" / "ja" / "zh" / "en", auto-detected by default
    prompt_language="auto", # Language of the prompt audio text: "auto" / "ja" / "zh" / "en", auto-detected by default
)

audio.play()
tts.audio_queue.wait()
```

## 📖 Single-Speaker Inference (TTS)

> [!NOTE]
> The following are advanced usages of the `TTS` single-speaker engine. MultiSpeakerTTS's `infer` / `infer_stream` / `infer_batched` provide the same capabilities (routed by speaker name).

<details>
<summary><strong>1. Stream Inference / Subtitle Synchronization</strong></summary>

`infer_stream` implements token-level streaming output, significantly reducing first-token latency. `infer`, `infer_stream`, `infer_batched`, and `infer_vc` all support character-level timestamp returns.

```python
import time
import queue
import threading
from gsv_tts import TTS

class SubtitlesQueue:
    def __init__(self):
        self.q = queue.Queue()
        self.t = None

    def process(self):
        last_i = 0
        last_t = time.time()

        while True:
            subtitles, text = self.q.get()

            if subtitles is None:
                break

            for subtitle in subtitles:
                if subtitle["start_s"] > time.time() - last_t:
                    time.sleep(subtitle["start_s"] - (time.time() - last_t))

                if subtitle["end_s"] and subtitle["end_s"] > time.time() - last_t:
                    if subtitle["orig_idx_end"] > last_i:
                        print(text[last_i:subtitle["orig_idx_end"]], end="", flush=True)
                        last_i = subtitle["orig_idx_end"]
                        time.sleep(subtitle["end_s"] - (time.time() - last_t))

        self.t = None

    def add(self, subtitles, text):
        self.q.put((subtitles, text))
        if self.t is None:
            self.t = threading.Thread(target=self.process, daemon=True)
            self.t.start()

tts = TTS(use_bert=True, sovits_cache=[50, 55]) # 50 = stream_chunk * 2 = 25 * 2, 55 = stream_chunk * 2 + overlap_len = 25 * 2 + 5

subtitlesqueue = SubtitlesQueue()

generator = tts.infer_stream(
    spk_audio_path="examples\laffey.mp3",
    prompt_audio_path="examples\AnAn.ogg",
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
    text="へぇー、ここまでしてくれるんですね。",
    text_language="auto", # Language of the target text: "auto" / "ja" / "zh" / "en"
    prompt_language="auto", # Language of the prompt audio text: "auto" / "ja" / "zh" / "en"
    stream_chunk = 25,
    overlap_len = 5,
    return_subtitles=True,
    debug=False,
)

for audio in generator:
    audio.play()
    subtitlesqueue.add(audio.subtitles, audio.orig_text)

tts.audio_queue.wait()
subtitlesqueue.add(None, None)
```

</details>

<details>
<summary><strong>2. Batched Inference</strong></summary>

`infer_batched` is optimized for long-form text and multi-sentence synthesis, supporting different reference audios for different sentences within the same batch.

```python
from gsv_tts import TTS

# gpt_cache: Static cache configuration for the GPT model's CUDA graph, a list of tuples [(batch_size, sequence_length), ...].
# Note: the maximum batch_size determines the max concurrent throughput; the max sequence_length in a batch determines the max generation length per request.
tts = TTS(use_bert=True)

audios = tts.infer_batched(
    spk_audio_paths="examples\laffey.mp3",
    prompt_audio_paths="examples\AnAn.ogg",
    prompt_audio_texts="ちが……ちがう。レイア、貴様は間違っている。",
    texts=["へぇー、ここまでしてくれるんですね。", "The old map crinkled in Leo’s trembling hands."],
    text_languages="auto", # Language of the target texts; accepts str or per-sentence list[str]: "auto" / "ja" / "zh" / "en"
    prompt_languages="auto", # Language of the prompt audio texts; accepts str or per-sentence list[str]: "auto" / "ja" / "zh" / "en"
    bert_batch_size=20,
    sovits_batch_size=10,
)

for i, audio in enumerate(audios):
    audio.save(f"audio{i}.wav")
```

</details>

<details>
<summary><strong>3. Voice Conversion / Speaker Verification</strong></summary>

```python
from gsv_tts import TTS

# Zero-shot voice conversion
tts = TTS(use_bert=True, always_load_cnhubert=True)
audio = tts.infer_vc(
    spk_audio_path="examples\laffey.mp3",
    prompt_audio_path="examples\AnAn.ogg",
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
)
audio.play()

# Speaker verification: check whether two audio clips are the same speaker
tts2 = TTS(use_bert=True, always_load_sv=True)
similarity = tts2.verify_speaker("examples\laffey.mp3", "examples\AnAn.ogg")
print("Speaker Similarity:", similarity)
```

</details>

<details>
<summary><strong>4. Other Function Interfaces</strong></summary>

#### Model Management

- `init_language_module(languages)` — preload necessary language processing modules
- `load_gpt_model(model_paths)` / `load_sovits_model(model_paths)` — load model weights from specified paths into memory
- `unload_gpt_model(model_paths)` / `unload_sovits_model(model_paths)` — unload models from memory to free up resources
- `get_gpt_list()` / `get_sovits_list()` — get the list of currently loaded models
- `to_safetensors(checkpoint_path)` — convert PyTorch checkpoint files (.pth / .ckpt) to the safetensors directory format

#### Audio Cache Management

- `cache_spk_audio(spk_audio_paths)` — preprocess and cache speaker reference audio data
- `cache_prompt_audio(prompt_audio_paths, prompt_audio_texts, prompt_audio_languages)` — preprocess and cache prompt reference audio data
- `del_spk_audio(spk_audio_paths)` / `del_prompt_audio(prompt_audio_paths)` — remove audio data from the cache
- `get_spk_audio_list()` / `get_prompt_audio_list()` — get the list of audio data in the cache

#### Asynchronous Invocations

- `infer_async(...)` — asynchronous version of the `infer` method
- `infer_stream_async(...)` — asynchronous version of the `infer_stream` method
- `infer_batched_async(...)` — asynchronous version of the `infer_batched` method

</details>

## 🌐 WebUI Visual Interface

```bash
cd WebUI
pip install -r requirements.txt   # installs local gsv_tts via -e .. (MultiSpeakerTTS is not on PyPI)
python web.py                     # optional args: --port 9881 / --use_asr / --models_dir ...
```

> [!TIP]
> WebUI supports **single-model / multi-speaker** modes with one-click toggle. Multi-speaker mode supports `<speaker:name>text</speaker:name>` tags for mixed-speaker synthesis with automatic GPU batch parallelism.

## 🔌 API Service Interface

```bash
cd API
pip install -r requirements.txt
```

- Core documentation: [API Guide](API/README.md), [Personal API Docs](API/PERSONAL_API.md)
- Service entry points: `API/personal_api.py` (MultiSpeaker endpoints), `API/realtime_api.py` (real-time streaming)

> [!TIP]
> The FastAPI server includes **6 MultiSpeaker endpoints** (`/multi-speaker/init`, `/multi-speaker/add`, `/multi-speaker/remove`, `/multi-speaker/list`, `/multi-speaker/infer`, `/multi-speaker/batch`) plus the `/multi-speaker/stream` SSE endpoint, supporting multi-speaker management and batch inference.

## 📁 Project Structure

```
gsv_tts/                  # Core Python package (pip install -e .)
├── MultiSpeaker.py       # 🎭 Multi-speaker shared-backbone inference engine: MultiSpeakerTTS (core of this repo)
├── SpeakerWeights.py     # 🎭 Speaker config & weight extraction: SpeakerConfig / SpeakerWeights
├── TTS.py                # Single-speaker inference engine: infer / infer_stream / infer_batched / infer_vc
├── Loader.py             # Weight loading & SoVITS version sniffing
├── Download.py           # Automatic model download (multi-mirror selection)
├── TextProcessor.py      # Text → phonemes / BERT features
├── Player.py             # Audio playback: AudioQueue / AudioClip
├── Config.py             # Global configuration
└── GPT_SoVITS/           # Model architectures + text processing
    ├── GPT/              # GPT semantic model (t2s)
    ├── SoVITS/           # SoVITS acoustic model
    ├── G2P/              # Chinese/Japanese/English phoneme conversion
    ├── Featurizer/       # CNHubert / CNRoBERTa feature extractors
    └── SV/               # Speaker verification model (ERes2Net)
tests/                    # Test scripts (MultiSpeaker self-consistency test)
benchmarks/               # MultiSpeaker performance benchmark scripts
WebUI/                    # Gradio Web UI (own requirements.txt)
API/                      # FastAPI servers (own requirements.txt)
examples/                 # Example reference audio (laffey.mp3 / AnAn.ogg)
```

## 🛠️ Development & Debugging

### Tests

```bash
# Self-consistency test: verifies shared-backbone output matches full-model output (MCD metric)
python tests/test_sovits_sharing.py

# Real-model evaluation (optional args)
python tests/test_sovits_sharing.py --speaker-gpt path/to/speaker_gpt.ckpt --speaker-sovits path/to/speaker_sovits.pth
```

> [!NOTE]
> MCD computation requires `librosa`; without it the metric is skipped with a warning. No pytest needed — the test runs as a plain script.

### Benchmarks

```bash
python benchmarks/bench_multi_speaker.py
```

> [!TIP]
> The script auto-discovers `.ckpt` / `.pth` model files under the repo (or `--models-dir`), pairing them by filename prefix; you can also specify model pairs explicitly with `--gpt` / `--sovits` (repeatable, safetensors directories accepted). It prints a hint and exits when no models are found.

### Model Formats & Compatibility

- **Weight formats**: legacy `.ckpt` / `.pth` checkpoints are supported (loaded via pickle deserialization — a security warning is emitted at startup), as well as the safer **safetensors directory format** (`hps.json` + `model.safetensors`). Convert with `tts.to_safetensors(path)`.
- **SoVITS version sniffing**: the version is auto-detected from the file header bytes (`01`=v2, `05`=v2Pro, `06`=v2ProPlus) or the MD5 of known pretrained files; unrecognized files default to v2 with a warning.
- **Device differences**: MPS/CPU environments force `float32` and clear `sovits_cache`; on CPU, BERT uses an INT8-quantized ONNX model, while GPU uses the original PyTorch model.

### Download Mirrors

The download mirror is auto-selected by latency (ModelScope → hf-mirror → HuggingFace). For download issues during development, force a mirror via the environment variable:

```bash
# Windows (PowerShell)
$env:GSV_MIRROR = "modelscope"   # modelscope / huggingface / hf-mirror
# Linux/macOS
export GSV_MIRROR=modelscope
```

### Common Pitfalls

- Modifying the `sys.modules['utils']` monkey-patch at the top of `Loader.py` breaks deserialization of legacy GPT-SoVITS checkpoints — **do not remove it**.
- `gpt_cache` / `sovits_cache` control CUDA-graph static cache sizes; misconfiguration triggers CUDA graph errors — don't change defaults casually.
- Inference is serialized by `_infer_lock` with automatic cache cleanup (`_empty_cache`); models are lazily loaded (first inference only) — keep this in mind when debugging VRAM usage.

## ❓ FAQ

**Q1: The multi-speaker feature can't be installed from PyPI?**
The MultiSpeakerTTS feature of this repo is not yet published to PyPI. Install from this repository with `pip install -e .`.

**Q2: A speaker's VRAM/memory usage is abnormally high — it didn't use the shared backbone?**
The speaker's model architecture is incompatible with the base model (e.g. v2's `upsample_initial_channel=512` vs base `768`) and it auto-degraded to full loading. The load log will show this; we recommend fine-tuned models matching the base architecture (v2ProPlus).

**Q3: I changed the reference audio content, but inference results didn't change?**
The speaker/style reference audio cache is keyed by path — replacing the file content at the same path hits the stale cache. After changing content, remove the cache entries (`del_spk_audio` / `del_prompt_audio`) or use a new file path.

**Q4: Model downloads on first run are slow or failing?**
Force a mirror with the `GSV_MIRROR` environment variable (`modelscope` is recommended in China), or place the model files manually into the cache directory (default `~/.cache/gsv`; see the "First Run" section for the layout).

**Q5: CUDA graph related errors?**
Usually caused by misconfigured `gpt_cache` / `sovits_cache` — restore the defaults; MPS/CPU environments don't need these params.

**Q6: `weights_only=False` security warning when loading models?**
This is intentional for compatibility with legacy GPT-SoVITS checkpoints. Only load models from trusted sources, or convert weights to the safetensors directory format (`tts.to_safetensors`) to eliminate the risk.

## ⚡ Flash Attention

For **lower latency** and **higher throughput**, enabling Flash Attention is highly recommended:

- 🐧 **Linux / Build from Source**: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- 🪟 **Windows Users**: [lldacing/flash-attention-windows-wheel](https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main) (pre-compiled wheels)

> [!TIP]
> After installation, set `use_flash_attn=True` in your TTS configuration to enjoy the acceleration! 🚀

## Credits

Special thanks to the following projects:
- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [chinokikiss/GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite)
