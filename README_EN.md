<div align="center">

> [!IMPORTANT]
> ### 🔀 Standalone MultiSpeaker Development Fork
> This repository originates from a fork of [GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite): the upstream repo has no `multi-speaker-inference` branch — it was created within the fork and later spun off into this standalone repository, focused on independently developing and optimizing **multi-speaker (MultiSpeakerTTS) shared-backbone inference**.
>
> - **Upstream repo**: [chinokikiss/GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite) (published on PyPI as `gsv-tts-lite`)
> - The multi-speaker features are **NOT yet published to PyPI** — this repo is the only source; upstream bug fixes can be synced via cherry-pick / merge

</div>

<div align="center">
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

**In one sentence: this project lets an AI read text aloud in "many different voices", and it saves a lot of VRAM/memory compared to the traditional approach.**

Imagine you are making a podcast or game voice-over with 3 characters. The traditional way loads a complete AI model per character — 3 characters means 3 full models, which is very resource-hungry.

This project needs only **1 shared "base model" (backbone)**, plus a **small "tuning pack" (~5-15% lightweight weights) per character**. Synthesis automatically switches by character name — **the more characters, the more you save** (measured 40%~75%).

- Supported languages: **Chinese, Japanese, English**
- Supported models: **V2 / V2Pro / V2ProPlus** (don't worry about this — the defaults work)
- Besides multi-speaker, full single-speaker features are also included (see [Usage](#-single-speaker-inference-tts) below)

> Don't want to write code? Use the [WebUI](#-webui-visual-interface) — click around in your browser.

## ✨ Features

- 🎭 **Multi-speaker shared backbone**: 1 model serves 10+ speakers, each needs only a tiny per-speaker weight pack
- 🔀 **Zero-cost speaker switching**: switch speakers anytime — no lag, no extra memory
- 🔌 **Auto compatibility check**: incompatible speaker models auto-degrade instead of crashing, without affecting other speakers
- ⚡ **Three usage modes**: single-shot, streaming (speaks while generating), batched (auto parallel for same speaker)
- 🎵 **Timbre and style controlled separately**: who the voice sounds like (timbre) and the speaking tone (style) are independent; you can even override the style per call
- ⏱️ **Character-level timestamps**: get per-character timing for subtitles
- 🖥️ **WebUI / API support**: click in the browser, or integrate via API
- 🌐 **Auto language detection**: Chinese/Japanese/English — no need to tell it the language

## 🎭 MultiSpeakerTTS: Shared-Backbone Inference (The Core)

### How It Works (plain-language version)

Think of it as **a voice-acting studio**:

- **Backbone** = the studio's fixed cast and equipment (1 set, shared by everyone)
- **Speaker weights** = each actor's personal "voice-tuning kit" (very small)
- **Switching speakers** = the actor swaps kits; the studio and equipment stay put

The traditional approach is like giving every actor their own complete studio (cast + equipment) — 3 characters means 3 studios, expensive and wasteful. This project: 1 studio + N personal kits. That's the secret behind the memory/VRAM savings.

> Technical detail (for developers): the shared backbone is 1 GPT + SoVITS model; each speaker injects only ~25 GPT weights + 37 SoVITS weights. Weights are dynamically injected by speaker name, and only 1 speaker's weights are active at a time, so usage ≈ 1 backbone + 1 speaker's weights.

### Benchmarks (for the numbers people — feel free to skip)

> [!NOTE]
> Test environment: CPU (no GPU), using real fine-tuned models (CyreneV3.7 / shouanren / LuoTianyi), average of short-text inference runs.

| Metric | Shared Backbone | Traditional Full Loading | Notes |
| :--- | :---: | :---: | :--- |
| Per-speaker avg inference latency | 0.7~0.9s | 0.8~0.9s | ⚖️ No speed loss |
| Peak memory (RAM) | **2.77 GB** | 4.65 GB | 💾 **-40%** |
| 3-speaker init time | 30.0s | 16.2s | One-time prep; zero-cost switching afterwards |

| Approach | 1 Speaker | 3 Speakers | 5 Speakers | 10 Speakers |
|------|--------|--------|--------|---------|
| Traditional full loading | ~800MB | ~2.4GB | ~4.0GB | ~8.0GB |
| **This project (shared backbone)** | ~800MB | **~1.2GB** | **~1.4GB** | **~2.0GB** |
| Saved | — | **51%** | **65%** | **75%** |

> [!IMPORTANT]
> **Speaker model compatibility**: ideally all speaker models are the same generation as the backbone (v2ProPlus architecture). If not — no problem: the program auto-detects it and degrades that speaker to traditional full loading. You just lose the memory savings for that speaker, nothing crashes.

### Usage (copy & paste)

```python
from gsv_tts import MultiSpeakerTTS, SpeakerConfig

# Step 1: tell the program your speakers (replace paths with your own models/audio)
speakers = [
    SpeakerConfig(
        name="alice",                     # speaker name (anything you like)
        gpt_model_path="models/alice_gpt.ckpt",    # this speaker's GPT model
        sovits_model_path="models/alice_sovits.pth",  # this speaker's SoVITS model
        spk_audio_path="audio/alice_ref.wav",      # timbre reference audio
        prompt_audio_path="audio/alice_prompt.ogg", # style reference audio (optional, defaults to timbre audio)
        prompt_audio_text="Hello, I'm Alice.",  # what is said in the style reference audio
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

# Step 2: load all speakers at once
tts = MultiSpeakerTTS(speakers=speakers, use_bert=True)

# Step 3: synthesize by speaker name
audio = tts.infer("alice", "Today is a great day!", text_language="en")
audio.play()
tts.audio_queue.wait()

# Mix multiple speakers in one script? Use the <speaker:name> style list
audios = tts.infer_batched(
    [
        ("alice", "Hello there"),
        ("bob",   "Nice to meet you"),
    ],
    text_languages=["en", "en"],
)

# Add / remove speakers at runtime, no restart needed
tts.add_speaker(SpeakerConfig(name="charlie", ...))
tts.remove_speaker("bob")
```

## 🚀 Quick Start

### What You Need

- ✅ A computer (CPU works; an NVIDIA GPU is faster)
- ✅ Python **3.10 or newer** (search "Python install" if unsure)
- ✅ Internet access (the first run downloads models, ~5-10 GB)
- ✅ Disk space: models are stored at `~/.cache/gsv` by default (change with the `models_dir` argument)

### Install (3 commands)

```bash
# 1. Install PyTorch (the deep-learning framework)
#    With an NVIDIA GPU (China mirror; switch to https://download.pytorch.org/whl/cu128 if slow):
pip install torch torchvision torchaudio --index-url https://mirrors.aliyun.com/pytorch-wheels/cu128
#    Without a GPU (Mac / regular PC):
#    pip install torch torchvision torchaudio

# 2. Clone this repo and install it
git clone https://github.com/jinyiwei2012/gsv-tts-lite-multispeaker.git
cd gsv-tts-lite-multispeaker
pip install -e .
```

> [!WARNING]
> **Important**: the multi-speaker feature is NOT on PyPI yet (`pip install gsv-tts-lite` only gives the single-speaker version), so you **must** install from this repository as shown above.

### First Run: Automatic Model Download (one time only)

The first time you run it, the program **automatically downloads** all the "materials" (pretrained models) needed for synthesis to `~/.cache/gsv`:

| File | What it does |
| :--- | :--- |
| `s1v3.ckpt` | GPT model: decides "what to say, how to say it" |
| `s2Gv2ProPlus.pth` | SoVITS model: turns semantics into sound |
| `chinese-hubert-base` | Speech feature extraction (for reference audio) |
| `g2p` | Text-to-pronunciation |
| `sv` | Speaker verification (judges who the voice sounds like) |
| `chinese-roberta-wwm-ext-large` | Chinese understanding booster (better Chinese output) |

The download takes minutes to tens of minutes depending on your network. In China the ModelScope mirror is usually chosen automatically; if it's slow or fails, force a mirror:

```bash
# Windows (PowerShell)
$env:GSV_MIRROR = "modelscope"
# Linux/macOS
export GSV_MIRROR=modelscope
```

### Single-Speaker Basic Inference (want to hear it right now?)

```python
from gsv_tts import TTS

tts = TTS(use_bert=True)

# The example audio files in this repo work out of the box
audio = tts.infer(
    spk_audio_path="examples/laffey.mp3",   # timbre reference: whose voice
    prompt_audio_path="examples/AnAn.ogg",  # style reference: what tone
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",  # what the style reference says
    text="Hello, world!",                   # text to synthesize
    text_language="en",                     # text language: auto detects
)

audio.play()
tts.audio_queue.wait()
```

## 📖 Single-Speaker Inference (TTS)

> The advanced usage below belongs to the `TTS` single-speaker engine. The multi-speaker engine (MultiSpeakerTTS) provides the same capabilities.

<details>
<summary><strong>1. Streaming synthesis (speaks while generating — great for real-time chat)</strong></summary>

```python
from gsv_tts import TTS

tts = TTS(use_bert=True, sovits_cache=[50, 55])

for chunk in tts.infer_stream(
    spk_audio_path="examples/laffey.mp3",
    prompt_audio_path="examples/AnAn.ogg",
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
    text="へぇー、ここまでしてくれるんですね。",
    text_language="auto",
    prompt_language="auto",
    stream_chunk=25,
    overlap_len=5,
    debug=False,
):
    chunk.play()

tts.audio_queue.wait()
```

> Want per-character timestamps for subtitles? Add `return_subtitles=True` — every character's start/end time is in the result.

</details>

<details>
<summary><strong>2. Batched synthesis (more efficient for long text / many sentences)</strong></summary>

```python
from gsv_tts import TTS

tts = TTS(use_bert=True)

audios = tts.infer_batched(
    spk_audio_paths="examples/laffey.mp3",
    prompt_audio_paths="examples/AnAn.ogg",
    prompt_audio_texts="ちが……ちがう。レイア、貴様は間違っている。",
    texts=["Hello", "The old map crinkled in Leo's trembling hands."],
    text_languages="auto",
    prompt_languages="auto",
)

for i, audio in enumerate(audios):
    audio.save(f"audio{i}.wav")
```

</details>

<details>
<summary><strong>3. Voice conversion & speaker verification</strong></summary>

```python
from gsv_tts import TTS

# Voice conversion: say the content of one audio clip in another person's voice
tts = TTS(use_bert=True, always_load_cnhubert=True)
audio = tts.infer_vc(
    spk_audio_path="examples/laffey.mp3",    # target timbre
    prompt_audio_path="examples/AnAn.ogg",   # source audio content
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
)
audio.play()

# Speaker verification: are two clips the same person?
tts2 = TTS(use_bert=True, always_load_sv=True)
similarity = tts2.verify_speaker("examples/laffey.mp3", "examples/AnAn.ogg")
print("Speaker similarity:", similarity)
```

</details>

<details>
<summary><strong>4. More function interfaces (for developers)</strong></summary>

- `load_gpt_model(path)` / `load_sovits_model(path)` — load model weights into memory
- `unload_gpt_model(path)` / `unload_sovits_model(path)` — unload models to free resources
- `get_gpt_list()` / `get_sovits_list()` — list loaded models
- `to_safetensors(path)` — convert .pth/.ckpt models to the safer safetensors format
- `cache_spk_audio(path)` / `cache_prompt_audio(path, text)` — pre-cache audio to reduce first-call latency
- `infer_async(...)` / `infer_stream_async(...)` / `infer_batched_async(...)` — async versions

</details>

## 🌐 WebUI Visual Interface (no code needed)

Prefer clicking over coding? Use the WebUI: open it in a browser, upload audio, type text, hit the button. Supports both single-model and multi-speaker modes (multi-speaker supports `<speaker:name>text</speaker:name>` tags for mixed synthesis).

```bash
cd WebUI
pip install -r requirements.txt   # installs local gsv_tts via -e .. (MultiSpeakerTTS is not on PyPI)
python web.py                     # optional args: --port 9881 / --use_asr / --models_dir ...
```

Your browser opens `http://127.0.0.1:9881` automatically.

## 🔌 API Service Interface (for developers)

```bash
cd API
pip install -r requirements.txt
```

- Core docs: [API Guide](API/README.md), [Personal API Docs](API/PERSONAL_API.md)
- Entry points: `API/personal_api.py` (MultiSpeaker endpoints), `API/realtime_api.py` (real-time streaming)

> 6 multi-speaker management endpoints (`/multi-speaker/init`, `/add`, `/remove`, `/list`, `/infer`, `/batch`) + `/multi-speaker/stream` streaming endpoint for programmatic integration.

<details>
<summary><strong>📁 Project Structure (for developers)</strong></summary>

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

</details>

<details>
<summary><strong>🛠️ Development & Debugging (for developers)</strong></summary>

**Tests** (no pytest needed — run as scripts):

```bash
# Self-consistency test: shared backbone output vs full model (MCD metric)
python tests/test_sovits_sharing.py

# Real-model evaluation
python tests/test_sovits_sharing.py --speaker-gpt path/to/gpt.ckpt --speaker-sovits path/to/sovits.pth
```

> MCD needs `librosa`; without it the metric is skipped with a warning.

**Benchmarks** (auto-discover repo models, or specify manually):

```bash
python benchmarks/bench_multi_speaker.py                          # auto-pair .ckpt/.pth under the repo
python benchmarks/bench_multi_speaker.py --models-dir models      # scan a custom directory
python benchmarks/bench_multi_speaker.py --gpt a.ckpt --sovits b.pth   # explicit pairs (repeatable)
```

**Model formats & compatibility**:

- Legacy `.ckpt` / `.pth` checkpoints are supported (a security warning at load time is normal), plus the safer **safetensors directory format** (`hps.json` + `model.safetensors`); convert with `tts.to_safetensors(path)`.
- SoVITS version auto-detection: header bytes (`01`=v2, `05`=v2Pro, `06`=v2ProPlus); unknown files default to v2 with a warning.
- Device differences: Mac/CPU force float32 and disable some caches; CPU uses an INT8-quantized BERT, GPU uses the original model.

**Common pitfalls**:

- Don't remove the `sys.modules['utils']` monkey-patch at the top of `Loader.py` (legacy checkpoints need it).
- Don't change `gpt_cache` / `sovits_cache` casually — misconfiguration triggers CUDA graph errors.
- Inference is serialized by `_infer_lock` with automatic cache cleanup; models are lazily loaded (first inference only).

</details>

## ❓ FAQ

**Q1: `pip install gsv-tts-lite` doesn't include multi-speaker?**
The multi-speaker feature isn't on PyPI yet — you must install from this repo: `git clone` + `pip install -e .` (see [Quick Start](#-quick-start)).

**Q2: First-run model download is slow or failing?**
Force a mirror with the `GSV_MIRROR` env var (`modelscope` recommended in China), or manually place the model files into `~/.cache/gsv` (file list in [First Run](#first-run-automatic-model-download-one-time-only)).

**Q3: One speaker uses way too much memory — shared backbone not applied?**
That speaker's model is incompatible with the backbone (e.g. v2 vs v2ProPlus) and was auto-degraded to traditional full loading (see the logs). To save memory, unify all speaker models to the v2ProPlus architecture.

**Q4: I changed the reference audio content but results didn't change?**
The audio cache is keyed by path — the same path with new content still hits the old cache. After changing content, delete the cache (`del_spk_audio` / `del_prompt_audio`) or use a new file name.

**Q5: CUDA graph errors?**
Usually caused by modified `gpt_cache` / `sovits_cache` — restore defaults; Mac/CPU environments don't need these params.

**Q6: `weights_only=False` security warning when loading models?**
Intentional for legacy checkpoint compatibility. Only load models you trust, or convert to safetensors (`tts.to_safetensors`) to eliminate the risk.

<details>
<summary><strong>📖 Glossary (plain-language definitions)</strong></summary>

| Term | Plain-language meaning |
| :--- | :--- |
| **GPT model** | Decides "what to say and how to say it" (text → semantics) |
| **SoVITS model** | Turns semantics into sound (semantics → audio) |
| **Backbone** | The single model shared by all speakers |
| **Weights** | "What the model has learned" — each speaker's small personal tuning pack |
| **Timbre reference audio** | Tells the program "whose voice to use" (a few seconds of speech) |
| **Style reference audio** | Tells the program "what tone/emotion to use" (optional) |
| **VRAM** | The memory on your graphics card — more is better |
| **safetensors** | A safer model file format (alternative to .pth/.ckpt) |
| **v2 / v2Pro / v2ProPlus** | Three generations of SoVITS models, newer = better |
| **Fine-tuning** | Training a dedicated speaker model from one person's voice data |
| **Phoneme** | The smallest pronunciation unit of a language |
| **BERT** | A language model that boosts Chinese understanding |
| **RTF** | Real-time factor: time to synthesize 1s of audio; < 1 means faster than real-time |

</details>

<details>
<summary><strong>⚡ Flash Attention (optional speed boost)</strong></summary>

For lower latency and higher throughput you can enable Flash Attention:

- 🐧 **Linux**: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) (build from source)
- 🪟 **Windows**: [lldacing/flash-attention-windows-wheel](https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main) (pre-built wheels)

Then set `use_flash_attn=True` in your code.

</details>

## Credits

Special thanks to the following projects:
- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [chinokikiss/GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite)
