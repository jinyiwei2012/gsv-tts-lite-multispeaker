<div align="center">

> [!IMPORTANT]
> ### 🔀 MultiSpeaker 独立开发仓库
> 本仓库源自 [GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite) 的 fork：上游仓库并无 `multi-speaker-inference` 分支，该分支在 fork 中创建后独立为本仓库，专注多说话人（MultiSpeakerTTS）共享骨干推理的独立开发与优化。
>
> - **上游主仓库**：[chinokikiss/GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite)（PyPI 发版 `gsv-tts-lite`）
> - 多说话人功能**尚未发布至 PyPI**，本仓库是唯一来源；上游 bug 修复可通过 cherry-pick / merge 同步到本仓库

</div>

<div align="center">
  <h1>GSV-TTS-Lite · MultiSpeaker</h1>

  <p>
    GPT-SoVITS 多角色共享骨干推理引擎（MultiSpeakerTTS）
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

## 关于项目 (About)

本仓库是 **GSV-TTS-Lite 的多说话人（MultiSpeaker）独立开发仓库**，核心能力是 **MultiSpeakerTTS 多角色共享骨干推理**：

传统方案为每个角色加载一套完整模型，显存/内存随角色数量线性增长；本仓库仅加载**一套共享 GPT+SoVITS 骨干**，每个角色只注入约 5-15% 的轻量专属权重（约 25 GPT keys + 37 SoVITS keys），按角色名**动态切换、零开销**——多角色场景显存/内存节省 **40%~75%**，**角色越多节省越大**。

同时保留完整单说话人推理能力（`TTS`）：Token 级流式、批量并行、字级时间戳、零样本音色迁移、声纹识别，并提供 **WebUI** 与 **API** 服务。

支持语言：**中日英**；支持模型：**V2**、**V2Pro**、**V2ProPlus**。

## ✨ 功能特性 (Features)

- 🎭 **多角色共享骨干**：一套 GPT+SoVITS 骨干承载 10+ 角色，每角色仅需约 25 GPT keys + 37 SoVITS keys 的专属权重
- 🔀 **零开销角色切换**：角色权重按需动态注入，切换角色无额外推理成本
- 🔌 **自动兼容性校验**：角色模型与骨干架构不匹配时自动降级为全量加载，不中断其他角色
- ⚡ **全模式推理**：单角色 `infer`、Token 级流式 `infer_stream`、同角色 GPU 并行 `infer_batched`
- 🎵 **音色与风格解耦**：音色（Speaker）与风格（Prompt）独立控制，支持按次调用覆盖风格参考
- 🖥️ **WebUI / API 全支持**：`<speaker:角色名>` 标签混用合成、6+1 个 MultiSpeaker API 端点
- ⏱️ **字级时间戳**：逐字返回时间戳，支持字幕同步
- 🌐 **三语支持**：中日英自动语言检测（`auto` / `ja` / `zh` / `en`）

## 🎭 MultiSpeakerTTS 共享骨干推理（核心特性）

### 工作原理

与传统"每角色全量加载一套模型"不同，`MultiSpeakerTTS` 先加载**一套共享的 GPT+SoVITS 骨干**，再将每个角色的微调权重差异（约 25 GPT keys + 37 SoVITS keys）单独保存；推理时按角色名**动态注入**对应权重。

因此内存/显存占用 ≈ **1 套骨干 + 1 份角色权重**，而非"角色数 × 全量模型"。GPU 环境下权重注入几乎不增加显存开销，收益随角色数量线性增长。

### 实测基准（共享骨干 vs 全量加载）

> [!NOTE]
> **测试环境**：CPU 参考环境（无 GPU），使用真实微调角色模型（CyreneV3.7 / shouanren / LuoTianyi，v2ProPlus 兼容架构），短文本推理平均值。

| 指标 | 共享骨干 | 全量加载 | 说明 |
| :--- | :---: | :---: | :--- |
| 单角色推理平均延迟 | 0.7~0.9s | 0.8~0.9s | ⚖️ 无性能损失 |
| 峰值内存 (RAM) | **2.77 GB** | 4.65 GB | 💾 节省 **40%**（CPU 实测） |
| 3 角色初始化耗时 | 30.0s | 16.2s | 含一次性权重提取，后续换角色零开销 |

> [!IMPORTANT]
> **架构兼容性验证**（真实模型）：
> - ✅ CyreneV3.7、shouanren、LuoTianyi（Agent-LuoTianyi 项目角色模型）→ 共享骨干模式
> - ⚠️ aimisi（v2 架构，`upsample_initial_channel=512` vs base `768`）→ **自动降级**为完整模型加载，不影响其他角色
>
> 内存节省随共享角色数量增长（2 角色 -17% → 3 角色 -40%）；GPU 环境显存节省远高于 CPU 实测值（权重注入不依赖显存带宽）。

| 方案 | 1 角色 | 3 角色 | 5 角色 | 10 角色 |
|------|--------|--------|--------|---------|
| 全量加载 | ~800MB | ~2.4GB | ~4.0GB | ~8.0GB |
| **MultiSpeakerTTS** | ~800MB | **~1.2GB** | **~1.4GB** | **~2.0GB** |
| 显存节省 | — | **51%** | **65%** | **75%** |

### 使用方法

```python
from gsv_tts import MultiSpeakerTTS, SpeakerConfig

# 定义多个角色（模型路径同样支持 safetensors 目录格式）
speakers = [
    SpeakerConfig(
        name="alice",
        gpt_model_path="models/alice_gpt.ckpt",
        sovits_model_path="models/alice_sovits.pth",
        spk_audio_path="audio/alice_ref.wav",
        prompt_audio_path="audio/alice_prompt.ogg",  # 可选，默认复用 spk_audio_path
        prompt_audio_text="こんにちは、アリスです。",
    ),
    SpeakerConfig(
        name="bob",
        gpt_model_path="models/bob_gpt.ckpt",
        sovits_model_path="models/bob_sovits.pth",
        spk_audio_path="audio/bob_ref.wav",
        prompt_audio_path="audio/bob_prompt.ogg",
        prompt_audio_text="こんにちは、ボブです。",
    ),
]

# 一次性加载所有角色（共享骨干 + 角色专属权重）
tts = MultiSpeakerTTS(speakers=speakers, use_bert=True)

# 单角色推理——按角色名自动路由，支持语言参数与按次调用的 prompt 覆盖
audio = tts.infer(
    "alice",
    "今日も頑張りましょう！",
    text_language="ja",       # "auto" / "ja" / "zh" / "en"
    prompt_language="ja",     # "auto" / "ja" / "zh" / "en"
    # prompt_audio_path="other_style.ogg",   # 可选：临时覆盖风格参考音频
    # prompt_audio_text="別のスタイルのテキスト。",  # 覆盖时需同步提供对应文本
)
audio.play()

# 流式推理——Token 级流式输出，低延迟实时反馈
for chunk in tts.infer_stream(
    "alice",
    "へぇー、ここまでしてくれるんですね。",
    text_language="ja",
    stream_chunk=25,
    overlap_len=5,
    return_subtitles=True,
):
    chunk.play()

tts.audio_queue.wait()

# 批量推理——相同角色自动 GPU 并行，支持逐句语言指定
audios = tts.infer_batched(
    [
        ("alice", "こんにちは"),
        ("alice", "お元気ですか"),
        ("bob",   "よろしくお願いします"),
    ],
    text_languages=["ja", "ja", "ja"],  # 或直接传 "auto"
)

# 运行时管理：动态增删角色，无需重启
tts.add_speaker(SpeakerConfig(name="charlie", ...))
tts.remove_speaker("bob")
print(tts.speaker_names)  # ["alice", "charlie"]
```

> [!TIP]
> **自动兼容性校验**：加载时自动比对角色模型与基模型的架构参数（`vocab_size`、`n_layer`、`gin_channels`、`upsample_initial_channel` 等）。不兼容的角色会**自动降级**为完整模型加载，无需用户干预。

## 🚀 快速开始 (Quick Start)

### 环境准备

- Python **>= 3.10**，建议使用虚拟环境
- 支持 **CUDA**、**MPS (Apple Silicon)**、**CPU** 三种推理后端

```bash
# NVIDIA GPU (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Apple Silicon (MPS) 或 Linux/Windows (仅 CPU)
pip install torch torchvision torchaudio
```

### 安装 GSV-TTS-Lite

> [!WARNING]
> **多说话人（MultiSpeakerTTS）功能尚未发布到 PyPI**，PyPI 的 `gsv-tts-lite` 包仅有单说话人推理能力。如需使用多角色共享骨干，必须从本仓库安装：

```bash
git clone https://github.com/jinyiwei2012/gsv-tts-lite-multispeaker.git
cd gsv-tts-lite-multispeaker
pip install -e .
```

### 首次运行：模型自动下载

> [!NOTE]
> 首次构造 `TTS` / `MultiSpeakerTTS` 时，程序会自动下载所需预训练模型（数 GB）到本地缓存目录 **`~/.cache/gsv`**（可通过 `TTS(models_dir=...)` 自定义）：
> - GPT 模型：`s1v3.ckpt`；SoVITS 模型：`s2Gv2ProPlus.pth`
> - 预训练组件：CNHubert、G2P、声纹模型、CNRoBERTa（BERT）
>
> 下载源按延迟自动选择：**ModelScope → hf-mirror → HuggingFace**。国内网络环境一般会自动选中 ModelScope；也可用环境变量强制指定：
>
> ```bash
> # 可选：强制指定下载镜像 modelscope / huggingface / hf-mirror
> set GSV_MIRROR=modelscope
> ```

### 单说话人基础推理

> [!NOTE]
> 仅需单角色的场景，可直接使用 `TTS`（本仓库同样提供完整单说话人推理能力，见[单说话人推理](#-单说话人推理-tts)一节）。

```python
from gsv_tts import TTS

tts = TTS(use_bert=True)
# tts = TTS(use_flash_attn=True) 如果安装了Flash Attention，建议这样设置

# 将 GPT / SoVITS 模型权重从指定路径加载到内存中，这里加载默认模型。
tts.load_gpt_model()
tts.load_sovits_model()

# infer 是最简单、最原始的推理方式，只适用于短文本推理，一般建议用 infer_batched 替代 infer 推理。
audio = tts.infer(
    spk_audio_path="examples\laffey.mp3", # 音色参考音频
    prompt_audio_path="examples\AnAn.ogg", # 风格参考音频
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。", # 风格参考音频对应的文本
    text="へぇー、ここまでしてくれるんですね。", # 目标生成文本
    text_language="auto", # 目标文本语言："auto" / "ja" / "zh" / "en"，默认自动检测
    prompt_language="auto", # 参考音频文本语言："auto" / "ja" / "zh" / "en"，默认自动检测
)

audio.play()
tts.audio_queue.wait()
```

## 📖 单说话人推理 (TTS)

> [!NOTE]
> 以下为 `TTS` 单说话人引擎的进阶用法。MultiSpeakerTTS 的 `infer` / `infer_stream` / `infer_batched` 均具备同等能力（按角色名路由）。

<details>
<summary><strong>1. 流式推理 / 字幕同步</strong></summary>

`infer_stream` 实现了 Token 级别的流式输出，显著降低首字延迟。`infer`、`infer_stream`、`infer_batched`、`infer_vc` 均支持字级时间戳返回。

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
    text_language="auto", # 目标文本语言："auto" / "ja" / "zh" / "en"
    prompt_language="auto", # 参考音频文本语言："auto" / "ja" / "zh" / "en"
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
<summary><strong>2. 批量推理</strong></summary>

`infer_batched` 专为长文本及多句合成场景优化，支持在同一批次中为不同句子指定不同的参考音频。

```python
from gsv_tts import TTS

# gpt_cache: GPT 模型 CUDA graph 的静态缓存配置，参数为元组列表 [(batch_size, sequence_length), ...]。
# 注意：设置的最大 batch_size 决定了该模式下的最大并发吞吐量；同一批次的最大 sequence_length 决定单次生成的最大长度限制。
tts = TTS(use_bert=True)

audios = tts.infer_batched(
    spk_audio_paths="examples\laffey.mp3",
    prompt_audio_paths="examples\AnAn.ogg",
    prompt_audio_texts="ちが……ちがう。レイア、貴様は間違っている。",
    texts=["へぇー、ここまでしてくれるんですね。", "The old map crinkled in Leo’s trembling hands."],
    text_languages="auto", # 目标文本语言，支持 str 或逐句 list[str]："auto" / "ja" / "zh" / "en"
    prompt_languages="auto", # 参考音频文本语言，支持 str 或逐句 list[str]："auto" / "ja" / "zh" / "en"
    bert_batch_size=20,
    sovits_batch_size=10,
)

for i, audio in enumerate(audios):
    audio.save(f"audio{i}.wav")
```

</details>

<details>
<summary><strong>3. 音色迁移 / 声纹识别</strong></summary>

```python
from gsv_tts import TTS

# 零样本音色迁移（变声）
tts = TTS(use_bert=True, always_load_cnhubert=True)
audio = tts.infer_vc(
    spk_audio_path="examples\laffey.mp3",
    prompt_audio_path="examples\AnAn.ogg",
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
)
audio.play()

# 声纹识别：对比两段音频是否为同一说话人
tts2 = TTS(use_bert=True, always_load_sv=True)
similarity = tts2.verify_speaker("examples\laffey.mp3", "examples\AnAn.ogg")
print("声纹相似度：", similarity)
```

</details>

<details>
<summary><strong>4. 其他函数接口</strong></summary>

#### 模型管理

- `init_language_module(languages)` — 预加载必要的语言处理模块
- `load_gpt_model(model_paths)` / `load_sovits_model(model_paths)` — 将模型权重从指定路径加载到内存
- `unload_gpt_model(model_paths)` / `unload_sovits_model(model_paths)` — 从内存中卸载模型释放资源
- `get_gpt_list()` / `get_sovits_list()` — 获取当前已加载的模型列表
- `to_safetensors(checkpoint_path)` — 将 PyTorch 权重文件（.pth / .ckpt）转换为 safetensors 目录格式

#### 音频缓存管理

- `cache_spk_audio(spk_audio_paths)` — 预处理并缓存音色参考音频
- `cache_prompt_audio(prompt_audio_paths, prompt_audio_texts, prompt_audio_languages)` — 预处理并缓存风格参考音频
- `del_spk_audio(spk_audio_paths)` / `del_prompt_audio(prompt_audio_paths)` — 从缓存中移除音频数据
- `get_spk_audio_list()` / `get_prompt_audio_list()` — 获取缓存中的音频数据列表

#### 异步调用

- `infer_async(...)` — `infer` 方法的异步版本
- `infer_stream_async(...)` — `infer_stream` 方法的异步版本
- `infer_batched_async(...)` — `infer_batched` 方法的异步版本

</details>

## 🌐 WebUI 可视化界面

```bash
cd WebUI
pip install -r requirements.txt   # 自动以 -e .. 安装本仓库 gsv_tts（MultiSpeakerTTS 未发布至 PyPI）
python web.py                     # 可选参数: --port 9881 / --use_asr / --models_dir ...
```

> [!TIP]
> WebUI 支持**单模型 / 多角色**两种推理模式，一键切换。多角色模式下支持 `<speaker:角色名>文本</speaker:角色名>` 标签混用，自动 GPU 批量并行。

## 🔌 API 服务接口

```bash
cd API
pip install -r requirements.txt
```

- 核心文档：[API 详细指南](API/README.md)、[Personal API 文档](API/PERSONAL_API.md)
- 服务入口：`API/personal_api.py`（MultiSpeaker 端点）、`API/realtime_api.py`（实时流式）

> [!TIP]
> FastAPI 服务包含 **6 个 MultiSpeaker 端点**（`/multi-speaker/init`、`/multi-speaker/add`、`/multi-speaker/remove`、`/multi-speaker/list`、`/multi-speaker/infer`、`/multi-speaker/batch`）及 `/multi-speaker/stream` SSE 流式端点，支持多角色管理与批量推理。

## 📁 项目结构 (Project Structure)

```
gsv_tts/                  # 核心 Python 包（pip install -e . 安装）
├── MultiSpeaker.py       # 🎭 多说话人共享骨干推理引擎：MultiSpeakerTTS（本仓库核心）
├── SpeakerWeights.py     # 🎭 角色配置与权重提取：SpeakerConfig / SpeakerWeights
├── TTS.py                # 单说话人推理引擎：infer / infer_stream / infer_batched / infer_vc
├── Loader.py             # 权重加载与 SoVITS 版本嗅探
├── Download.py           # 模型自动下载（多镜像选择）
├── TextProcessor.py      # 文本 → 音素 / BERT 特征
├── Player.py             # 音频播放：AudioQueue / AudioClip
├── Config.py             # 全局配置
└── GPT_SoVITS/           # 模型架构 + 文本处理
    ├── GPT/              # GPT 语义模型（t2s）
    ├── SoVITS/           # SoVITS 声学模型
    ├── G2P/              # 中日英音素转换
    ├── Featurizer/       # CNHubert / CNRoBERTa 特征提取
    └── SV/               # 声纹模型（ERes2Net）
tests/                    # 测试脚本（MultiSpeaker 自洽性测试）
benchmarks/               # MultiSpeaker 性能基准脚本
WebUI/                    # Gradio Web 界面（独立 requirements.txt）
API/                      # FastAPI 服务（独立 requirements.txt）
examples/                 # 示例参考音频（laffey.mp3 / AnAn.ogg）
```

## 🛠️ 开发与调试指南 (Development)

### 测试

```bash
# 自洽性测试：验证共享骨干输出与全量模型一致（MCD 指标）
python tests/test_sovits_sharing.py

# 使用真实微调模型评估（可选参数）
python tests/test_sovits_sharing.py --speaker-gpt path/to/speaker_gpt.ckpt --speaker-sovits path/to/speaker_sovits.pth
```

> [!NOTE]
> MCD 计算需要 `librosa`，未安装时自动跳过该指标并警告。测试无需 pytest，直接以脚本方式运行。

### 基准测试

```bash
python benchmarks/bench_multi_speaker.py
```

> [!TIP]
> 脚本默认自动扫描仓库内（或 `--models-dir` 指定目录）的 `.ckpt` / `.pth` 模型文件，按文件名前缀自动配对；也可用 `--gpt` / `--sovits` 显式指定测试模型对（可重复指定，支持 safetensors 目录）。仓库内没有模型时会打印提示并退出。

### 模型格式与兼容性

- **权重格式**：支持传统 `.ckpt` / `.pth` 检查点（通过 pickle 反序列化加载，启动时会输出安全警告），也支持更安全的 **safetensors 目录格式**（`hps.json` + `model.safetensors`）。可用 `tts.to_safetensors(path)` 转换。
- **SoVITS 版本嗅探**：加载时通过文件头字节（`01`=v2、`05`=v2Pro、`06`=v2ProPlus）或预训练文件 MD5 自动识别版本；无法识别时默认按 v2 处理并输出警告。
- **设备差异**：MPS/CPU 环境强制使用 `float32` 并清空 `sovits_cache`；CPU 下 BERT 使用 INT8 量化 ONNX 模型，GPU 下使用 PyTorch 原版模型。

### 镜像下载

模型下载镜像按延迟自动选择（ModelScope → hf-mirror → HuggingFace）。开发调试下载问题时可用环境变量强制指定：

```bash
# Windows (PowerShell)
$env:GSV_MIRROR = "modelscope"   # modelscope / huggingface / hf-mirror
# Linux/macOS
export GSV_MIRROR=modelscope
```

### 常见坑位

- 修改 `Loader.py` 顶部的 `sys.modules['utils']` monkey-patch 会导致旧版 GPT-SoVITS 检查点反序列化失败——**不要移除**。
- `gpt_cache` / `sovits_cache` 参数控制 CUDA graph 静态缓存尺寸，配置不当会触发 CUDA graph 报错——不要随意改动默认值。
- 推理由 `_infer_lock` 串行化并自动执行显存清理（`_empty_cache`），模型为懒加载（首次推理时才加载），调试显存占用时注意区分。

## ❓ 常见问题 (FAQ)

**Q1：多说话人功能无法从 PyPI 安装？**
本仓库的 MultiSpeakerTTS 功能尚未发布至 PyPI。请从本仓库 `pip install -e .` 安装。

**Q2：某角色显存/内存占用异常高，没有走共享骨干？**
该角色模型架构与基模型不兼容（如 v2 的 `upsample_initial_channel=512` vs 基模型 `768`），已自动降级为全量加载。加载日志会提示；建议统一使用与基模型同架构（v2ProPlus）的微调模型。

**Q3：修改了参考音频内容，但推理结果没变？**
说话人/风格参考音频缓存以路径为键——同一路径替换文件内容后会命中旧缓存。更换内容后请删除对应缓存（`del_spk_audio` / `del_prompt_audio`）或更换文件路径。

**Q4：首次运行下载模型很慢或失败？**
用 `GSV_MIRROR` 环境变量强制指定镜像（国内建议 `modelscope`），或将模型文件手动放置到缓存目录（默认 `~/.cache/gsv`，结构见「首次运行」小节）。

**Q5：报 CUDA graph 相关错误？**
多为 `gpt_cache` / `sovits_cache` 参数配置不当，恢复默认值即可；MPS/CPU 环境不需要这两个参数。

**Q6：加载模型时出现 `weights_only=False` 安全警告？**
这是为兼容旧版 GPT-SoVITS 检查点的刻意行为。仅加载可信来源的模型，或将权重转换为 safetensors 目录格式（`tts.to_safetensors`）以消除该风险。

## ⚡ Flash Attention

追求**更低延迟**和**更高吞吐量**时，强烈建议开启 Flash Attention：

- 🐧 **Linux / 源码构建**：[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- 🪟 **Windows 用户**：[lldacing/flash-attention-windows-wheel](https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main)（预编译 Wheel）

> [!TIP]
> 安装完成后，在 TTS 配置中设置 `use_flash_attn=True` 即可享受加速效果！🚀

## 致谢 (Credits)

特别感谢以下项目：
- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [chinokikiss/GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite)
