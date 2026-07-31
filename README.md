<div align="center">

> [!IMPORTANT]
> ### 🔀 MultiSpeaker 独立开发仓库
> 本仓库源自 [GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite) 的 fork：上游仓库并无 `multi-speaker-inference` 分支，该分支在 fork 中创建后独立为本仓库，专注多说话人（MultiSpeakerTTS）共享骨干推理的独立开发与优化。
>
> - **上游主仓库**：[chinokikiss/GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite)（PyPI 发版 `gsv-tts-lite`）
> - 多说话人功能**尚未发布至 PyPI**，本仓库是唯一来源；上游 bug 修复可通过 cherry-pick / merge 同步到本仓库

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

## 关于项目 (About)

**一句话：这个项目能让 AI 用"多个不同的声音"朗读文字，而且比传统方案省很多显存/内存。**

想象一下：你想做一个播客或游戏配音，里面有 3 个角色。传统做法是每个角色都准备一套完整的 AI 模型——3 个角色就是 3 套模型，非常占资源。

本项目只需要 **1 套"公共模型"（骨干）**，每个角色再配一份**很小的"专属调整包"（约 5-15% 的轻量权重）**。合成时按角色名自动切换，**角色越多，省得越多**（实测省 40%~75%）。

- 支持语言：**中文、日文、英文**
- 支持模型：**V2 / V2Pro / V2ProPlus**（看不懂没关系，用默认的就行）
- 除了多角色，也保留了完整的单说话人功能（见下方 [使用指南](#-单说话人推理-tts)）

> 不用写代码？可以直接用 [WebUI 图形界面](#-webui-可视化界面)，网页上点点点就能合成。

## ✨ 功能特性 (Features)

- 🎭 **多角色共享骨干**：1 套模型服务 10+ 个角色，每个角色只需一份很小的专属权重
- 🔀 **切换角色零成本**：角色随时切换，不卡顿、不额外占内存
- 🔌 **自动兼容检查**：角色模型不匹配时自动降级处理，不会崩，也不影响其他角色
- ⚡ **三种用法都支持**：单句合成、边说边出（流式）、批量合成（同一角色多句自动并行加速）
- 🎵 **音色和风格分开控制**：声音像谁（音色）和说话语气（风格）可以分别指定，每次还能临时换风格
- ⏱️ **字幕时间轴**：每个字的时间都能拿到，方便做字幕
- 🖥️ **WebUI / API 全支持**：网页界面一键操作，也提供接口给程序调用
- 🌐 **中日英自动识别**：不用手动告诉它文本是什么语言

## 🎭 MultiSpeakerTTS：多角色共享骨干（本项目的核心）

### 它是怎么工作的？（大白话版）

把它想象成**一家配音公司**：

- **骨干** = 公司的固定配音团队和设备（1 套，大家共用）
- **角色权重** = 每位配音演员随身带的"声线调整包"（很小）
- **换角色** = 演员换包，团队和设备原地不动

传统方案等于给每位演员都配一套完整的公司（团队+设备），3 个角色就是 3 家公司，又贵又占地方。本项目 1 家公司 + N 个随身包就能干同样的事——这就是省内存/显存的原理。

> 技术细节（给开发者）：共享骨干 = 1 套 GPT + SoVITS 模型；每个角色只注入约 25 个 GPT 权重 + 37 个 SoVITS 权重。合成时按角色名动态注入，同一时刻只有 1 份角色权重生效，所以占用 ≈ 1 套骨干 + 1 份角色权重。

### 实测数据（给想看数字的人，跳过也行）

> [!NOTE]
> 测试环境：CPU（无显卡），使用真实微调角色模型（CyreneV3.7 / shouanren / LuoTianyi），短文本推理平均值。

| 指标 | 共享骨干 | 传统全量加载 | 说明 |
| :--- | :---: | :---: | :--- |
| 单角色推理平均延迟 | 0.7~0.9s | 0.8~0.9s | ⚖️ 速度没有损失 |
| 峰值内存 (RAM) | **2.77 GB** | 4.65 GB | 💾 省 **40%** |
| 3 角色初始化耗时 | 30.0s | 16.2s | 首次一次性准备，之后换角色零开销 |

| 方案 | 1 角色 | 3 角色 | 5 角色 | 10 角色 |
|------|--------|--------|--------|---------|
| 传统全量加载 | ~800MB | ~2.4GB | ~4.0GB | ~8.0GB |
| **本项目（共享骨干）** | ~800MB | **~1.2GB** | **~1.4GB** | **~2.0GB** |
| 节省 | — | **51%** | **65%** | **75%** |

> [!IMPORTANT]
> **角色模型兼容性**：角色模型最好是和公共骨干同一代（v2ProPlus 架构）。版本不同也没关系——程序会自动检测，不兼容的角色自动降级为传统方式加载，只是少了省内存的好处，不会报错。

### 使用方法（复制粘贴就能跑）

```python
from gsv_tts import MultiSpeakerTTS, SpeakerConfig

# 第 1 步：告诉程序有哪些角色（路径换成你自己的模型和音频文件）
speakers = [
    SpeakerConfig(
        name="alice",                     # 角色名（随便起）
        gpt_model_path="models/alice_gpt.ckpt",    # 该角色的 GPT 模型
        sovits_model_path="models/alice_sovits.pth",  # 该角色的 SoVITS 模型
        spk_audio_path="audio/alice_ref.wav",      # 该角色的音色参考音频
        prompt_audio_path="audio/alice_prompt.ogg", # 风格参考音频（可选，默认用音色参考）
        prompt_audio_text="こんにちは、アリスです。",  # 风格参考音频里说的内容
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

# 第 2 步：一次性加载所有角色
tts = MultiSpeakerTTS(speakers=speakers, use_bert=True)

# 第 3 步：按角色名合成语音
audio = tts.infer("alice", "今日も頑張りましょう！", text_language="ja")
audio.play()
tts.audio_queue.wait()

# 想在同一段话里混多个角色？用 <speaker:角色名> 标签
audios = tts.infer_batched(
    [
        ("alice", "こんにちは"),
        ("bob",   "よろしくお願いします"),
    ],
    text_languages=["ja", "ja"],
)

# 运行时还能动态加角色 / 删角色，不用重启
tts.add_speaker(SpeakerConfig(name="charlie", ...))
tts.remove_speaker("bob")
```

## 🚀 快速开始 (Quick Start)

### 需要准备什么？

- ✅ 一台电脑（CPU 也能跑，有 NVIDIA 显卡更快）
- ✅ Python **3.10 或更高**（安装教程：网上搜"Python 安装"）
- ✅ 能联网（首次运行要下载模型，约 5~10 GB）
- ✅ 硬盘空间：模型默认存在 `C:\Users\你的用户名\.cache\gsv`（可用 `models_dir` 参数改位置）

### 安装（3 条命令）

```bash
# 1. 安装 PyTorch（深度学习框架）
#    有 NVIDIA 显卡：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
#    没显卡（Mac / 普通电脑）：
#    pip install torch torchvision torchaudio

# 2. 克隆本仓库并安装
git clone https://github.com/jinyiwei2012/gsv-tts-lite-multispeaker.git
cd gsv-tts-lite-multispeaker
pip install -e .
```

> [!WARNING]
> **注意**：多说话人功能还没有发布到 PyPI（`pip install gsv-tts-lite` 只能装到单说话人版），所以**必须**用上面的方式从本仓库安装。

### 首次运行：自动下载模型（只需要一次）

第一次运行程序时，它会**自动下载**配音所需的所有"材料"（预训练模型）到 `~/.cache/gsv`：

| 文件 | 是干什么的 |
| :--- | :--- |
| `s1v3.ckpt` | GPT 模型：决定"说什么、怎么说" |
| `s2Gv2ProPlus.pth` | SoVITS 模型：把语义变成声音 |
| `chinese-hubert-base` | 语音特征提取（处理参考音频用） |
| `g2p` | 文字转读音 |
| `sv` | 声纹识别（判断声音像谁） |
| `chinese-roberta-wwm-ext-large` | 中文理解增强（提升中文效果） |

下载需要几分钟到几十分钟，取决于网速。国内一般自动选 ModelScope 镜像；如果太慢或失败，可以强制指定镜像：

```bash
# Windows (PowerShell)
$env:GSV_MIRROR = "modelscope"
# Linux/macOS
export GSV_MIRROR=modelscope
```

### 单说话人基础推理（想先听个响？）

```python
from gsv_tts import TTS

tts = TTS(use_bert=True)

# 用仓库自带的示例音频就能直接合成
audio = tts.infer(
    spk_audio_path="examples/laffey.mp3",   # 音色参考：用谁的声音
    prompt_audio_path="examples/AnAn.ogg",  # 风格参考：用什么语气
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",  # 风格参考里说的话
    text="你好，世界！",                      # 要合成的文字
    text_language="zh",                      # 文本语言：auto 自动识别
)

audio.play()
tts.audio_queue.wait()
```

## 📖 单说话人推理 (TTS)

> 下面的进阶用法属于 `TTS` 单说话人引擎。多角色引擎（MultiSpeakerTTS）同样具备这些能力。

<details>
<summary><strong>1. 流式合成（边说边出，适合实时对话）</strong></summary>

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

> 想拿逐字时间戳做字幕？加 `return_subtitles=True`，每个字的开始/结束时间都在结果里。

</details>

<details>
<summary><strong>2. 批量合成（长文本/多句话更高效）</strong></summary>

```python
from gsv_tts import TTS

tts = TTS(use_bert=True)

audios = tts.infer_batched(
    spk_audio_paths="examples/laffey.mp3",
    prompt_audio_paths="examples/AnAn.ogg",
    prompt_audio_texts="ちが……ちがう。レイア、貴様は間違っている。",
    texts=["你好", "The old map crinkled in Leo's trembling hands."],
    text_languages="auto",
    prompt_languages="auto",
)

for i, audio in enumerate(audios):
    audio.save(f"audio{i}.wav")
```

</details>

<details>
<summary><strong>3. 音色迁移（变声）与声纹识别</strong></summary>

```python
from gsv_tts import TTS

# 音色迁移：把一段音频的内容，用另一个人的声音说出来
tts = TTS(use_bert=True, always_load_cnhubert=True)
audio = tts.infer_vc(
    spk_audio_path="examples/laffey.mp3",    # 目标音色
    prompt_audio_path="examples/AnAn.ogg",   # 源音频内容
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
)
audio.play()

# 声纹识别：判断两段音频是不是同一个人
tts2 = TTS(use_bert=True, always_load_sv=True)
similarity = tts2.verify_speaker("examples/laffey.mp3", "examples/AnAn.ogg")
print("声纹相似度：", similarity)
```

</details>

<details>
<summary><strong>4. 更多函数接口（开发者）</strong></summary>

- `load_gpt_model(path)` / `load_sovits_model(path)` — 加载模型权重到内存
- `unload_gpt_model(path)` / `unload_sovits_model(path)` — 卸载模型释放资源
- `get_gpt_list()` / `get_sovits_list()` — 查看已加载的模型
- `to_safetensors(path)` — 把 .pth/.ckpt 模型转换成更安全的 safetensors 格式
- `cache_spk_audio(path)` / `cache_prompt_audio(path, text)` — 预缓存音频，减少首次延迟
- `infer_async(...)` / `infer_stream_async(...)` / `infer_batched_async(...)` — 异步版本

</details>

## 🌐 WebUI 可视化界面（不用写代码）

不想写代码？网页版适合你：浏览器打开，上传音频、输入文字、点按钮，就能合成（支持单模型和多角色两种模式，多角色还支持 `<speaker:角色名>文本</speaker:角色名>` 标签混用）。

```bash
cd WebUI
pip install -r requirements.txt   # 自动以 -e .. 安装本仓库 gsv_tts（MultiSpeakerTTS 未发布至 PyPI）
python web.py                     # 可选参数: --port 9881 / --use_asr / --models_dir ...
```

启动后浏览器会自动打开 `http://127.0.0.1:9881`。

## 🔌 API 服务接口（给开发者）

```bash
cd API
pip install -r requirements.txt
```

- 核心文档：[API 详细指南](API/README.md)、[Personal API 文档](API/PERSONAL_API.md)
- 服务入口：`API/personal_api.py`（含 MultiSpeaker 端点）、`API/realtime_api.py`（实时流式）

> 提供 6 个多角色管理端点（`/multi-speaker/init`、`/add`、`/remove`、`/list`、`/infer`、`/batch`）+ `/multi-speaker/stream` 流式端点，方便程序集成。

<details>
<summary><strong>📁 项目结构（给开发者）</strong></summary>

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

</details>

<details>
<summary><strong>🛠️ 开发与调试（给开发者）</strong></summary>

**测试**（无需 pytest，直接运行）：

```bash
# 自洽性测试：验证共享骨干输出与全量模型一致（MCD 指标）
python tests/test_sovits_sharing.py

# 使用真实微调模型评估
python tests/test_sovits_sharing.py --speaker-gpt path/to/gpt.ckpt --speaker-sovits path/to/sovits.pth
```

> MCD 计算需要 `librosa`，未安装时自动跳过并警告。

**基准测试**（自动发现仓库内模型，或手动指定）：

```bash
python benchmarks/bench_multi_speaker.py                          # 自动扫描仓库内 .ckpt/.pth 配对
python benchmarks/bench_multi_speaker.py --models-dir models      # 指定扫描目录
python benchmarks/bench_multi_speaker.py --gpt a.ckpt --sovits b.pth   # 手动指定（可重复）
```

**模型格式与兼容性**：

- 支持传统 `.ckpt` / `.pth` 检查点（加载时会提示安全警告，属正常现象），也支持更安全的 **safetensors 目录格式**（`hps.json` + `model.safetensors`），可用 `tts.to_safetensors(path)` 转换。
- SoVITS 版本自动识别：看文件头（`01`=v2、`05`=v2Pro、`06`=v2ProPlus），识别不了就按 v2 处理并警告。
- 设备差异：Mac/CPU 环境自动用 float32 并禁用部分缓存；CPU 下中文增强用 INT8 量化模型，显卡下用原版模型。

**常见坑位**：

- 不要删除 `Loader.py` 顶部的 `sys.modules['utils']` monkey-patch（旧版模型加载依赖它）。
- `gpt_cache` / `sovits_cache` 参数别乱改，配置不当会触发 CUDA graph 报错。
- 推理由 `_infer_lock` 串行化并自动清理显存；模型是懒加载（第一次推理才加载）。

</details>

## ❓ 常见问题 (FAQ)

**Q1：`pip install gsv-tts-lite` 装完没有多角色功能？**
多说话人功能还没发布到 PyPI，必须从本仓库安装：`git clone` + `pip install -e .`（见[快速开始](#-快速开始-quick-start)）。

**Q2：第一次运行下载模型很慢 / 失败？**
用 `GSV_MIRROR` 环境变量强制换镜像（国内建议 `modelscope`），或者手动把模型文件放到 `~/.cache/gsv`（文件清单见[首次运行](#首次运行自动下载模型只需要一次)）。

**Q3：某个角色特别占内存，好像没用上"共享骨干"？**
这个角色的模型和公共骨干版本不兼容（比如 v2 和 v2ProPlus），程序自动降级为传统方式加载了。日志里会提示。想省内存就把所有角色模型统一成 v2ProPlus 架构。

**Q4：换了参考音频内容，但合成结果没变化？**
音频缓存是按路径记的：同一个路径换了新内容，程序还以为是旧文件。换内容后删除缓存（`del_spk_audio` / `del_prompt_audio`）或换个文件名。

**Q5：报 CUDA graph 相关错误？**
多为 `gpt_cache` / `sovits_cache` 参数被改过，恢复默认值即可；Mac/CPU 环境不需要这两个参数。

**Q6：加载模型时有 `weights_only=False` 安全警告？**
这是为了兼容旧版模型文件的刻意行为。只加载你信任的模型文件，或转成 safetensors 格式（`tts.to_safetensors`）消除风险。

<details>
<summary><strong>📖 术语速查（看不懂的名词都在这）</strong></summary>

| 名词 | 大白话解释 |
| :--- | :--- |
| **GPT 模型** | 负责"想说什么、怎么说"的模型（文字 → 语义） |
| **SoVITS 模型** | 负责"把语义变成声音"的模型（语义 → 音频） |
| **骨干（Backbone）** | 所有角色共用的那 1 套模型 |
| **角色权重（Weights）** | 模型里"学到的东西"，每个角色的专属小调整包 |
| **音色参考音频** | 告诉程序"用谁的声音"（几秒的人声即可） |
| **风格参考音频** | 告诉程序"用什么语气、情绪"（可选） |
| **显存（VRAM）** | 显卡上的内存，越大能跑的模型越多 |
| **safetensors** | 一种更安全的模型文件格式（替代 .pth/.ckpt） |
| **v2 / v2Pro / v2ProPlus** | SoVITS 模型的三个版本，越新效果越好 |
| **微调（Fine-tune）** | 用某个人的声音数据训练出的专属角色模型 |
| **音素（Phoneme）** | 语言的最小发音单位，比如拼音的声母韵母 |
| **BERT** | 提升中文理解效果的语言模型 |
| **RTF** | 实时率：合成 1 秒音频需要的时间，小于 1 表示比实时快 |

</details>

<details>
<summary><strong>⚡ Flash Attention（进阶加速，可选）</strong></summary>

追求更低延迟、更高吞吐量时可以启用 Flash Attention 加速：

- 🐧 **Linux**：[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)（源码编译）
- 🪟 **Windows**：[lldacing/flash-attention-windows-wheel](https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main)（预编译包）

安装后在代码里设置 `use_flash_attn=True` 即可。

</details>

## 致谢 (Credits)

特别感谢以下项目：
- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [chinokikiss/GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite)
