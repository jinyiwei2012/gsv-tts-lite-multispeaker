# GSV-TTS-Lite API 服务文档

本仓库提供三个 HTTP 服务 + 若干测试脚本，覆盖**单说话人**与**多说话人（MultiSpeakerTTS）**推理：

| 文件 | 端口 | 说明 |
| :--- | :---: | :--- |
| `fastapi_server_example.py` | 8000 | 示例服务：单说话人 + 多说话人全套端点，含外链音频与 ASR |
| `personal_api.py` | 9880 | 个人应用 API：兼容原版 GPT-SoVITS API 参数，支持流式/批量/模型热切换/多角色 |
| `realtime_api.py` | 8080 | 实时音视频服务（aiohttp） |
| `test_async_performance.py` | - | 同步 vs 异步性能对比测试 |
| `test_url_audio.py` | - | 外链音频功能测试 |
| `test_realtime_api.py` | - | 实时 API 测试 |

> [!NOTE]
> 多说话人功能未发布到 PyPI，`API/requirements.txt` 已通过 `-e ..` 自动安装本仓库的 `gsv_tts` 包，无需额外操作。

## 🚀 快速开始

```bash
# 1. 安装依赖（自动从仓库根安装本地 gsv_tts）
cd API
pip install -r requirements.txt

# 2. 启动服务（任选其一）
python fastapi_server_example.py   # 示例服务，端口 8000
# python personal_api.py           # 个人 API，端口 9880（-p 可改）

# 3. 浏览器打开交互式 API 文档
# http://localhost:8000/docs   （personal_api 为 http://localhost:9880/docs）
```

> 首次运行会自动下载预训练模型（数 GB）到 `~/.cache/gsv`；`personal_api.py` 默认下载到 `API/models`，可用 `--models_dir` 指定。下载慢可用环境变量 `GSV_MIRROR=modelscope` 强制镜像。

## 🎯 多说话人端点（MultiSpeaker，核心）

两个 FastAPI 服务都提供同一套多角色管理端点（默认端口分别为 8000 / 9880）：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/multi-speaker/init` | 初始化多角色引擎（共享骨干） |
| POST | `/multi-speaker/add` | 添加角色 |
| POST | `/multi-speaker/remove` | 移除角色 |
| GET  | `/multi-speaker/list` | 列出已加载角色 |
| POST | `/multi-speaker/infer` | 单角色推理 |
| POST | `/multi-speaker/batch` | 多角色批量推理 |
| POST | `/multi-speaker/stream` | 单角色流式推理 (SSE) |

**使用流程**（以 8000 端口为例）：

**1. 初始化引擎**（可选，不调用则首次添加角色时自动初始化）

```bash
curl -X POST "http://localhost:8000/multi-speaker/init" \
  -H "Content-Type: application/json" \
  -d '{"use_bert": true, "use_flash_attn": false}'
```

> `base_gpt_path` / `base_sovits_path` 可指定共享骨干模型路径，默认使用 `~/.cache/gsv` 下的默认模型。

**2. 添加角色**（`speaker_audio` / `prompt_audio` 支持本地路径或 URL）

```bash
curl -X POST "http://localhost:8000/multi-speaker/add" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "alice",
    "gpt_model_path": "models/alice_gpt.ckpt",
    "sovits_model_path": "models/alice_sovits.pth",
    "speaker_audio": "audio/alice_ref.wav",
    "prompt_audio": "audio/alice_prompt.ogg",
    "prompt_text": "こんにちは、アリスです。"
  }'
```

> 响应返回 `mode`：`shared`（共享骨干）或 `full_model_degraded`（架构不兼容时自动降级为完整模型，不影响其他角色）。

**3. 查看已加载角色**

```bash
curl "http://localhost:8000/multi-speaker/list"
```

```json
{
  "initialized": true,
  "speakers": [
    {"name": "alice", "mode": "shared", "gpt_keys": 25, "sovits_keys": 37}
  ]
}
```

**4. 单角色推理**（支持语言参数与**按次 prompt 覆盖**）

```bash
curl -X POST "http://localhost:8000/multi-speaker/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "speaker": "alice",
    "text": "こんにちは！",
    "text_language": "ja",
    "prompt_language": "ja",
    "prompt_audio_path": "audio/other_style.ogg",
    "prompt_audio_text": "別のスタイルのテキスト。"
  }'
```

> `text_language` / `prompt_language` 支持 `"auto"` / `"ja"` / `"zh"` / `"en"`（默认 `"auto"`）。
> `prompt_audio_path` / `prompt_audio_text` 可选：不传则使用添加角色时配置的风格参考。

**5. 多角色批量推理**（相同角色自动 GPU 并行，每条可独立指定语言与 prompt 覆盖）

```bash
curl -X POST "http://localhost:8000/multi-speaker/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "speaker_texts": [
      {"speaker": "alice", "text": "こんにちは", "text_language": "ja"},
      {"speaker": "bob",   "text": "你好", "text_language": "zh"}
    ]
  }'
```

**6. 单角色流式推理 (SSE)**（Token 级流式输出，低延迟实时反馈）

```bash
curl -N -X POST "http://localhost:8000/multi-speaker/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "speaker": "alice",
    "text": "こんにちは！長いテキストでも低遅延で読み上げます。",
    "text_language": "ja",
    "stream_chunk": 25,
    "overlap_len": 5
  }'
```

SSE 事件格式：

```
event: audio
data: {"audio": "<base64>", "sample_rate": 32000, "duration": 0.5, "text": "..."}

event: done
data: {"total_duration": 5.2}

event: error
data: {"error": "错误信息"}
```

**自动兼容性**：加载角色时自动校验角色模型与骨干的架构参数（`vocab_size`、`n_layer`、`gin_channels`、`upsample_initial_channel` 等），不兼容的角色自动降级为完整模型加载，不中断其他角色。

## 📖 服务一：fastapi_server_example.py（示例服务，端口 8000）

**单说话人端点**：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/tts/single` | 单句合成 |
| POST | `/tts/batch` | 批量合成 |
| GET  | `/audio/{filename}` | 下载生成的音频 |
| POST | `/multi-speaker/*` | 多角色端点（见上） |

**单句合成**：

```bash
curl -X POST "http://localhost:8000/tts/single" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是测试。",
    "speaker_audio": "examples/laffey.mp3",
    "prompt_audio": "examples/AnAn.ogg",
    "prompt_text": "ちが……ちがう。レイア、貴様は間違っている。",
    "text_language": "auto",
    "prompt_language": "auto",
    "top_k": 5,
    "top_p": 0.9,
    "temperature": 1.0,
    "repetition_penalty": 1.35,
    "noise_scale": 0.5,
    "speed": 1.0
  }'
```

```json
{
  "success": true,
  "audio_len": 1.72,
  "filename": "tts_06a1a5fc.wav",
  "prompt_text_used": "ちが……ちがう。レイア、貴様は間違っている。"
}
```

> `prompt_text` 可选：不提供时自动用 ASR 识别。`speaker_audio` / `prompt_audio` 支持本地路径或 HTTP(S) URL。

**批量合成**：

```bash
curl -X POST "http://localhost:8000/tts/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["第一句", "第二句"],
    "speaker_audio": "examples/laffey.mp3",
    "prompt_audio": "examples/AnAn.ogg",
    "prompt_text": "ちが……ちがう。レイア、貴様は間違っている。",
    "text_languages": "auto",
    "prompt_languages": "auto"
  }'
```

> `text_languages` / `prompt_languages` 支持单个字符串（全部相同）或逐句列表（`["ja", "zh"]`）。

**下载音频**：`GET /audio/tts_06a1a5fc.wav`

## 📖 服务二：personal_api.py（个人应用 API，端口 9880）

> 详细文档见 [PERSONAL_API.md](PERSONAL_API.md)。启动参数：`-p/--port`（默认 9880）、`--models_dir`（默认 `models`）、`--use_asr`。

**端点一览**：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET/POST | `/tts` | 兼容原版 GPT-SoVITS API 参数（v2 格式） |
| POST | `/tts/stream` | 流式推理 (SSE) |
| POST | `/tts/batched` | 批量推理 |
| GET | `/set_gpt_weights` | 热切换 GPT 模型 |
| GET | `/set_sovits_weights` | 热切换 SoVITS 模型 |
| GET | `/audio/{filename}` | 下载生成的音频 |
| POST | `/multi-speaker/*` | 多角色端点（见上） |

**流式推理 `/tts/stream`**：SSE 实时推送，`stream_mode` 支持 `token`（低延迟，适合实时对话）与 `sentence`（更连贯，适合长文本）。

**批量推理 `/tts/batched`**：一次请求生成多个音频，支持 `return_subtitles` 返回逐字时间戳。

**模型热切换**：

```bash
curl "http://localhost:9880/set_gpt_weights?weights_path=models/new_gpt.ckpt"
curl "http://localhost:9880/set_sovits_weights?weights_path=models/new_sovits.pth"
```

## 📖 服务三：realtime_api.py（实时音视频，端口 8080）

基于 aiohttp 的实时音频服务，适合实时通话类场景（配合 WebRTC 等使用）。

## 🌐 外链音频 & ASR

- **外链音频**：所有 `speaker_audio` / `prompt_audio` 参数支持 HTTP(S) URL，服务自动下载到临时文件（支持 mp3/wav/ogg/flac）。
- **ASR 自动识别**：`prompt_text` 不提供时自动用 ASR（Qwen3-ASR）识别参考音频文本。
  - `fastapi_server_example.py`：环境变量 `USE_ASR=true`（默认）/ `USE_ASR=false`
  - `personal_api.py`：启动参数 `--use_asr`

## 🔍 常见问题

**Q：服务启动失败？**
检查端口是否被占用（`netstat -ano | findstr 8000`），或换端口：`personal_api.py -p 9881`；`fastapi_server_example.py` 需修改文件末尾的 `uvicorn.run(..., port=...)`。

**Q：首次请求很慢？**
首次运行会自动下载模型（数 GB）；模型懒加载，第一次推理会额外耗时（预热）。可先调用一次任意接口预热。

**Q：外链音频下载失败？**
确认 URL 可访问；`httpx` 默认超时 60 秒。

**Q：ASR 识别不准？**
手动提供 `prompt_text` 参数，或确保参考音频音质清晰。

**Q：`/multi-speaker/add` 返回 `full_model_degraded`？**
该角色模型与共享骨干架构不兼容，已自动降级为完整模型加载——功能正常，只是该角色不共享骨干、占用更多显存。建议将角色模型统一为 v2ProPlus 架构。
