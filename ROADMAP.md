# 功能路线图（Roadmap）

> 基于对仓库现状的完整审查整理的功能建议与可行性分析。
> 优先级：P0 = 立即做，P1 = 近期做，P2 = 按需做，P3 = 观望。

## 一、核心推理功能

### 1. 多角色混合流式合成 — P0｜可行性：高｜工作量：中

- **现状**：WebUI 流式模式明确不支持 `<speaker:>` 混合标签（`web.py` 中提示"混合角色标签暂不支持流式合成"），只能逐角色流式
- **方案**：按标签顺序解析角色段落，逐段调用 `MultiSpeakerTTS.infer_stream` 串接成统一 SSE/音频流（每段角色切换时重新注入权重）
- **依赖**：无（核心 `infer_stream` 已支持按角色路由）

### 2. 自动角色匹配（声纹路由）— P2｜可行性：高｜工作量：小

- **方案**：利用现有 `verify_speaker` 声纹相似度，新增 `MultiSpeakerTTS.infer_auto(speaker_audio, text)`：自动比对已加载角色，选最相似者合成；或暴露 `/multi-speaker/auto-infer` API 端点
- **应用**：电话客服、直播互动——无需指定角色名，按声音自动路由
- **注意**：需加载 SV 模型（`always_load_sv=True`），额外占用约 200MB

### 3. 剧本对话模式 — P1｜可行性：高｜工作量：小

- **方案**：`multi_tts.infer_script("alice: 你好\nbob: 嗨")` 或 WebUI 增加"对话剧本"输入框，自动解析角色行并批量合成，返回拼接音频 + 带角色名的字幕时间轴
- **依赖**：`infer_batched` 分组逻辑现成

### 4. 多角色配置导出/导入（preset 扩展）— P2｜可行性：高｜工作量：小

- **现状**：WebUI 的 preset 只保存单模型的 prompt/音色参考，多角色配置（SpeakerConfig 列表）无法保存
- **方案**：新增"角色组"预设（JSON 序列化角色列表），一键恢复整个多角色环境

## 二、工程质量

### 5. 测试体系 — P1｜可行性：高｜工作量：中

- **现状**：仅 1 个自洽性脚本（`tests/test_sovits_sharing.py`），无 pytest 配置、无单元测试
- **方案**：
  - 模型配对逻辑单测（benchmark 的 `discover_models`）
  - 权重提取单测（合成小权重文件）
  - API 端点测试（FastAPI TestClient，mock TTS）
  - WebUI 纯函数测试（`multi_add_speaker` 等）
- **收益**：防回归（WebUI 曾一次性修复 10 个 bug，无一有测试保护）

### 6. CI（GitHub Actions）— P1｜可行性：高｜工作量：小-中

- **方案**：push/PR 触发：`ruff check` + `py_compile` + 纯逻辑单测（不跑推理，避免下载模型）；可选 nightly 跑真实推理测试
- **前置**：仓库无 lint 配置，先引入 ruff

### 7. 模型下载容错 — P0｜可行性：高｜工作量：中

- **现状**：`check_pretrained_models` 下载 `pretrained_models5/6.zip` 时**没有镜像 fallback 链**（只有 `ensure_default_models` 有兜底）；下载失败即删文件重来，无断点续传
- **方案**：统一走 `ensure_default_models` 的 fallback 模式 + 下载完整性校验 + 断点续传（HTTP `Range`）
- **收益**：直接提升国内用户首次运行体验

### 8. 音频缓存自动失效 — P1｜可行性：高｜工作量：小

- **现状**：`spk_audio_cache` / `prompt_audio_cache` 按路径 keyed，同一路径换内容会命中旧缓存（README FAQ 已记录此坑）
- **方案**：缓存键加入文件 mtime + size（或轻量 hash），内容变化自动重算

### 9. Benchmark 报告化 — P2｜可行性：高｜工作量：小

- **方案**：`bench_multi_speaker.py` 增加 `--output report.json`（各角色延迟/RSS/RTF），可选与历史基线 diff

## 三、部署与体验

### 10. Docker 镜像 — P2｜可行性：高｜工作量：小-中

- **方案**：`Dockerfile`（CUDA 运行时 + CPU 版）+ `docker-compose` 一键起 API 服务；模型目录挂载卷
- **收益**：服务端部署门槛骤降

### 11. CLI 入口 — P2｜可行性：高｜工作量：中

- **方案**：`python -m gsv_tts` 子命令：`infer`（文本→wav）、`models`（列出/校验/下载模型）、`convert`（.pth→safetensors）、`bench`（跑基准）

### 12. WebUI 角色批量扫描添加 — P1｜可行性：高｜工作量：小

- **现状**：多角色模式需手填 GPT/SoVITS 路径
- **方案**：把 benchmark 的 `discover_models` 配对逻辑提取到 `gsv_tts` 公共模块，WebUI 加"扫描目录自动发现角色"按钮（一键填充/批量添加）

### 13. 字幕导出（SRT/ASS）— P1｜可行性：高｜工作量：小

- **方案**：字级时间戳已有 → `AudioClip.export_subtitles("out.srt")` + WebUI 下载按钮
- **收益**：视频字幕制作场景直接可用

### 14. 输出格式选择 — P2｜可行性：高｜工作量：小

- **方案**：WebUI/API 支持 wav/ogg 输出（`personal_api.py` 的 `pack_ogg` 已有实现，WebUI 可复用）

### 15. ONNX 导出（CPU 加速）— P3｜可行性：中｜工作量：大

- **现状**：CPU 下 BERT 已走 INT8 ONNX，但 GPT/SoVITS 仍是 PyTorch
- **方案**：GPT 语义模型导出 ONNX + ORT 推理后端
- **风险**：SoVITS 含 VQ/流匹配等自定义算子，导出难；建议只导出 GPT 部分

### 16. 多 GPU 路由 — P3｜可行性：低｜工作量：大

- **方案**：角色按显存均衡分配到多卡（`device_map` per speaker）
- **依赖**：需重构模型驻留管理——不建议近期做

## 优先级推荐

| 优先级 | 项目 | 理由 |
|:---:|---|---|
| P0 | 模型下载容错（#7） | 真实缺口，影响所有国内用户首次体验 |
| P0 | 多角色混合流式（#1） | WebUI 已明示不支持，功能空白 |
| P1 | 测试体系（#5）+ CI（#6） | 修复过 10 个 bug，无一有测试保护 |
| P1 | 音频缓存失效（#8）、字幕导出（#13） | 小改动高感知 |
| P1 | WebUI 角色扫描（#12）、剧本对话（#3） | 复用现有代码，立即可用 |
| P2 | 自动角色匹配（#2）、Docker（#10）、CLI（#11） | 中工作量，按需做 |
| P3 | ONNX（#15）、多 GPU（#16） | 投入大，观望 |

## 状态追踪

- [x] #1 多角色混合流式合成（`WebUI/web.py`，`split_speaker_text`）
- [x] #2 自动角色匹配（声纹路由）(`MultiSpeakerTTS.infer_auto`)
- [x] #3 剧本对话模式（`MultiSpeakerTTS.infer_script` / `parse_script`）
- [x] #4 多角色配置导出/导入（WebUI 角色组预设 JSON）
- [x] #5 测试体系（pytest，3 个测试文件 22 例，见 `tests/`）
- [x] #6 CI（`.github/workflows/ci.yml`，Python 3.10-3.12）
- [x] #7 模型下载容错（`Download._download_zip_with_fallback` 镜像链）
- [x] #8 音频缓存自动失效（mtime+size 指纹）
- [x] #9 Benchmark 报告化（`--output report.json`）
- [x] #10 Docker 镜像 + compose（CPU 版）
- [x] #11 CLI 入口（`python -m gsv_tts`：infer/multi/models/convert）
- [x] #12 WebUI 角色批量扫描添加（`gsv_tts/model_discovery.py`）
- [x] #13 字幕导出（`AudioClip.export_subtitles` SRT/ASS + WebUI 自动 SRT）
- [x] #14 输出格式选择（WebUI wav/ogg）
- [ ] #15 ONNX 导出（P3，观望）
- [ ] #16 多 GPU 路由（P3，观望）
