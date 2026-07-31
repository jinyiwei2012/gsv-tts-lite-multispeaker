<div align="center">

> [!IMPORTANT]
> ### 🔀 MultiSpeaker 独立開発リポジトリ
> 本リポジトリは [GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite) のフォークに由来します：上流リポジトリに `multi-speaker-inference` ブランチは存在せず、フォーク内で作成された後に独立して本リポジトリとなり、**マルチスピーカー（MultiSpeakerTTS）共有骨格推論**の独立開発・最適化に特化しています。
>
> - **上流リポジトリ**：[chinokikiss/GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite)（PyPI では `gsv-tts-lite` として公開）
> - マルチスピーカー機能は**まだ PyPI に公開されていません**。本リポジトリが唯一のソースです。上流のバグ修正は cherry-pick / merge で同期できます

</div>

<div align="center">
  <h1>GSV-TTS-Lite · MultiSpeaker</h1>

  <p>
    GPT-SoVITS マルチスピーカー共有骨格推論エンジン（MultiSpeakerTTS）
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

## プロジェクトについて (About)

本リポジトリは **GSV-TTS-Lite のマルチスピーカー（MultiSpeaker）独立開発リポジトリ**であり、コア機能は **MultiSpeakerTTS マルチスピーカー共有骨格推論**です：

従来方式では話者ごとに完全なモデルをロードするため、VRAM/メモリは話者数に比例して増加します。本リポジトリでは**1 セットの共有 GPT+SoVITS 骨格**のみをロードし、各話者はわずか ~5-15% の軽量な専用重み（約 25 GPT keys + 37 SoVITS keys）を注入するだけで、話者名に応じて**ゼロコストで動的切り替え**できます——マルチスピーカー環境では VRAM/メモリを **40%~75%** 節約でき、**話者が多ければ多いほど効果的**です。

単一話者推論（`TTS`）の全機能も備えています：Token レベルストリーミング、バッチ並列、文字単位タイムスタンプ、ゼロショット音色変換、声紋認識、そして **WebUI** と **API** サービスに対応。

対応言語：**中国語、日本語、英語**。対応モデル：**V2**、**V2Pro**、**V2ProPlus**。

## ✨ 機能一覧 (Features)

- 🎭 **マルチスピーカー共有骨格**：1 セットの GPT+SoVITS 骨格で 10+ 話者に対応。各話者の専用重みは約 25 GPT keys + 37 SoVITS keys のみ
- 🔀 **ゼロコスト話者切替**：話者重みを必要に応じて動的注入。切替時の追加推論コストなし
- 🔌 **自動互換性チェック**：骨格とアーキテクチャが不一致の話者は自動的に全量ロードへデグレード。他の話者には影響なし
- ⚡ **全推論モード**：単一話者 `infer`、Token レベルストリーミング `infer_stream`、同一話者 GPU 並列 `infer_batched`
- 🎵 **音色とスタイルの分離**：音色（Speaker）とスタイル（Prompt）を独立制御。呼び出しごとのスタイル上書きに対応
- 🖥️ **WebUI / API 全対応**：`<speaker:名前>` タグによる混在合成、6+1 の MultiSpeaker API エンドポイント
- ⏱️ **文字単位タイムスタンプ**：字幕同期に対応した文字単位のタイムスタンプ返却
- 🌐 **3 言語対応**：中日英の自動言語検出（`auto` / `ja` / `zh` / `en`）

## 🎭 MultiSpeakerTTS 共有骨格推論（コア機能）

### 動作原理

従来の「話者ごとにモデルをフルロード」方式とは異なり、`MultiSpeakerTTS` はまず**1 セットの共有 GPT+SoVITS 骨格**をロードし、各話者のファインチューニング差分重み（約 25 GPT keys + 37 SoVITS keys）を個別に保存します。推論時は話者名に応じて対応する重みを**動的に注入**します。

そのため、メモリ/VRAM 使用量 ≈ **1 骨格 + 1 話者分の重み**であり、「話者数 × フルモデル」ではありません。GPU 環境では重み注入による VRAM オーバーヘッドはほぼゼロで、効果は話者数の増加に比例して拡大します。

### 実測ベンチマーク（共有骨格 vs 全量ロード）

> [!NOTE]
> **テスト環境**：CPU 参考環境（GPU なし）。実在のファインチューニングモデル（CyreneV3.7 / shouanren / LuoTianyi、v2ProPlus 互換アーキテクチャ）を使用、短文推論の平均値。

| 指標 | 共有骨格 | 全量ロード | 説明 |
| :--- | :---: | :---: | :--- |
| 話者あたり平均推論遅延 | 0.7~0.9s | 0.8~0.9s | ⚖️ 性能損失なし |
| ピークメモリ (RAM) | **2.77 GB** | 4.65 GB | 💾 **-40%**（CPU 実測） |
| 3話者初期化時間 | 30.0s | 16.2s | 初回のみの重み抽出。以降の話者切替はゼロコスト |

> [!IMPORTANT]
> **アーキテクチャ互換性検証**（実モデル）：
> - ✅ CyreneV3.7、shouanren、LuoTianyi（Agent-LuoTianyi プロジェクトのモデル）→ 共有骨格モード
> - ⚠️ aimisi（v2 アーキテクチャ、`upsample_initial_channel=512` vs base `768`）→ **自動的に完全モデルロードへデグレード**。他の話者には影響なし
>
> メモリ節約は共有話者数の増加に伴い拡大（2話者 -17% → 3話者 -40%）。GPU 環境では CPU 実測値よりもはるかに大きな VRAM 節約が期待できます（重み注入はメモリ帯域に依存しないため）。

| 方式 | 1 キャラ | 3 キャラ | 5 キャラ | 10 キャラ |
|------|--------|--------|--------|---------|
| 完全ロード | ~800MB | ~2.4GB | ~4.0GB | ~8.0GB |
| **MultiSpeakerTTS** | ~800MB | **~1.2GB** | **~1.4GB** | **~2.0GB** |
| VRAM 節約 | — | **51%** | **65%** | **75%** |

### 使用方法

```python
from gsv_tts import MultiSpeakerTTS, SpeakerConfig

# 複数のキャラクターを定義（モデルパスは safetensors ディレクトリ形式にも対応）
speakers = [
    SpeakerConfig(
        name="alice",
        gpt_model_path="models/alice_gpt.ckpt",
        sovits_model_path="models/alice_sovits.pth",
        spk_audio_path="audio/alice_ref.wav",
        prompt_audio_path="audio/alice_prompt.ogg",  # 省略可、デフォルトでは spk_audio_path を使用
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

# すべてのキャラクターを一度にロード（共有骨格 + キャラクター専用の重み）
tts = MultiSpeakerTTS(speakers=speakers, use_bert=True)

# 単一キャラクター推論 — キャラクター名で自動ルーティング。言語パラメータと呼び出しごとの prompt 上書きに対応
audio = tts.infer(
    "alice",
    "今日も頑張りましょう！",
    text_language="ja",       # "auto" / "ja" / "zh" / "en"
    prompt_language="ja",     # "auto" / "ja" / "zh" / "en"
    # prompt_audio_path="other_style.ogg",   # 任意：スタイル参照オーディオをこの呼び出しでのみ上書き
    # prompt_audio_text="別のスタイルのテキスト。",  # 上書き時は対応テキストも必須
)
audio.play()

# ストリーミング推論 — Token レベルのストリーミング出力、低遅延のリアルタイムフィードバック
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

# バッチ推論 — 同じキャラクターは自動的に GPU 並列処理、文ごとの言語指定にも対応
audios = tts.infer_batched(
    [
        ("alice", "こんにちは"),
        ("alice", "お元気ですか"),
        ("bob",   "よろしくお願いします"),
    ],
    text_languages=["ja", "ja", "ja"],  # または単に "auto" を渡す
)

# 実行時管理：話者の追加・削除を動的に実行、再起動不要
tts.add_speaker(SpeakerConfig(name="charlie", ...))
tts.remove_speaker("bob")
print(tts.speaker_names)  # ["alice", "charlie"]
```

> [!TIP]
> **自動互換性チェック**：ロード時にアーキテクチャパラメータ（`vocab_size`、`n_layer`、`gin_channels`、`upsample_initial_channel` など）を自動検証。互換性のない話者は**自動的に完全モデルロードへデグレード**され、ユーザーの介入は不要です。

## 🚀 クイックスタート (Quick Start)

### 環境準備

- Python **>= 3.10**（仮想環境を推奨）
- 推論バックエンド：**CUDA**、**MPS (Apple Silicon)**、**CPU**

```bash
# NVIDIA GPU (CUDA 12.8) の場合
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Apple Silicon (MPS) または Linux/Windows (CPU のみ) の場合
pip install torch torchvision torchaudio
```

### GSV-TTS-Lite のインストール

> [!WARNING]
> **マルチスピーカー（MultiSpeakerTTS）機能は PyPI に未公開です**。PyPI の `gsv-tts-lite` パッケージは単一話者推論のみ対応しています。マルチスピーカー共有骨格を使うには、必ず本リポジトリからインストールしてください：

```bash
git clone https://github.com/jinyiwei2012/gsv-tts-lite-multispeaker.git
cd gsv-tts-lite-multispeaker
pip install -e .
```

### 初回実行：モデルの自動ダウンロード

> [!NOTE]
> 初回に `TTS` / `MultiSpeakerTTS` を生成すると、必要な事前学習済みモデル（数 GB）がローカルキャッシュディレクトリ **`~/.cache/gsv`** に自動ダウンロードされます（`TTS(models_dir=...)` で変更可能）：
> - GPT モデル：`s1v3.ckpt`；SoVITS モデル：`s2Gv2ProPlus.pth`
> - 事前学習済みコンポーネント：CNHubert、G2P、声紋モデル、CNRoBERTa（BERT）
>
> ダウンロード元はレイテンシにより自動選択：**ModelScope → hf-mirror → HuggingFace**。中国国内では通常 ModelScope が自動選択されます。環境変数での強制指定も可能です：
>
> ```bash
> # 任意：ダウンロードミラーを強制指定 modelscope / huggingface / hf-mirror
> export GSV_MIRROR=modelscope
> ```

### 単一話者基本推論

> [!NOTE]
> 単一話者だけでよい場合は `TTS` を直接使用できます（本リポジトリは単一話者推論の全機能も提供しています。詳細は[単一話者推論](#-単一話者推論-tts)セクション参照）。

```python
from gsv_tts import TTS

tts = TTS(use_bert=True)
# tts = TTS(use_flash_attn=True) # Flash Attention をインストール済みの場合、この設定を推奨します

# GPT / SoVITS モデルの重みを指定されたパスからメモリにロードします。ここではデフォルトモデルをロードします。
tts.load_gpt_model()
tts.load_sovits_model()

# infer は最もシンプルで原始的な推論方式であり、短文の推論にのみ適しています。通常、infer の代わりに infer_batched を使用することが推奨されます。
audio = tts.infer(
    spk_audio_path="examples\laffey.mp3", # 音色参照オーディオ
    prompt_audio_path="examples\AnAn.ogg", # スタイル参照オーディオ
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。", # スタイル参照オーディオに対応するテキスト
    text="へぇー、ここまでしてくれるんですね。", # 生成対象テキスト
    text_language="auto", # 対象テキストの言語："auto" / "ja" / "zh" / "en"、デフォルトは自動検出
    prompt_language="auto", # 参照オーディオテキストの言語："auto" / "ja" / "zh" / "en"、デフォルトは自動検出
)

audio.play()
tts.audio_queue.wait()
```

## 📖 単一話者推論 (TTS)

> [!NOTE]
> 以下は `TTS` 単一話者エンジンの上級用法です。MultiSpeakerTTS の `infer` / `infer_stream` / `infer_batched` も同等の機能を備えています（話者名でルーティング）。

<details>
<summary><strong>1. ストリーミング推論 / 字幕同期</strong></summary>

`infer_stream` は Token レベルのストリーミング出力を実装し、初字遅延を大幅に低減します。`infer`、`infer_stream`、`infer_batched`、`infer_vc` はすべて文字単位のタイムスタンプ返却に対応しています。

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
    text_language="auto", # 対象テキストの言語："auto" / "ja" / "zh" / "en"
    prompt_language="auto", # 参照オーディオテキストの言語："auto" / "ja" / "zh" / "en"
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
<summary><strong>2. バッチ推論</strong></summary>

`infer_batched` は長テキストおよび多文合成シーン向けに最適化されており、同一バッチ内で異なる文に対して異なる参照オーディオを指定できます。

```python
from gsv_tts import TTS

# gpt_cache: GPT モデルの CUDA グラフ用静的キャッシュ設定。タプルのリスト [(batch_size, sequence_length), ...]。
# 注意：設定した最大 batch_size がバッチ処理の最大スループットを決定し、バッチ内の最大 sequence_length が 1 リクエストあたりの最大生成長を決定します。
tts = TTS(use_bert=True)

audios = tts.infer_batched(
    spk_audio_paths="examples\laffey.mp3",
    prompt_audio_paths="examples\AnAn.ogg",
    prompt_audio_texts="ちが……ちがう。レイア、貴様は間違っている。",
    texts=["へぇー、ここまでしてくれるんですね。", "The old map crinkled in Leo's trembling hands."],
    text_languages="auto", # 対象テキストの言語。str または文ごとの list[str] に対応："auto" / "ja" / "zh" / "en"
    prompt_languages="auto", # 参照オーディオテキストの言語。str または文ごとの list[str] に対応："auto" / "ja" / "zh" / "en"
    bert_batch_size=20,
    sovits_batch_size=10,
)

for i, audio in enumerate(audios):
    audio.save(f"audio{i}.wav")
```

</details>

<details>
<summary><strong>3. 音色変換 / 声紋認識</strong></summary>

```python
from gsv_tts import TTS

# ゼロショット音色変換（変声）
tts = TTS(use_bert=True, always_load_cnhubert=True)
audio = tts.infer_vc(
    spk_audio_path="examples\laffey.mp3",
    prompt_audio_path="examples\AnAn.ogg",
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
)
audio.play()

# 声紋認識：2 つのオーディオが同一話者かどうかを判定
tts2 = TTS(use_bert=True, always_load_sv=True)
similarity = tts2.verify_speaker("examples\laffey.mp3", "examples\AnAn.ogg")
print("声紋類似度：", similarity)
```

</details>

<details>
<summary><strong>4. その他の関数インターフェース</strong></summary>

#### モデル管理

- `init_language_module(languages)` — 必要な言語処理モジュールを事前にロード
- `load_gpt_model(model_paths)` / `load_sovits_model(model_paths)` — モデル重みを指定パスからメモリにロード
- `unload_gpt_model(model_paths)` / `unload_sovits_model(model_paths)` — リソース解放のためメモリからモデルをアンロード
- `get_gpt_list()` / `get_sovits_list()` — 現在ロードされているモデルのリストを取得
- `to_safetensors(checkpoint_path)` — PyTorch 形式の重みファイル（.pth / .ckpt）を safetensors ディレクトリ形式に変換

#### オーディオキャッシュ管理

- `cache_spk_audio(spk_audio_paths)` — 音色参照オーディオデータを前処理しキャッシュ
- `cache_prompt_audio(prompt_audio_paths, prompt_audio_texts, prompt_audio_languages)` — スタイル参照オーディオデータを前処理しキャッシュ
- `del_spk_audio(spk_audio_paths)` / `del_prompt_audio(prompt_audio_paths)` — キャッシュからオーディオデータを削除
- `get_spk_audio_list()` / `get_prompt_audio_list()` — キャッシュ内のオーディオデータリストを取得

#### 非同期呼び出し

- `infer_async(...)` — `infer` メソッドの非同期バージョン
- `infer_stream_async(...)` — `infer_stream` メソッドの非同期バージョン
- `infer_batched_async(...)` — `infer_batched` メソッドの非同期バージョン

</details>

## 🌐 WebUI 可視化インターフェース

```bash
cd WebUI
pip install -r requirements.txt   # -e .. で本リポジトリの gsv_tts を自動インストール（MultiSpeakerTTS は PyPI 未公開）
python web.py                     # オプション: --port 9881 / --use_asr / --models_dir ...
```

> [!TIP]
> WebUI は**単一モデル / マルチスピーカー**の 2 つの推論モードをワンクリックで切替可能。マルチスピーカーモードでは `<speaker:名前>テキスト</speaker:名前>` タグによる混在合成と自動 GPU バッチ並列処理をサポート。

## 🔌 API サービスインターフェース

```bash
cd API
pip install -r requirements.txt
```

- コアドキュメント：[API 詳細ガイド](API/README.md)、[Personal API ドキュメント](API/PERSONAL_API.md)
- サービスエントリポイント：`API/personal_api.py`（MultiSpeaker エンドポイント）、`API/realtime_api.py`（リアルタイムストリーミング）

> [!TIP]
> FastAPI サーバーには **6 つの MultiSpeaker エンドポイント**（`/multi-speaker/init`、`/multi-speaker/add`、`/multi-speaker/remove`、`/multi-speaker/list`、`/multi-speaker/infer`、`/multi-speaker/batch`）に加え、`/multi-speaker/stream` SSE エンドポイントがあり、マルチスピーカー管理とバッチ推論に対応しています。

## 📁 プロジェクト構成 (Project Structure)

```
gsv_tts/                  # コア Python パッケージ（pip install -e .）
├── MultiSpeaker.py       # 🎭 マルチスピーカー共有骨格推論エンジン：MultiSpeakerTTS（本リポジトリのコア）
├── SpeakerWeights.py     # 🎭 話者設定と重み抽出：SpeakerConfig / SpeakerWeights
├── TTS.py                # 単一話者推論エンジン：infer / infer_stream / infer_batched / infer_vc
├── Loader.py             # 重みのロードと SoVITS バージョン検出
├── Download.py           # モデル自動ダウンロード（複数ミラー選択）
├── TextProcessor.py      # テキスト → 音素 / BERT 特徴
├── Player.py             # 音声再生：AudioQueue / AudioClip
├── Config.py             # グローバル設定
└── GPT_SoVITS/           # モデルアーキテクチャ + テキスト処理
    ├── GPT/              # GPT セマンティックモデル（t2s）
    ├── SoVITS/           # SoVITS 音響モデル
    ├── G2P/              # 中日英の音素変換
    ├── Featurizer/       # CNHubert / CNRoBERTa 特徴抽出
    └── SV/               # 声紋モデル（ERes2Net）
tests/                    # テストスクリプト（MultiSpeaker 自己整合性テスト）
benchmarks/               # MultiSpeaker パフォーマンスベンチマーク
WebUI/                    # Gradio Web UI（独自 requirements.txt）
API/                      # FastAPI サーバー（独自 requirements.txt）
examples/                 # サンプル参照オーディオ（laffey.mp3 / AnAn.ogg）
```

## 🛠️ 開発とデバッグ (Development)

### テスト

```bash
# 自己整合性テスト：共有骨格の出力が全量モデルと一致することを検証（MCD 指標）
python tests/test_sovits_sharing.py

# 実モデルでの評価（オプション引数）
python tests/test_sovits_sharing.py --speaker-gpt path/to/speaker_gpt.ckpt --speaker-sovits path/to/speaker_sovits.pth
```

> [!NOTE]
> MCD の計算には `librosa` が必要です。未インストールの場合は警告とともに指標がスキップされます。pytest は不要で、スクリプトとして直接実行します。

### ベンチマーク

```bash
python benchmarks/bench_multi_speaker.py
```

> [!TIP]
> スクリプトはリポジトリ内（または `--models-dir` 指定ディレクトリ）の `.ckpt` / `.pth` モデルファイルを自動検出し、ファイル名プレフィックスでペアリングします。`--gpt` / `--sovits` でモデルペアを明示指定することも可能です（繰り返し指定可、safetensors ディレクトリにも対応）。モデルが見つからない場合はヒントを表示して終了します。

### モデル形式と互換性

- **重み形式**：従来の `.ckpt` / `.pth` チェックポイント（pickle 逆シリアライズでロード。起動時にセキュリティ警告が出力されます）に加え、より安全な **safetensors ディレクトリ形式**（`hps.json` + `model.safetensors`）に対応。`tts.to_safetensors(path)` で変換できます。
- **SoVITS バージョン検出**：ファイルヘッダバイト（`01`=v2、`05`=v2Pro、`06`=v2ProPlus）または既知の事前学習ファイルの MD5 で自動判定。認識できない場合は警告とともに v2 として処理します。
- **デバイス差異**：MPS/CPU 環境では `float32` が強制され、`sovits_cache` はクリアされます。CPU では BERT に INT8 量子化 ONNX モデル、GPU では PyTorch オリジナルモデルを使用します。

### ダウンロードミラー

ダウンロードミラーはレイテンシにより自動選択されます（ModelScope → hf-mirror → HuggingFace）。開発中のダウンロード問題に対処するには、環境変数でミラーを強制指定します：

```bash
# Windows (PowerShell)
$env:GSV_MIRROR = "modelscope"   # modelscope / huggingface / hf-mirror
# Linux/macOS
export GSV_MIRROR=modelscope
```

### よくある落とし穴

- `Loader.py` 先頭の `sys.modules['utils']` monkey-patch を変更すると、レガシーな GPT-SoVITS チェックポイントの逆シリアライズが失敗します——**削除しないでください**。
- `gpt_cache` / `sovits_cache` は CUDA グラフの静的キャッシュサイズを制御しており、設定を誤ると CUDA グラフエラーが発生します——デフォルト値を安易に変更しないでください。
- 推論は `_infer_lock` により直列化され、自動でキャッシュクリア（`_empty_cache`）が実行されます。モデルは遅延ロード（初回推論時のみ）です——VRAM 使用量のデバッグ時はこの点に注意してください。

## ❓ よくある質問 (FAQ)

**Q1：マルチスピーカー機能が PyPI からインストールできない？**
本リポジトリの MultiSpeakerTTS 機能はまだ PyPI に未公開です。本リポジトリから `pip install -e .` でインストールしてください。

**Q2：特定の話者の VRAM/メモリ使用量が異常に高く、共有骨格になっていない？**
その話者のモデルアーキテクチャがベースモデルと互換性がない（例：v2 の `upsample_initial_channel=512` vs base `768`）ため、自動的に全量ロードへデグレードされています。ロードログに表示されます。ベースアーキテクチャ（v2ProPlus）に一致するファインチューニングモデルの使用を推奨します。

**Q3：参照オーディオの内容を変更したのに、推論結果が変わらない？**
話者/スタイル参照オーディオのキャッシュはパスをキーにしています——同じパスでファイル内容を差し替えると古いキャッシュにヒットします。内容変更後はキャッシュを削除（`del_spk_audio` / `del_prompt_audio`）するか、新しいファイルパスを使用してください。

**Q4：初回実行のモデルダウンロードが遅い / 失敗する？**
`GSV_MIRROR` 環境変数でミラーを強制指定するか（中国国内では `modelscope` を推奨）、モデルファイルをキャッシュディレクトリ（デフォルト `~/.cache/gsv`。構成は「初回実行」セクション参照）に手動配置してください。

**Q5：CUDA グラフ関連のエラーが発生する？**
多くは `gpt_cache` / `sovits_cache` の設定不備によるものです。デフォルト値に戻してください。MPS/CPU 環境ではこれらのパラメータは不要です。

**Q6：モデルロード時に `weights_only=False` のセキュリティ警告が出る？**
レガシーな GPT-SoVITS チェックポイントとの互換性のための意図的な動作です。信頼できるソースのモデルのみをロードするか、`tts.to_safetensors` で safetensors ディレクトリ形式に変換してリスクを排除してください。

## ⚡ Flash Attention

**より低い遅延**と**より高いスループット**を追求する場合、Flash Attention の有効化を強く推奨します：

- 🐧 **Linux / ソースコードビルド**：[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- 🪟 **Windows ユーザー**：[lldacing/flash-attention-windows-wheel](https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main)（事前コンパイル済み Wheel）

> [!TIP]
> インストール完了後、TTS 設定で `use_flash_attn=True` を設定するだけで加速効果を楽しめます！🚀

## 謝辞 (Credits)

以下のプロジェクトに特別な感謝を表します：
- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [chinokikiss/GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite)
