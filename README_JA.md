<div align="center">

> [!IMPORTANT]
> ### 🔀 MultiSpeaker 独立開発リポジトリ
> 本リポジトリは [GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite) のフォークに由来します：上流リポジトリに `multi-speaker-inference` ブランチは存在せず、フォーク内で作成された後に独立して本リポジトリとなり、**マルチスピーカー（MultiSpeakerTTS）共有骨格推論**の独立開発・最適化に特化しています。
>
> - **上流リポジトリ**：[chinokikiss/GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite)（PyPI では `gsv-tts-lite` として公開）
> - **本リポジトリの配布名**：`gsv-tts-lite-multispeaker`（Python のインポート名は引き続き `gsv_tts`）。上流の配布パッケージとは独立しています
> - マルチスピーカー機能は**まだ PyPI に公開されていません**。本リポジトリが唯一のソースです。上流のバグ修正は cherry-pick / merge で同期できます

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

## プロジェクトについて (About)

**ひとことで言うと：このプロジェクトは AI が「複数の異なる声」でテキストを読み上げることができ、従来方式より VRAM/メモリを大幅に節約できます。**

想像してみてください：3 人のキャラクターが登場するポッドキャストやゲームのボイスを作る場合、従来方式ではキャラクターごとに完全な AI モデルを 1 式ロードする必要があり、3 キャラクターなら 3 式——非常にリソースを消費します。

このプロジェクトでは **1 式の「共通モデル」（骨格）** に加え、各キャラクターに**小さな「専用調整パック」（約 5-15% の軽量重み）**を用意するだけです。合成時はキャラクター名で自動切り替え——**キャラクターが多ければ多いほど節約効果が大きく**（実測 40%~75% 削減）。

- 対応言語：**中国語、日本語、英語**
- 対応モデル：**V2 / V2Pro / V2ProPlus**（分からなくても大丈夫、デフォルトで動きます）
- マルチスピーカーに加え、単一話者の全機能も備えています（下記 [使用ガイド](#-単一話者推論-tts) 参照）

> コードを書きたくない？[WebUI グラフィカルインターフェース](#-webui-可視化インターフェース) を使えば、ブラウザでクリックするだけで合成できます。

## ✨ 機能一覧 (Features)

- 🎭 **マルチスピーカー共有骨格**：1 式のモデルで 10+ 話者に対応。各話者の専用重みはごく小さなパックのみ
- 🔀 **ゼロコスト話者切替**：いつでも切り替え可能。ラグなし、追加メモリなし
- 🔌 **自動互換性チェック**：非互換の話者モデルはクラッシュせず自動デグレード。他の話者には影響なし
- ⚡ **3 つの利用モード**：単発合成、ストリーミング（生成しながら再生）、バッチ合成（同一話者の複数文を自動並列化）
- 🎵 **音色とスタイルを分離制御**：声が誰に似ているか（音色）と話し方のトーン（スタイル）を独立指定。呼び出しごとのスタイル上書きも可能
- ⏱️ **文字単位タイムスタンプ**：字幕用に各文字の時間を取得可能
- 🖥️ **WebUI / API 対応**：ブラウザでクリック操作、またはプログラムから API で連携
- 🌐 **自動言語検出**：中国語/日本語/英語——言語指定は不要

## 🎭 MultiSpeakerTTS：共有骨格推論（コア機能）

### 動作のしくみ（わかりやすく言うと）

**声優スタジオ**を想像してください：

- **骨格（バックボーン）** = スタジオの固定キャストと機材（1 式、全員で共用）
- **話者重み** = 各声優が持ち歩く小さな「声の調整キット」
- **話者切り替え** = 声優がキットを交換するだけで、スタジオと機材はそのまま

従来方式は全キャラクターにそれぞれ完全なスタジオ（キャスト＋機材）を持たせるようなもの——3 キャラクターでスタジオ 3 つ分のコストがかかります。このプロジェクトはスタジオ 1 つ＋キット N 個で同じことができます。これがメモリ/VRAM 削減のしくみです。

> 技術的な詳細（開発者向け）：共有骨格は 1 式の GPT + SoVITS モデル。各話者は約 25 GPT 重み + 37 SoVITS 重みのみ注入されます。話者名で動的に注入され、同時に有効なのは 1 話者分の重みだけなので、使用量 ≈ 骨格 1 式 + 話者 1 人分の重み。

### 実測データ（数字が気になる人向け。読み飛ばしても OK）

> [!NOTE]
> テスト環境：CPU（GPU なし）。実在のファインチューニングモデル（CyreneV3.7 / shouanren / LuoTianyi）を使用、短文推論の平均値。

| 指標 | 共有骨格 | 従来の全量ロード | 説明 |
| :--- | :---: | :---: | :--- |
| 話者あたり平均推論遅延 | 0.7~0.9s | 0.8~0.9s | ⚖️ 速度の低下なし |
| ピークメモリ (RAM) | **2.77 GB** | 4.65 GB | 💾 **-40%** |
| 3話者初期化時間 | 30.0s | 16.2s | 初回のみの準備。以降の切り替えはゼロコスト |

| 方式 | 1 キャラ | 3 キャラ | 5 キャラ | 10 キャラ |
|------|--------|--------|--------|---------|
| 従来の全量ロード | ~800MB | ~2.4GB | ~4.0GB | ~8.0GB |
| **本プロジェクト（共有骨格）** | ~800MB | **~1.2GB** | **~1.4GB** | **~2.0GB** |
| 節約 | — | **51%** | **65%** | **75%** |

> [!IMPORTANT]
> **話者モデルの互換性**：理想的には全話者モデルを骨格と同じ世代（v2ProPlus アーキテクチャ）に揃えます。違っても問題ありません——プログラムが自動検出し、その話者のみ従来方式の全量ロードにデグレードします。メモリ節約効果がなくなるだけで、エラーにはなりません。

### 使用方法（コピペで動きます）

```python
from gsv_tts import MultiSpeakerTTS, SpeakerConfig

# ステップ 1：話者を定義（パスは自分のモデル・音声に置き換え）
speakers = [
    SpeakerConfig(
        name="alice",                     # 話者名（自由に決めて OK）
        gpt_model_path="models/alice_gpt.ckpt",    # この話者の GPT モデル
        sovits_model_path="models/alice_sovits.pth",  # この話者の SoVITS モデル
        spk_audio_path="audio/alice_ref.wav",      # 音色参照音声
        prompt_audio_path="audio/alice_prompt.ogg", # スタイル参照音声（省略可、デフォルトで音色参照を使用）
        prompt_audio_text="こんにちは、アリスです。",  # スタイル参照音声の内容
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

# ステップ 2：全話者を一度にロード
tts = MultiSpeakerTTS(speakers=speakers, use_bert=True)

# ステップ 3：話者名で合成
audio = tts.infer("alice", "今日も頑張りましょう！", text_language="ja")
audio.play()
tts.audio_queue.wait()

# 1 つの台本で複数話者を混ぜる？タプルリストで一括合成
audios = tts.infer_batched(
    [
        ("alice", "こんにちは"),
        ("bob",   "よろしくお願いします"),
    ],
    text_languages=["ja", "ja"],
)

# 実行中に話者を追加 / 削除することも可能（再起動不要）
tts.add_speaker(SpeakerConfig(name="charlie", ...))
tts.remove_speaker("bob")
```

## 🚀 クイックスタート (Quick Start)

### 必要なもの

- ✅ パソコン（CPU でも動作。NVIDIA GPU があればより高速）
- ✅ Python **3.10 以上**（分からなければ「Python インストール」で検索）
- ✅ インターネット接続（初回実行でモデルをダウンロード、約 5~10 GB）
- ✅ ディスク容量：モデルはデフォルトで `~/.cache/gsv` に保存（`models_dir` 引数で変更可）

### インストール（コマンド 3 つ）

```bash
# 1. PyTorch（ディープラーニングフレームワーク）をインストール
#    NVIDIA GPU がある場合（中国国内ミラー。遅い場合は https://download.pytorch.org/whl/cu128 に変更）：
pip install torch torchvision torchaudio --index-url https://mirrors.aliyun.com/pytorch-wheels/cu128
#    GPU がない場合（Mac / 通常 PC）：
#    pip install torch torchvision torchaudio

# 2. 本リポジトリをクローンしてインストール
git clone https://github.com/jinyiwei2012/gsv-tts-lite-multispeaker.git
cd gsv-tts-lite-multispeaker
pip install -e .
```

> [!WARNING]
> **重要**：マルチスピーカー機能はまだ PyPI にありません（`pip install gsv-tts-lite` は単一話者版のみ）。**必ず**上記のように本リポジトリからインストールしてください。

### 初回実行：モデルの自動ダウンロード（一度だけ）

初回実行時に、合成に必要な「材料」（事前学習済みモデル）が `~/.cache/gsv` に**自動ダウンロード**されます：

| ファイル | 役割 |
| :--- | :--- |
| `s1v3.ckpt` | GPT モデル：「何を、どう言うか」を決定 |
| `s2Gv2ProPlus.pth` | SoVITS モデル：意味を音声に変換 |
| `chinese-hubert-base` | 音声特徴抽出（参照音声の処理用） |
| `g2p` | テキスト→読み（発音）変換 |
| `sv` | 声紋認識（声が誰に似ているかを判定） |
| `chinese-roberta-wwm-ext-large` | 中国語理解の強化（中国語品質の向上） |

ダウンロードは回線速度にもよりますが数分〜数十分かかります。中国国内では ModelScope ミラーが自動選択されます。遅い・失敗する場合はミラーを強制指定：

```bash
# Windows (PowerShell)
$env:GSV_MIRROR = "modelscope"
# Linux/macOS
export GSV_MIRROR=modelscope
```

### 単一話者の基本推論（とりあえず音を出してみたい人向け）

```python
from gsv_tts import TTS

tts = TTS(use_bert=True)

# リポジトリ付属のサンプル音声でそのまま合成できます
audio = tts.infer(
    spk_audio_path="examples/laffey.mp3",   # 音色参照：誰の声を使うか
    prompt_audio_path="examples/AnAn.ogg",  # スタイル参照：どんなトーンか
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",  # スタイル参照音声の内容
    text="こんにちは、世界！",                # 合成するテキスト
    text_language="ja",                      # テキストの言語：auto で自動検出
)

audio.play()
tts.audio_queue.wait()
```

## 📖 単一話者推論 (TTS)

> 以下は `TTS` 単一話者エンジンの上級用法です。マルチスピーカーエンジン（MultiSpeakerTTS）にも同等の機能があります。

<details>
<summary><strong>1. ストリーミング合成（生成しながら再生。リアルタイム対話向け）</strong></summary>

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

> 字幕用に文字単位のタイムスタンプが欲しい？`return_subtitles=True` を追加すれば、各文字の開始/終了時間が結果に含まれます。

</details>

<details>
<summary><strong>2. バッチ合成（長文・複数文でより効率的）</strong></summary>

```python
from gsv_tts import TTS

tts = TTS(use_bert=True)

audios = tts.infer_batched(
    spk_audio_paths="examples/laffey.mp3",
    prompt_audio_paths="examples/AnAn.ogg",
    prompt_audio_texts="ちが……ちがう。レイア、貴様は間違っている。",
    texts=["こんにちは", "The old map crinkled in Leo's trembling hands."],
    text_languages="auto",
    prompt_languages="auto",
)

for i, audio in enumerate(audios):
    audio.save(f"audio{i}.wav")
```

</details>

<details>
<summary><strong>3. 音色変換（変声）と声紋認識</strong></summary>

```python
from gsv_tts import TTS

# 音色変換：ある音声の内容を、別の人の声で読み上げる
tts = TTS(use_bert=True, always_load_cnhubert=True)
audio = tts.infer_vc(
    spk_audio_path="examples/laffey.mp3",    # 目標の音色
    prompt_audio_path="examples/AnAn.ogg",   # 元の音声コンテンツ
    prompt_audio_text="ちが……ちがう。レイア、貴様は間違っている。",
)
audio.play()

# 声紋認識：2 つの音声が同一人物か判定
tts2 = TTS(use_bert=True, always_load_sv=True)
similarity = tts2.verify_speaker("examples/laffey.mp3", "examples/AnAn.ogg")
print("声紋類似度：", similarity)
```

</details>

<details>
<summary><strong>4. その他の関数インターフェース（開発者向け）</strong></summary>

- `load_gpt_model(path)` / `load_sovits_model(path)` — モデル重みをメモリにロード
- `unload_gpt_model(path)` / `unload_sovits_model(path)` — モデルをアンロードしてリソース解放
- `get_gpt_list()` / `get_sovits_list()` — ロード済みモデルの一覧
- `to_safetensors(path)` — .pth/.ckpt をより安全な safetensors 形式に変換
- `cache_spk_audio(path)` / `cache_prompt_audio(path, text)` — 音声を事前キャッシュして初回遅延を削減
- `infer_async(...)` / `infer_stream_async(...)` / `infer_batched_async(...)` — 非同期版

</details>

## 🌐 WebUI 可視化インターフェース（コード不要）

コードを書きたくないなら WebUI をどうぞ：ブラウザで開き、音声をアップロードしてテキストを入力し、ボタンを押すだけ。単一モデル / マルチスピーカーの両モードに対応（マルチスピーカーは `<speaker:名前>テキスト</speaker:名前>` タグによる混在合成もサポート）。

```bash
cd WebUI
pip install -r requirements.txt   # -e .. で本リポジトリの gsv_tts を自動インストール（MultiSpeakerTTS は PyPI 未公開）
python web.py                     # オプション: --port 9881 / --use_asr / --models_dir ...
```

起動するとブラウザで `http://127.0.0.1:9881` が自動的に開きます。

## 🔌 API サービスインターフェース（開発者向け）

```bash
cd API
pip install -r requirements.txt
```

- コアドキュメント：[API 詳細ガイド](API/README.md)、[Personal API ドキュメント](API/PERSONAL_API.md)
- エントリポイント：`API/personal_api.py`（MultiSpeaker エンドポイント）、`API/realtime_api.py`（リアルタイムストリーミング）

> 6 つのマルチスピーカー管理エンドポイント（`/multi-speaker/init`、`/add`、`/remove`、`/list`、`/infer`、`/batch`）＋ `/multi-speaker/stream` ストリーミングエンドポイントで、プログラムからの統合に対応。

<details>
<summary><strong>📁 プロジェクト構成（開発者向け）</strong></summary>

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
examples/                 # サンプル参照音声（laffey.mp3 / AnAn.ogg）
```

</details>

<details>
<summary><strong>🛠️ 開発とデバッグ（開発者向け）</strong></summary>

**テスト**（pytest 不要。スクリプトとして直接実行）：

```bash
# 自己整合性テスト：共有骨格の出力と全量モデルの一致を検証（MCD 指標）
python tests/test_sovits_sharing.py

# 実モデルでの評価
python tests/test_sovits_sharing.py --speaker-gpt path/to/gpt.ckpt --speaker-sovits path/to/sovits.pth
```

> MCD の計算には `librosa` が必要。未インストールの場合は警告とともにスキップされます。

**ベンチマーク**（リポジトリ内のモデルを自動検出、または手動指定）：

```bash
python benchmarks/bench_multi_speaker.py                          # リポジトリ内の .ckpt/.pth を自動ペアリング
python benchmarks/bench_multi_speaker.py --models-dir models      # スキャン先ディレクトリを指定
python benchmarks/bench_multi_speaker.py --gpt a.ckpt --sovits b.pth   # 明示指定（繰り返し可）
```

**モデル形式と互換性**：

- 従来の `.ckpt` / `.pth` チェックポイント（ロード時にセキュリティ警告が出ますが正常です）に加え、より安全な **safetensors ディレクトリ形式**（`hps.json` + `model.safetensors`）に対応。`tts.to_safetensors(path)` で変換できます。
- SoVITS バージョン自動検出：ファイルヘッダ（`01`=v2、`05`=v2Pro、`06`=v2ProPlus）。認識できない場合は v2 として処理し警告。
- デバイス差異：Mac/CPU 環境では float32 が強制され一部キャッシュが無効化。CPU では INT8 量子化 BERT、GPU ではオリジナルモデルを使用。

**よくある落とし穴**：

- `Loader.py` 先頭の `sys.modules['utils']` monkey-patch は削除しない（旧モデルのロードに必要）。
- `gpt_cache` / `sovits_cache` は安易に変更しない——設定を誤ると CUDA グラフエラーになります。
- 推論は `_infer_lock` で直列化され自動でキャッシュクリアされます。モデルは遅延ロード（初回推論時のみ）です。

</details>

## ❓ よくある質問 (FAQ)

**Q1：`pip install gsv-tts-lite` してもマルチスピーカー機能がない？**
マルチスピーカー機能はまだ PyPI に未公開です。本リポジトリから `git clone` + `pip install -e .` でインストールしてください（[クイックスタート](#-クイックスタート-quick-start) 参照）。

**Q2：初回のモデルダウンロードが遅い / 失敗する？**
`GSV_MIRROR` 環境変数でミラーを強制指定（中国国内では `modelscope` を推奨）、またはモデルファイルを `~/.cache/gsv` に手動配置してください（ファイル一覧は[初回実行](#初回実行モデルの自動ダウンロード一度だけ)参照）。

**Q3：特定の話者だけメモリ使用量が異常に多い？**
その話者のモデルが骨格と非互換（例：v2 vs v2ProPlus）のため、自動的に従来方式の全量ロードへデグレードされています（ログに表示）。メモリを節約したい場合は全話者モデルを v2ProPlus アーキテクチャに統一してください。

**Q4：参照音声の内容を変えたのに結果が変わらない？**
音声キャッシュはパスをキーにしています——同じパスで内容を差し替えても古いキャッシュが使われます。内容変更後はキャッシュを削除（`del_spk_audio` / `del_prompt_audio`）するか、ファイル名を変えてください。

**Q5：CUDA グラフ関連のエラーが発生する？**
多くは `gpt_cache` / `sovits_cache` の変更によるものです。デフォルトに戻してください。Mac/CPU 環境ではこれらのパラメータは不要です。

**Q6：モデルロード時に `weights_only=False` のセキュリティ警告が出る？**
旧チェックポイントとの互換性のための意図的な動作です。信頼できるモデルファイルのみをロードするか、`tts.to_safetensors` で safetensors 形式に変換してリスクを排除してください。

<details>
<summary><strong>📖 用語集（わかりやすい説明）</strong></summary>

| 用語 | わかりやすい説明 |
| :--- | :--- |
| **GPT モデル** | 「何を、どう言うか」を決定するモデル（テキスト → 意味） |
| **SoVITS モデル** | 意味を音声に変換するモデル（意味 → 音声） |
| **骨格（バックボーン）** | 全話者が共用する 1 式のモデル |
| **重み（Weights）** | モデルが「学習したもの」——各話者の小さな専用調整パック |
| **音色参照音声** | 「誰の声を使うか」を指定する数秒の音声 |
| **スタイル参照音声** | 「どんなトーン・感情を使うか」を指定する音声（任意） |
| **VRAM** | グラフィックカード上のメモリ。多いほど大規模なモデルを扱える |
| **safetensors** | .pth/.ckpt に代わる、より安全なモデルファイル形式 |
| **v2 / v2Pro / v2ProPlus** | SoVITS モデルの 3 世代。新しいほど高品質 |
| **ファインチューニング** | 特定の人の声データで学習した専用話者モデル |
| **音素（Phoneme）** | 言語の最小の発音単位 |
| **BERT** | 中国語の理解力を高める言語モデル |
| **RTF** | リアルタイム率：1 秒分の音声合成にかかる時間。1 未満ならリアルタイムより速い |

</details>

<details>
<summary><strong>⚡ Flash Attention（任意の高速化）</strong></summary>

より低い遅延・より高いスループットを求める場合は Flash Attention を有効化できます：

- 🐧 **Linux**：[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)（ソースからビルド）
- 🪟 **Windows**：[lldacing/flash-attention-windows-wheel](https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main)（ビルド済み Wheel）

コードで `use_flash_attn=True` を設定するだけです。

</details>

## 謝辞 (Credits)

以下のプロジェクトに特別な感謝を表します：
- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [chinokikiss/GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite)
