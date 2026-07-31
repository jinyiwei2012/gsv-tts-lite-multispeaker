import re
import os
import sys
import time
import json
import uuid
import pickle
import torch
import logging
import argparse
import gradio as gr
import numpy as np
from datetime import datetime
from pedalboard import Pedalboard, Compressor, HighpassFilter, PeakFilter, Reverb, Gain
import pyloudnorm as pyln
from pathlib import Path
from huggingface_hub import snapshot_download
import platform

# 添加项目根目录到 sys.path，确保能找到 gsv_tts 模块
project_root = Path(__file__).parent.parent
webui_dir = Path(__file__).parent
PRESETS_DIR = webui_dir / "presets"
HISTORY_DIR = webui_dir / "output_history"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if platform.system() == "Windows":
    import psutil
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)

from gsv_tts import TTS, AudioClip, MultiSpeakerTTS, SpeakerConfig, ConfigMismatchError

logging.getLogger('asyncio').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.CRITICAL)

# Module-level sentinel defaults — avoid NameError when module is imported
# (e.g. Gradio reload).  Actual values are set in the __main__ block.
GSV_ROOT_DIR: str | None = None
USE_BERT: bool = True
tts: TTS | None = None
multi_tts: MultiSpeakerTTS | None = None
asr = None


# Copied from https://github.com/Icelinea/BetterAIVoice/blob/main/process.py
def enhance_audio(audio_data, sample_rate):
    # 1. 构建美化链
    board = Pedalboard([
        # 去除低频浑浊
        HighpassFilter(cutoff_frequency_hz=80),
        
        # 增加女声磁性：250Hz-350Hz 提升
        PeakFilter(cutoff_frequency_hz=300, gain_db=2.5, q=1.0),
        
        # 压制 AI 齿音：6kHz-8kHz 微微削减
        PeakFilter(cutoff_frequency_hz=7000, gain_db=-3.0, q=2.0),
        
        # 稳定动态：防止有声书音量忽大忽小
        Compressor(threshold_db=-18, ratio=3.5),
        
        # 赋予录音棚空间感
        # 使用内建 Reverb 模拟 Ambience 预设 (Mix 3%, 极小衰减)
        Reverb(room_size=0.1, dry_level=0.97, wet_level=0.03, damping=0.5),
        
        # 最终增益补偿
        Gain(gain_db=2)
    ])

    # 2. 执行处理
    effected = board(audio_data, sample_rate)
    input_for_norm = effected.reshape(-1, 1)

    # 3. 响度标准化
    # 测量当前响度
    meter = pyln.Meter(sample_rate) 
    loudness = meter.integrated_loudness(input_for_norm)
    # 将响度统一调整至 -18.0 LUFS (播客标准)
    normalized_audio = pyln.normalize.loudness(input_for_norm, loudness, -18.0).T

    return normalized_audio.flatten()

S1_MODEL_PATH = [
    "GPT_weights_v2",
    "GPT_weights_v2Pro",
    "GPT_weights_v2ProPlus",
]
S2_MODEL_PATH = [
    "SoVITS_weights_v2",
    "SoVITS_weights_v2Pro",
    "SoVITS_weights_v2ProPlus",
]

def find_gsv_models():
    if GSV_ROOT_DIR is None or not os.path.isdir(GSV_ROOT_DIR):
        return gr.update(choices=['']), gr.update(choices=[''])
    s1 = ['']
    s2 = ['']
    for item in S2_MODEL_PATH:
        cd = os.path.join(GSV_ROOT_DIR, item)
        if os.path.isdir(cd):
            s2 += [(f"{item}/{i}",os.path.join(GSV_ROOT_DIR, item, i)) for i in os.listdir(cd) if i.endswith(".pth")]
    for item in S1_MODEL_PATH:
        cd = os.path.join(GSV_ROOT_DIR, item)
        if os.path.isdir(cd):
            s1 += [(f"{item}/{i}", os.path.join(GSV_ROOT_DIR, item, i)) for i in os.listdir(cd) if i.endswith(".ckpt")]
    return gr.update(choices=s1), gr.update(choices=s2)


def upload_gpt(new_gpt):
    if tts is None:
        raise RuntimeError("TTS not initialized. Run web.py as main module.")
    if not new_gpt is None:
        for gpt in tts.get_gpt_list():
            tts.unload_gpt_model(gpt)
        
        tts.load_gpt_model(new_gpt.strip('"“”'))

def upload_sovits(new_sovits):
    if tts is None:
        raise RuntimeError("TTS not initialized. Run web.py as main module.")
    if not new_sovits is None:
        for sovits in tts.get_sovits_list():
            tts.unload_sovits_model(sovits)
        
        tts.load_sovits_model(new_sovits.strip('"“”'))


def update_spk_weights(files, weights):
    if not files:
        return "1.0"

    weights = re.split(r'[：:]\s*', weights)
    weights = [weight for weight in weights if weight]

    f_len = len(files)
    w_len = len(weights)
    if f_len <= w_len:
        new_weights = weights[:f_len]
    else:
        new_weights = weights + ["1.0"]*(f_len-w_len)

    return ": ".join(new_weights)


ignore_transcribe = False
def audio_transcriber(audio_file):
    global ignore_transcribe

    if ignore_transcribe:
        ignore_transcribe = False
        audio_file = None

    if not audio_file is None and not asr is None:
        results = asr.transcribe(audio_file)
        text = results[0].text

        return text
    
    return gr.update()


def auto_fill_speaker_ref(prompt_audio_path, current_spk_files):
    """当上传风格参考音频时，如果音色参考音频没有上传，则自动将风格参考音频添加到音色参考"""
    if prompt_audio_path is not None and (not current_spk_files or len(current_spk_files) == 0):
        return gr.update(value=[prompt_audio_path])
    return gr.update()


def parse_tagged_text(text):
    parts = re.split(r'(<(?!(?:break))[^>]+>.*?</[^>]+>)', text)

    cut_texts = []
    tags = []
    for part in parts:
        if not part: continue

        match = re.search(r'<([^>]+)>(.*?)</[^>]+>', part)
        if match:
            tag_name = match.group(1)
            content = match.group(2)
            sub_parts = re.split(r'(<break:.*?>)', content)
            sub_parts = [p for p in sub_parts if p]
            tags.extend([tag_name]*len(sub_parts))
        else:    
            sub_parts = re.split(r'(<break:.*?>)', part)
            sub_parts = [p for p in sub_parts if p]
            tags.extend([None]*len(sub_parts))

        cut_texts.extend(sub_parts)
    
    for i in range(len(cut_texts)-1, -1, -1):
        if len(re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\u3040-\u309f\u30a0-\u30ff]', '', cut_texts[i])) == 0:
            cut_texts.pop(i)
            tags.pop(i)

    return cut_texts, tags

def parse_speaker_weights(multi_spk_files, spk_weights):
    spk_weights = re.split(r'[：:]\s*', spk_weights)
    spk_audio = {multi_spk_files[i]: float(item) for i, item in enumerate(spk_weights) if item}
    return spk_audio


def get_preset_path(name):
    return PRESETS_DIR / f"{name}.pkl"


def get_preset_names():
    return sorted(p.stem for p in PRESETS_DIR.glob("*.pkl"))


def refresh_preset_dropdown():
    return gr.update(choices=get_preset_names())


def read_preset(name):
    with open(get_preset_path(name), "rb") as f:
        return pickle.load(f)


def save_preset(name, prompt_audio, prompt_text, multi_spk_files, spk_weights):
    if not name:
        return refresh_preset_dropdown(), "请输入预设名称"
    PRESETS_DIR.mkdir(exist_ok=True)
    preset = {
        "prompt_audio": prompt_audio,
        "prompt_text": prompt_text,
        "multi_spk_files": multi_spk_files,
        "spk_weights": spk_weights
    }
    with open(get_preset_path(name), "wb") as f:
        pickle.dump(preset, f)
    return gr.update(choices=get_preset_names(), value=name), f"预设 '{name}' 已保存"

def load_preset(name):
    global ignore_transcribe
    ignore_transcribe = True

    if not name:
        return None, "", None, "1.0"
    p = read_preset(name)
    return p["prompt_audio"], p["prompt_text"], p["multi_spk_files"], p["spk_weights"]


# ============================================================
# Multi-Speaker 多角色推理
# ============================================================

multi_tts: MultiSpeakerTTS | None = None
_speaker_data: list[dict] = []  # [{name, gpt_path, sovits_path, spk_audio, prompt_audio, prompt_text, mode}]


def _render_speaker_table():
    """Render speaker list table - returns data directly for Gradio 6.x"""
    if not _speaker_data:
        return []
    return [
        [s["name"], Path(s["gpt_path"]).name, Path(s["sovits_path"]).name, s["mode"]]
        for s in _speaker_data
    ]


def multi_add_speaker(
    name, gpt_path, sovits_path,
    spk_audio, prompt_audio, prompt_text,
):
    """Add a speaker to MultiSpeakerTTS (auto-initializes if needed)."""
    global multi_tts, _speaker_data
    if not name:
        return _render_speaker_table(), "❌ 请输入角色名"
    if not gpt_path or not sovits_path:
        return _render_speaker_table(), "❌ 请填写 GPT 和 SoVITS 模型路径"
    if not spk_audio:
        return _render_speaker_table(), "❌ 请上传音色参考音频"
    if not prompt_text:
        return _render_speaker_table(), "❌ 请填写风格参考文本（上传风格音频或复用音色参考时都需要）"

    try:
        spk = SpeakerConfig(
            name=name,
            gpt_model_path=gpt_path,
            sovits_model_path=sovits_path,
            spk_audio_path=spk_audio,
            prompt_audio_path=prompt_audio or spk_audio,
            prompt_audio_text=prompt_text,
        )

        # Auto-initialize on the first speaker — the core requires >= 1
        # speaker at construction time (empty list raises ValueError).
        if multi_tts is None:
            multi_tts = MultiSpeakerTTS(speakers=[spk], use_bert=USE_BERT)
        else:
            multi_tts.add_speaker(spk)

        w = multi_tts._speakers[name]
        mode = "🔄 完整模型" if w.is_full_model else "✅ 共享骨干"
        _speaker_data.append({
            "name": name, "gpt_path": gpt_path, "sovits_path": sovits_path,
            "spk_audio": spk_audio, "prompt_audio": prompt_audio,
            "prompt_text": prompt_text, "mode": mode,
        })
        return _render_speaker_table(), f"✅ 角色 '{name}' 已添加 ({mode})"
    except Exception as e:
        return _render_speaker_table(), f"❌ 添加失败: {e}"


def multi_remove_speaker(name):
    """Remove a speaker."""
    global multi_tts, _speaker_data
    if not name:
        return _render_speaker_table(), "❌ 请选择要移除的角色"
    if multi_tts is None:
        return _render_speaker_table(), "❌ 引擎未初始化"

    try:
        multi_tts.remove_speaker(name)
        _speaker_data = [s for s in _speaker_data if s["name"] != name]
        return _render_speaker_table(), f"✅ 角色 '{name}' 已移除"
    except Exception as e:
        return _render_speaker_table(), f"❌ 移除失败: {e}"


def _get_speaker_choices():
    names = [s["name"] for s in _speaker_data]
    return gr.update(choices=names, value=names[0] if names else None)

def _get_remove_choices():
    names = [s["name"] for s in _speaker_data]
    return gr.update(choices=names)


def vc_request(
    multi_spk_files, spk_weights,
    prompt_audio, prompt_text,
):
    try:
        start_time = time.time()

        audio = tts.infer_vc(
            spk_audio_path=parse_speaker_weights(multi_spk_files, spk_weights),
            prompt_audio_path=prompt_audio,
            prompt_audio_text=prompt_text,
        )

        end_time = time.time()

        infer_duration = end_time - start_time

        msg = (
            f"成功！\n"
            f"音频时长: {audio.audio_len_s:.2f}s | "
            f"推理耗时: {infer_duration:.2f}s"
        )

        return (audio.samplerate, audio.audio_data), msg

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"异常: {str(e)}"

def tts_request(
    multi_spk_files, spk_weights,
    prompt_audio, prompt_text,
    text,
    top_k, top_p, temperature, rep_penalty, noise_scale, speed,
    enable_enhance,
    is_cut_text, cut_minlen, cut_mute, cut_mute_scale_map,
    sovits_batch_size,
    text_language, prompt_language,
    mode, multi_cur_speaker,
):
    """Unified TTS inference — routes to single-model or multi-speaker engine."""
    try:
        start_time = time.time()

        if mode == "多角色" and multi_tts is not None:
            # ── Multi-speaker mode ──
            return _tts_multi_infer(
                multi_cur_speaker, text,
                top_k, top_p, temperature, rep_penalty, noise_scale, speed,
                enable_enhance, start_time,
                text_language, prompt_language,
            )

        # ── Single-model mode (original logic) ──
        spk_audio = parse_speaker_weights(multi_spk_files, spk_weights)
        cut_mute_scale_map = json.loads(cut_mute_scale_map)
        cut_texts, tags = parse_tagged_text(text)

        orig_idx = []
        spk_audio_paths = []
        prompt_audio_paths = []
        prompt_audio_texts = []
        texts = []

        preset_names = set(get_preset_names())
        for i in range(len(cut_texts)):
            result = re.search(r'<break:(.*?)/>', cut_texts[i])
            if result:
                cut_texts[i] = float(result.group(1))
                tags[i] = 'break'
            else:
                orig_idx.append(i)
                if tags[i] is None or tags[i] not in preset_names:
                    spk_audio_paths.append(spk_audio)
                    prompt_audio_paths.append(prompt_audio)
                    prompt_audio_texts.append(prompt_text)
                else:
                    p = read_preset(tags[i])
                    spk_audio_paths.append(parse_speaker_weights(p["multi_spk_files"], p["spk_weights"]))
                    prompt_audio_paths.append(p["prompt_audio"])
                    prompt_audio_texts.append(p["prompt_text"])
                texts.append(cut_texts[i])

        audios = tts.infer_batched(
            spk_audio_paths=spk_audio_paths,
            prompt_audio_paths=prompt_audio_paths,
            prompt_audio_texts=prompt_audio_texts,
            texts=texts,
            text_languages=text_language,
            prompt_languages=prompt_language,
            is_cut_text=is_cut_text,
            cut_minlen=cut_minlen,
            cut_mute=cut_mute,
            cut_mute_scale_map=cut_mute_scale_map,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=rep_penalty,
            noise_scale=noise_scale,
            speed=speed,
            sovits_batch_size=sovits_batch_size,
            return_subtitles=True,
        )

        samplerate = audios[0].samplerate
        audio_data = []
        audio_len_s = 0
        all_subtitles = []
        offset = 0.0
        for i in range(len(cut_texts)):
            if tags[i] == "break":
                offset += cut_texts[i]
            else:
                tmp_audio = audios[orig_idx.index(i)]
                audio_data.append(tmp_audio.audio_data)
                audio_len_s += tmp_audio.audio_len_s
                if tmp_audio.subtitles:
                    for s in tmp_audio.subtitles:
                        shifted = dict(s)
                        shifted["start_s"] = s["start_s"] + offset
                        shifted["end_s"] = s["end_s"] + offset
                        all_subtitles.append(shifted)
                offset += tmp_audio.audio_len_s

        audio_data = np.concatenate(audio_data)
        audio = AudioClip(None, audio_data, samplerate, audio_len_s, all_subtitles or None, None)

        if enable_enhance:
            audio.audio_data = enhance_audio(audio.audio_data, audio.samplerate)

        end_time = time.time()
        infer_duration = end_time - start_time
        rtf = infer_duration / audio.audio_len_s

        msg = (
            f"成功！\n"
            f"音频时长: {audio.audio_len_s:.2f}s | "
            f"推理耗时: {infer_duration:.2f}s | "
            f"RTF: {rtf:.3f}"
        )

        filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}.wav"
        save_path = HISTORY_DIR / filename
        audio.save(str(save_path))
        if all_subtitles:
            audio.export_subtitles(str(save_path)[:-4] + ".srt")
        history_entry = [datetime.now().strftime("%H:%M:%S"), text[:20] + "...", str(save_path)]

        return (audio.samplerate, audio.audio_data), msg, history_entry

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"异常: {str(e)}", None


def _tts_multi_infer(speaker, text, top_k, top_p, temperature, rep_penalty,
                     noise_scale, speed, enable_enhance, start_time,
                     text_language="auto", prompt_language="auto"):
    """Multi-speaker inference backend (called from tts_request)."""
    # Parse <speaker:name> tags for multi-speaker mixing
    tagged = re.findall(r'<speaker:([^>]+)>(.*?)</speaker>', text, re.DOTALL)

    if tagged:
        speaker_texts = [(spk.strip(), t.strip()) for spk, t in tagged]
        remaining = re.sub(r'<speaker:[^>]+>.*?</speaker>', '', text, flags=re.DOTALL).strip()
        if remaining and speaker:
            parts = re.split(r'(<speaker:[^>]+>.*?</speaker>)', text)
            for part in parts:
                part = part.strip()
                if not part: continue
                m = re.match(r'<speaker:([^>]+)>(.*?)</speaker>', part)
                if m:
                    speaker_texts.append((m.group(1).strip(), m.group(2).strip()))
                else:
                    speaker_texts.append((speaker, part))
        seen = set()
        ordered = []
        for st in speaker_texts:
            key = (st[0], st[1])
            if key not in seen:
                seen.add(key)
                ordered.append(st)

        audios = multi_tts.infer_batched(ordered, text_languages=text_language,
                                         prompt_languages=prompt_language,
                                         top_k=top_k, top_p=top_p,
                                         temperature=temperature,
                                         repetition_penalty=rep_penalty,
                                         noise_scale=noise_scale, speed=speed)
        samplerate = audios[0].samplerate
        audio_data = np.concatenate([a.audio_data for a in audios])
        audio_len_s = sum(a.audio_len_s for a in audios)
    else:
        audio = multi_tts.infer(speaker=speaker, text=text,
                                text_language=text_language,
                                prompt_language=prompt_language,
                                top_k=top_k, top_p=top_p,
                                temperature=temperature,
                                repetition_penalty=rep_penalty,
                                noise_scale=noise_scale, speed=speed)
        samplerate = audio.samplerate
        audio_data = audio.audio_data
        audio_len_s = audio.audio_len_s

    if enable_enhance:
        audio_data = enhance_audio(audio_data, samplerate)

    infer_duration = time.time() - start_time
    rtf = infer_duration / audio_len_s if audio_len_s > 0 else 0
    spk_count = len(set(spk for spk, _ in tagged)) if tagged else 1
    msg = (f"✅ 多角色合成 · {spk_count} 个角色\n"
           f"音频时长: {audio_len_s:.2f}s | "
           f"推理耗时: {infer_duration:.2f}s | RTF: {rtf:.3f}")

    filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}.wav"
    save_path = HISTORY_DIR / filename
    import soundfile as sf
    sf.write(str(save_path), audio_data, samplerate)
    history_entry = [datetime.now().strftime("%H:%M:%S"), text[:20] + "...", str(save_path)]

    return (samplerate, audio_data), msg, history_entry


def _parse_speaker_stream_segments(text, default_speaker):
    """Split text into [(speaker, segment), ...] honoring <speaker:name> tags.

    Plain text outside tags uses default_speaker. Empty segments are dropped.
    """
    segments = []
    pattern = re.compile(r'<speaker:([^>]+)>(.*?)</speaker>', re.DOTALL)
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            plain = text[pos:m.start()].strip()
            if plain:
                segments.append((default_speaker, plain))
        seg_text = m.group(2).strip()
        if seg_text:
            segments.append((m.group(1).strip(), seg_text))
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        segments.append((default_speaker, tail))
    return segments


def tts_stream_request(
    multi_spk_files, spk_weights,
    prompt_audio, prompt_text,
    text,
    top_k, top_p, temperature, rep_penalty, noise_scale, speed,
    enable_enhance,
    text_language, prompt_language,
    mode, multi_cur_speaker,
):
    """Streaming inference (generator) — token-level streaming for both engines.

    Single-model mode uses TTS.infer_stream; multi-speaker mode uses the
    MultiSpeakerTTS.infer_stream (shared backbone, token-level streaming).
    Multi-speaker mode also supports <speaker:name> tags — each tagged
    segment is streamed sequentially with the matching speaker.
    Yields (audio, status) progressively so Gradio renders chunks in real time.
    """
    def gen():
        try:
            start_time = time.time()
            chunks = []

            if mode == "多角色" and multi_tts is not None:
                if "<speaker:" in text:
                    # 混合角色：按标签逐段流式，每段用对应角色
                    segments = _parse_speaker_stream_segments(text, multi_cur_speaker)
                    if not segments:
                        yield None, "⚠️ 未能解析出有效文本", None
                        return
                    for spk, seg in segments:
                        seg_gen = multi_tts.infer_stream(
                            speaker=spk,
                            text=seg,
                            text_language=text_language,
                            prompt_language=prompt_language,
                            top_k=top_k, top_p=top_p,
                            temperature=temperature,
                            repetition_penalty=rep_penalty,
                            noise_scale=noise_scale,
                            speed=speed,
                            debug=False,
                        )
                        for chunk in seg_gen:
                            chunks.append(chunk)
                            total_s = sum(c.audio_len_s for c in chunks)
                            yield (chunk.samplerate, chunk.audio_data), \
                                f"⚡ 多角色流式生成中 [{spk}]... 已输出 {len(chunks)} 段 (累计 {total_s:.2f}s)", None
                else:
                    stream_gen = multi_tts.infer_stream(
                        speaker=multi_cur_speaker,
                        text=text,
                        text_language=text_language,
                        prompt_language=prompt_language,
                        top_k=top_k, top_p=top_p,
                        temperature=temperature,
                        repetition_penalty=rep_penalty,
                        noise_scale=noise_scale,
                        speed=speed,
                        debug=False,
                    )
                    for i, chunk in enumerate(stream_gen):
                        chunks.append(chunk)
                        total_s = sum(c.audio_len_s for c in chunks)
                        yield (chunk.samplerate, chunk.audio_data), f"⚡ 流式生成中... 已输出 {i + 1} 段 (累计 {total_s:.2f}s)", None
            else:
                spk_audio = parse_speaker_weights(multi_spk_files, spk_weights)
                stream_gen = tts.infer_stream(
                    spk_audio_path=spk_audio,
                    prompt_audio_path=prompt_audio,
                    prompt_audio_text=prompt_text,
                    text=text,
                    text_language=text_language,
                    prompt_language=prompt_language,
                    top_k=top_k, top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=rep_penalty,
                    noise_scale=noise_scale,
                    speed=speed,
                    debug=False,
                )
                for i, chunk in enumerate(stream_gen):
                    chunks.append(chunk)
                    total_s = sum(c.audio_len_s for c in chunks)
                    yield (chunk.samplerate, chunk.audio_data), f"⚡ 流式生成中... 已输出 {i + 1} 段 (累计 {total_s:.2f}s)", None

            if not chunks:
                yield None, "⚠️ 未生成任何音频", None
                return

            samplerate = chunks[0].samplerate
            audio_data = np.concatenate([c.audio_data for c in chunks])
            audio_len_s = sum(c.audio_len_s for c in chunks)

            if enable_enhance:
                audio_data = enhance_audio(audio_data, samplerate)

            infer_duration = time.time() - start_time
            rtf = infer_duration / audio_len_s if audio_len_s > 0 else 0
            msg = (f"✅ 流式合成成功\n"
                   f"音频时长: {audio_len_s:.2f}s | "
                   f"推理耗时: {infer_duration:.2f}s | RTF: {rtf:.3f}")

            filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}.wav"
            save_path = HISTORY_DIR / filename
            import soundfile as sf
            sf.write(str(save_path), audio_data, samplerate)
            history_entry = [datetime.now().strftime("%H:%M:%S"), text[:20] + "...", str(save_path)]

            yield (samplerate, audio_data), msg, history_entry
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield None, f"异常: {str(e)}", None

    return gen()


# --- UI 界面 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# GSV-TTS")

    with gr.Tabs():
        with gr.TabItem("文本转语音 (TTS)"):

            history_state = gr.State([])
            tts_mode = gr.Radio(
                choices=["单模型", "多角色"],
                value="单模型",
                label="推理模式",
                interactive=True,
            )

            # ============ 单模型区域 ============
            with gr.Column(visible=True) as single_model_col:
                with gr.Group():
                    gr.Markdown("### 第一步：加载模型文件(可手动填写路径，留空使用默认模型)")
                    with gr.Row():
                        gpt_path = gr.Dropdown(label="GPT 模型路径 (.ckpt)", choices=[''], value='', allow_custom_value=True, scale=1)
                        sovits_path = gr.Dropdown(label="SoVITS 模型路径 (.pth)", choices=[''], value='', allow_custom_value=True, scale=1)

            # ============ 多角色区域 ============
            with gr.Column(visible=False) as multi_model_col:
                with gr.Group():
                    gr.Markdown("### 多角色管理（每个角色的音色/风格参考在添加时单独配置）")

                    # ── Add speaker form ──
                    with gr.Row():
                        multi_name = gr.Textbox(label="角色名", placeholder="例如: alice", scale=1)
                        multi_gpt = gr.Textbox(label="GPT 模型 (.ckpt)", placeholder="路径或留空扫描", scale=2)
                        multi_sovits = gr.Textbox(label="SoVITS 模型 (.pth)", placeholder="路径或留空扫描", scale=2)

                    with gr.Row():
                        multi_spk_audio = gr.Audio(label="音色参考音频", type="filepath", scale=2)
                        multi_prompt_audio = gr.Audio(label="风格参考音频 (可选)", type="filepath", scale=2)
                        with gr.Column(scale=1):
                            multi_prompt_text = gr.Textbox(label="风格参考文本", placeholder="可选", lines=2)
                            multi_add_btn = gr.Button("➕ 添加角色", variant="secondary", size="sm")

                    # ── Speaker list + controls ──
                    with gr.Row():
                        multi_table = gr.Dataframe(
                            headers=["角色名", "GPT 模型", "SoVITS 模型", "模式"],
                            label="已加载角色",
                            value=[],
                            interactive=False,
                            scale=3,
                        )
                        with gr.Column(scale=1):
                            multi_cur_speaker = gr.Dropdown(
                                label="当前发言角色",
                                choices=[],
                                interactive=True,
                                info="未使用 <speaker:> 标签时的默认角色",
                            )
                            multi_remove_name = gr.Dropdown(
                                label="移除角色",
                                choices=[],
                                interactive=True,
                            )
                            multi_remove_btn = gr.Button("➖ 移除", variant="stop", size="sm")

            # Remove old multi_log textbox - status goes to main log

            # ============ 文本输入（共享） ============
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 第二步：合成内容")
                    text = gr.Textbox(
                        label="合成目标文本",
                        lines=5,
                        value="谁罕见?啊？骂谁罕见！",
                        info="多角色模式支持 <speaker:角色名>文本</speaker:角色名> 标签",
                    )
                    enable_enhance = gr.Checkbox(label="启用音频增强", value=False)

                    with gr.Accordion("生成参数", open=False):
                        speed = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="语速")
                        noise_scale = gr.Slider(0.1, 1.0, 0.5, step=0.05, label="噪声比例")
                        temperature = gr.Slider(0.1, 1.5, 1.0, label="温度")
                        top_k = gr.Slider(1, 50, 15, step=1, label="Top K")
                        top_p = gr.Slider(0.1, 1.0, 1.0, label="Top P")
                        rep_penalty = gr.Slider(1.0, 2.0, 1.35, label="重复惩罚")
                        sovits_batch_size = gr.Number(label="SoVITS最大并行推理大小", value=10)
                        text_language = gr.Dropdown(
                            choices=["auto", "ja", "zh", "en"],
                            value="auto",
                            label="目标文本语言",
                            info="auto 自动检测；混语文本建议手动指定",
                        )
                        prompt_language = gr.Dropdown(
                            choices=["auto", "ja", "zh", "en"],
                            value="auto",
                            label="参考音频文本语言",
                        )
                        is_cut_text = gr.Checkbox(label="是否切分文本", value=True)
                        cut_minlen = gr.Number(label="最小切分长度", value=10)
                        cut_mute = gr.Number(label="切分静音时长(s)", value=0.3)
                        cut_mute_scale_map = gr.Textbox(label="标点静音缩放映射", value='{"…": 2.0, ".": 1.5, "。": 1.5, "?": 1.5, "？": 1.5, "!": 1.5, "！": 1.5, ",": 1.0, "，": 1.0, ":": 1.0, "：": 1.0, ";": 1.0, "；": 1.0, "~": 1.0, "、": 0.8, "・": 0.8}')

                with gr.Column(scale=1, visible=True) as ref_audio_col:
                    gr.Markdown("### 第三步：风格与音色参考")
                    with gr.Row():
                        preset_dropdown = gr.Dropdown(choices=get_preset_names(), label="加载预设", scale=2)
                        preset_name = gr.Textbox(label="预设名称", placeholder="保存当前设置为...", scale=2)
                        save_btn = gr.Button("💾 保存预设", scale=1)

                    with gr.Tab("风格参考"):
                        prompt_audio = gr.Audio(label="风格参考音频 (决定语气、情感)", type="filepath")
                        prompt_text = gr.Textbox(label="风格参考音频对应文本", placeholder="输入参考音频中的文本内容")

                    with gr.Tab("音色参考（支持多音色融合）"):
                        multi_spk_files = gr.File(label="可上传多个音色参考音频", file_count="multiple")
                        spk_weights = gr.Textbox(label="音色权重 (用冒号分隔)", value="1.0", placeholder="例如: 1.0: 1.0")

            # ============ 推理输出（共享） ============
            with gr.Group():
                with gr.Row():
                    btn = gr.Button("🔥 开始语音合成", variant="primary", size="lg", scale=3)
                    stream_btn = gr.Button("⚡ 流式合成", variant="secondary", size="lg", scale=1)
                with gr.Row():
                    with gr.Column(scale=2):
                        output_audio = gr.Audio(label="生成的音频结果")
                        log_output = gr.Textbox(label="系统状态信息")

                    with gr.Column(scale=1):
                        gr.Markdown("### 🕒 最近生成历史")
                        history_display = gr.Dataset(
                            components=[gr.Textbox(visible=False)],
                            label="点击下方条目可重新加载音频",
                            samples=[],
                            type="values"
                        )

        with gr.TabItem("音色迁移 (VC)"):
            gr.Markdown("### 将一段音频的内容迁移到另一个人的音色上")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 1. 源音频参考")
                    vc_source_audio = gr.Audio(label="上传源音频", type="filepath")
                    vc_source_text = gr.Textbox(label="源音频对应文本", placeholder="输入源音频中的文本内容", lines=2)

                    gr.Markdown("#### 2. 目标音色参考（支持多音色融合）")
                    vc_multi_spk_files = gr.File(label="可上传多个音色参考音频", file_count="multiple")
                    vc_spk_weights = gr.Textbox(label="音色权重 (用冒号分隔)", value="1.0", placeholder="例如: 1.0: 1.0")

                with gr.Column(scale=1):
                    gr.Markdown("#### 3. 执行与输出")
                    vc_btn = gr.Button("🚀 开始音色迁移", variant="primary", size="lg")

                    vc_output_audio = gr.Audio(label="音色迁移结果", interactive=False)
                    vc_log_output = gr.Textbox(label="处理日志", lines=5)

    def update_history(history_entry, current_history):
        if history_entry is None:
            return current_history, gr.update(samples=current_history)

        current_history.insert(0, history_entry)
        current_history = current_history[:10]

        return current_history, gr.update(samples=current_history)

    def load_from_history(selected_row_data):
        if selected_row_data and len(selected_row_data) > 0:
            audio_path = selected_row_data[-1] 
            return audio_path
        return None

    def toggle_mode(mode):
        """Show/hide sections based on mode."""
        is_single = mode == "单模型"
        return (
            gr.update(visible=is_single),    # single_model_col
            gr.update(visible=not is_single), # multi_model_col
            gr.update(visible=is_single),    # ref_audio_col
        )

    # ── Mode toggle ──
    tts_mode.change(
        fn=toggle_mode,
        inputs=[tts_mode],
        outputs=[single_model_col, multi_model_col, ref_audio_col],
    )

    save_btn.click(
        fn=save_preset,
        inputs=[preset_name, prompt_audio, prompt_text, multi_spk_files, spk_weights],
        outputs=[preset_dropdown, log_output]
    )

    preset_dropdown.change(
        fn=load_preset,
        inputs=[preset_dropdown],
        outputs=[prompt_audio, prompt_text, multi_spk_files, spk_weights]
    )
    preset_dropdown.focus(
        fn=refresh_preset_dropdown,
        outputs=[preset_dropdown]
    )

    multi_spk_files.change(
        fn=update_spk_weights,
        inputs=[multi_spk_files, spk_weights],
        outputs=spk_weights
    )

    vc_multi_spk_files.change(
        fn=update_spk_weights,
        inputs=[vc_multi_spk_files, vc_spk_weights],
        outputs=vc_spk_weights
    )

    prompt_audio.change(
        fn=audio_transcriber,
        inputs=prompt_audio,
        outputs=prompt_text
    ).then(
        fn=auto_fill_speaker_ref,
        inputs=[prompt_audio, multi_spk_files],
        outputs=multi_spk_files
    )

    vc_source_audio.change(
        fn=audio_transcriber,
        inputs=vc_source_audio,
        outputs=vc_source_text
    )

    gpt_path.change(
        fn=upload_gpt,
        inputs=gpt_path
    )
    gpt_path.focus(
        fn=find_gsv_models,
        outputs=[gpt_path, sovits_path]
    )

    sovits_path.change(
        fn=upload_sovits,
        inputs=sovits_path
    )    
    sovits_path.focus(
        fn=find_gsv_models,
        outputs=[gpt_path, sovits_path]
    )

    temp_history_entry = gr.State()

    btn.click(
        fn=tts_request,
        inputs=[
            multi_spk_files, spk_weights,
            prompt_audio, prompt_text,
            text,
            top_k, top_p, temperature, rep_penalty, noise_scale, speed,
            enable_enhance,
            is_cut_text, cut_minlen, cut_mute, cut_mute_scale_map,
            sovits_batch_size,
            text_language, prompt_language,
            tts_mode, multi_cur_speaker,
        ],
        outputs=[output_audio, log_output, temp_history_entry]
    ).then(
        fn=update_history,
        inputs=[temp_history_entry, history_state],
        outputs=[history_state, history_display]
    )

    stream_btn.click(
        fn=tts_stream_request,
        inputs=[
            multi_spk_files, spk_weights,
            prompt_audio, prompt_text,
            text,
            top_k, top_p, temperature, rep_penalty, noise_scale, speed,
            enable_enhance,
            text_language, prompt_language,
            tts_mode, multi_cur_speaker,
        ],
        outputs=[output_audio, log_output, temp_history_entry]
    ).then(
        fn=update_history,
        inputs=[temp_history_entry, history_state],
        outputs=[history_state, history_display]
    )

    vc_btn.click(
        fn=vc_request,
        inputs=[
            vc_multi_spk_files, vc_spk_weights,
            vc_source_audio, vc_source_text,
        ],
        outputs=[vc_output_audio, vc_log_output]
    )

    history_display.click(
        fn=load_from_history,
        inputs=[history_display],
        outputs=[output_audio]
    )

    # ── Multi-speaker management events ──
    def _refresh_multi_ui():
        return _get_speaker_choices(), _get_remove_choices()

    multi_add_btn.click(
        fn=multi_add_speaker,
        inputs=[multi_name, multi_gpt, multi_sovits,
                multi_spk_audio, multi_prompt_audio, multi_prompt_text],
        outputs=[multi_table, log_output],
    ).then(
        fn=_refresh_multi_ui,
        inputs=[],
        outputs=[multi_cur_speaker, multi_remove_name],
    )

    multi_remove_btn.click(
        fn=multi_remove_speaker,
        inputs=[multi_remove_name],
        outputs=[multi_table, log_output],
    ).then(
        fn=_refresh_multi_ui,
        inputs=[],
        outputs=[multi_cur_speaker, multi_remove_name],
    )


if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() == 'true':
            return True
        elif v.lower() == 'false':
            return False
        
    parser = argparse.ArgumentParser(description="GSV-TTS")
    parser.add_argument("--gpt_cache_len", type=int, default=1024, help="GPT KV cache 上下文长度")
    parser.add_argument("--gpt_batch_size", type=int, default=8, help="GPT 最大并行推理大小")
    parser.add_argument("--use_bert", type=str2bool, default=True, help="使用BERT提升中文语义理解能力")
    parser.add_argument("--use_flash_attn", type=str2bool, default=True, help="使用Flash Attn加速推理")
    parser.add_argument("--use_asr", type=str2bool, default=False, help="使用ASR自动识别音频文本")
    parser.add_argument("--models_dir", type=str, help="预训练模型目录")
    parser.add_argument("--port", type=int, default=9881, help="Gradio 端口号")
    parser.add_argument("--share", action="store_true", help="是否开启公网分享")
    parser.add_argument("--gsv_root_dir", type=str, default=str(project_root), help="原版GSV根目录，用于自动扫描模型（默认：仓库根）")
    
    args, _ = parser.parse_known_args()

    GSV_ROOT_DIR = args.gsv_root_dir
    USE_BERT = args.use_bert
    PRESETS_DIR.mkdir(exist_ok=True)
    HISTORY_DIR.mkdir(exist_ok=True)

    if args.models_dir is None:
        args.models_dir = Path.home() / ".cache" / "gsv"

    if args.use_asr:
        from qwen_asr import Qwen3ASRModel

        local_model_path = args.models_dir / "qwen3_asr"
        
        # 可改1.7B
        repo_id = "Qwen/Qwen3-ASR-0.6B"

        # 检查本地是否已有
        if not (local_model_path.exists() and (local_model_path / "config.json").exists()):
            print(f"⬇️ 本地未找到模型，正在从 Hugging Face 下载: {repo_id}")
            print(f"📂 保存路径: {local_model_path}")
            
            try:
               
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_model_path),
                    local_dir_use_symlinks=False,
                    # 下载慢的话可以用下面这个镜像
                    # endpoint="https://hf-mirror.com" 
                )
                print("✅ 模型下载完成！")
            except Exception as e:
                print(f"❌ 下载失败: {e}")
                print("💡 可以用https://hf-mirror.com镜像尝试")
                raise e
        else:
            print(f"✅ 检测到本地模型已存在: {local_model_path}")

        # 4. 加载模型 (始终使用绝对路径)
        print(f"🚀 正在加载 ASR 模型...")
        use_cuda = torch.cuda.is_available()
        asr = Qwen3ASRModel.from_pretrained(
            str(local_model_path),  # 传入绝对路径字符串
            dtype=torch.bfloat16 if use_cuda else torch.float32,
            device_map="cuda:0" if use_cuda else "cpu",
            attn_implementation="flash_attention_2" if (args.use_flash_attn and use_cuda) else None,
            local_files_only=True,
        )
    else:
        asr = None




    batch_sizes = [1] + list(range(4, args.gpt_batch_size, 4)) + [args.gpt_batch_size]
    cache_lens = []
    length = 512
    while length <= args.gpt_cache_len:
        cache_lens.append(length)
        length *= 2
    
    gpt_cache = [(b, c) for b in batch_sizes for c in cache_lens]

    tts = TTS(
        gpt_cache=gpt_cache,
        sovits_cache=[],
        use_bert=args.use_bert,
        use_flash_attn=args.use_flash_attn,
        models_dir=args.models_dir,
    )
    
    demo.queue().launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )
