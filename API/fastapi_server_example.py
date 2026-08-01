"""
FastAPI 服务端使用示例
展示如何使用异步 TTS 接口处理并发请求
支持外链音频URL和ASR自动识别
"""

import sys
import asyncio
import json
import base64
from pathlib import Path
from contextlib import asynccontextmanager

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Union
from gsv_tts import TTS, MultiSpeakerTTS, SpeakerConfig, ConfigMismatchError
import uuid
import os
import tempfile
import logging

models_dir = project_root / "API" / "models"
output_dir = project_root / "output"
output_dir.mkdir(exist_ok=True)
AUDIO_DIR = output_dir.resolve()

tts: Optional[TTS] = None
multi_tts: Optional[MultiSpeakerTTS] = None
asr = None

temp_dir = tempfile.mkdtemp(prefix="gsv_tts_")


def is_url(path: str) -> bool:
    """检查是否为URL"""
    return path.startswith("http://") or path.startswith("https://")


async def download_audio(url: str) -> str:
    """下载音频URL到临时文件"""
    import httpx
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
    
    ext = ".wav"
    content_type = response.headers.get("content-type", "")
    if "mp3" in content_type or url.lower().endswith(".mp3"):
        ext = ".mp3"
    elif "ogg" in content_type or url.lower().endswith(".ogg"):
        ext = ".ogg"
    elif "flac" in content_type or url.lower().endswith(".flac"):
        ext = ".flac"
    
    temp_path = os.path.join(temp_dir, f"download_{uuid.uuid4().hex}{ext}")
    with open(temp_path, "wb") as f:
        f.write(response.content)
    
    logging.info(f"下载音频到: {temp_path}")
    return temp_path


def transcribe_audio(audio_path: str) -> str:
    """使用ASR识别音频文本"""
    global asr
    if asr is None:
        raise HTTPException(status_code=500, detail="ASR模型未启用，请设置 --use_asr 或提供 prompt_text")
    
    results = asr.transcribe(audio_path)
    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'text'):
            text = result.text
        elif isinstance(result, dict):
            text = result.get("text", "")
        else:
            text = str(result)
        logging.info(f"ASR识别结果: {text}")
        return text
    return ""


class TTSSingleRequest(BaseModel):
    text: str
    speaker_audio: str
    prompt_audio: str
    prompt_text: Optional[str] = None
    text_language: str = "auto"
    prompt_language: str = "auto"
    top_k: int = 5
    top_p: float = 0.9
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    noise_scale: float = 0.5
    speed: float = 1.0


class TTSBatchRequest(BaseModel):
    texts: List[str]
    speaker_audio: str
    prompt_audio: str
    prompt_text: Optional[str] = None
    text_languages: Union[str, List[str]] = "auto"
    prompt_languages: Union[str, List[str]] = "auto"
    top_k: int = 5
    top_p: float = 0.9
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    noise_scale: float = 0.5
    speed: float = 1.0


# ============================================================
# Multi-Speaker API models
# ============================================================

class MultiInitRequest(BaseModel):
    base_gpt_path: Optional[str] = None
    base_sovits_path: Optional[str] = None
    use_bert: bool = True
    use_flash_attn: bool = False


class MultiAddSpeakerRequest(BaseModel):
    name: str
    gpt_model_path: str
    sovits_model_path: str
    speaker_audio: str
    prompt_audio: Optional[str] = None
    prompt_text: Optional[str] = None


class MultiRemoveSpeakerRequest(BaseModel):
    name: str


class MultiInferRequest(BaseModel):
    speaker: str
    text: str
    text_language: str = "auto"
    prompt_language: str = "auto"
    prompt_audio_path: Optional[str] = None
    prompt_audio_text: Optional[str] = None
    top_k: int = 5
    top_p: float = 0.9
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    noise_scale: float = 0.5
    speed: float = 1.0


class MultiBatchRequest(BaseModel):
    speaker_texts: List[MultiInferRequest]
    """List of (speaker, text) pairs for multi-speaker batch inference."""


class MultiStreamRequest(BaseModel):
    speaker: str
    text: str
    text_language: str = "auto"
    prompt_language: str = "auto"
    prompt_audio_path: Optional[str] = None
    prompt_audio_text: Optional[str] = None
    top_k: int = 5
    top_p: float = 0.9
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    noise_scale: float = 0.5
    speed: float = 1.0
    stream_chunk: int = 25
    overlap_len: int = 5
    boost_first_chunk: bool = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle for the FastAPI application."""
    global tts, asr
    print("🚀 正在加载 TTS 模型...")
    
    max_cache_len = 1024
    batch_sizes = [1, 4, 8]
    cache_lens = []
    length = 512
    while length <= max_cache_len:
        cache_lens.append(length)
        length *= 2
    gpt_cache = [(b, c) for b in batch_sizes for c in cache_lens]
    
    tts = TTS(
        models_dir=str(models_dir),
        gpt_cache=gpt_cache,
        sovits_cache=[50],
    )
    print("✅ TTS 模型加载完成！")
    
    use_asr = os.environ.get("USE_ASR", "true").lower() == "true"
    if use_asr:
        try:
            import torch
            from huggingface_hub import snapshot_download
            
            local_model_path = models_dir / "qwen3_asr"
            repo_id = "Qwen/Qwen3-ASR-0.6B"
            
            if not (local_model_path.exists() and (local_model_path / "config.json").exists()):
                print(f"⬇️ 本地未找到ASR模型，正在下载: {repo_id}")
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_model_path),
                    local_dir_use_symlinks=False,
                )
                print("✅ ASR模型下载完成！")
            
            from qwen_asr import Qwen3ASRModel
            print("🚀 正在加载 ASR 模型...")
            asr = Qwen3ASRModel.from_pretrained(
                str(local_model_path),
                dtype=torch.bfloat16,
                device_map="cuda:0",
                local_files_only=True
            )
            print("✅ ASR 模型加载完成！")
        except Exception as e:
            print(f"⚠️ ASR 模型加载失败: {e}")
            print("💡 提示：如果没有提供 prompt_text，请求将会失败")
            asr = None
    else:
        print("ℹ️ ASR 模型已禁用")
    
    yield
    # Shutdown: cleanup resources if needed


app = FastAPI(title="GSV-TTS 异步 API", version="1.1", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "message": "GSV-TTS 异步 API 服务已启动",
        "docs": "/docs",
        "features": {
            "url_support": True,
            "auto_asr": asr is not None
        }
    }


@app.post("/tts/single")
async def tts_single(request: TTSSingleRequest):
    """单个 TTS 请求的异步接口，支持外链音频和自动ASR"""
    try:
        speaker_audio = request.speaker_audio
        prompt_audio = request.prompt_audio
        prompt_text = request.prompt_text
        
        if is_url(speaker_audio):
            speaker_audio = await download_audio(speaker_audio)
        
        if is_url(prompt_audio):
            prompt_audio = await download_audio(prompt_audio)
        
        if prompt_text is None or prompt_text == "":
            prompt_text = transcribe_audio(prompt_audio)
            if not prompt_text:
                raise HTTPException(
                    status_code=400, 
                    detail="无法自动识别prompt_audio文本，请手动提供prompt_text"
                )
        
        audio_clip = await tts.infer_async(
            spk_audio_path=speaker_audio,
            prompt_audio_path=prompt_audio,
            prompt_audio_text=prompt_text,
            text=request.text,
            text_language=request.text_language,
            prompt_language=request.prompt_language,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            noise_scale=request.noise_scale,
            speed=request.speed,
        )
        
        output_filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
        output_path = output_dir / output_filename
        audio_clip.save(str(output_path))
        
        return {
            "success": True,
            "audio_len": audio_clip.audio_len_s,
            "filename": output_filename,
            "prompt_text_used": prompt_text,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/batch")
async def tts_batch(request: TTSBatchRequest):
    """批量 TTS 请求的异步接口，支持外链音频和自动ASR"""
    try:
        speaker_audio = request.speaker_audio
        prompt_audio = request.prompt_audio
        prompt_text = request.prompt_text
        
        if is_url(speaker_audio):
            speaker_audio = await download_audio(speaker_audio)
        
        if is_url(prompt_audio):
            prompt_audio = await download_audio(prompt_audio)
        
        if prompt_text is None or prompt_text == "":
            prompt_text = transcribe_audio(prompt_audio)
            if not prompt_text:
                raise HTTPException(
                    status_code=400, 
                    detail="无法自动识别prompt_audio文本，请手动提供prompt_text"
                )
        
        audio_clips = await tts.infer_batched_async(
            spk_audio_paths=speaker_audio,
            prompt_audio_paths=prompt_audio,
            prompt_audio_texts=prompt_text,
            texts=request.texts,
            text_languages=request.text_languages,
            prompt_languages=request.prompt_languages,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            noise_scale=request.noise_scale,
            speed=request.speed,
        )
        
        filenames = []
        for clip in audio_clips:
            filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
            output_path = output_dir / filename
            clip.save(str(output_path))
            filenames.append(filename)
        
        return {
            "success": True,
            "count": len(audio_clips),
            "filenames": filenames,
            "prompt_text_used": prompt_text,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """获取生成的音频文件"""
    file_path = (AUDIO_DIR / filename).resolve()
    if not str(file_path).startswith(str(AUDIO_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件未找到")
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")
    return FileResponse(str(file_path), media_type="audio/wav")


# ============================================================
# Multi-Speaker API endpoints
# ============================================================

@app.post("/multi-speaker/init")
async def multi_init(request: MultiInitRequest):
    """Initialize MultiSpeakerTTS engine with shared backbone."""
    global multi_tts
    try:
        kwargs = {
            "use_bert": request.use_bert,
            "use_flash_attn": request.use_flash_attn,
        }
        if request.base_gpt_path:
            kwargs["base_gpt_path"] = request.base_gpt_path
        if request.base_sovits_path:
            kwargs["base_sovits_path"] = request.base_sovits_path

        multi_tts = MultiSpeakerTTS(speakers=[], **kwargs)
        return {"success": True, "message": "MultiSpeakerTTS engine initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/multi-speaker/add")
async def multi_add(request: MultiAddSpeakerRequest):
    """Add a speaker to the MultiSpeakerTTS engine."""
    global multi_tts
    if multi_tts is None:
        raise HTTPException(status_code=400, detail="MultiSpeakerTTS not initialized. Call /multi-speaker/init first.")

    try:
        speaker_audio = request.speaker_audio
        prompt_audio = request.prompt_audio
        prompt_text = request.prompt_text

        if is_url(speaker_audio):
            speaker_audio = await download_audio(speaker_audio)
        if prompt_audio and is_url(prompt_audio):
            prompt_audio = await download_audio(prompt_audio)

        spk = SpeakerConfig(
            name=request.name,
            gpt_model_path=request.gpt_model_path,
            sovits_model_path=request.sovits_model_path,
            spk_audio_path=speaker_audio,
            prompt_audio_path=prompt_audio or speaker_audio,
            prompt_audio_text=prompt_text,
        )
        multi_tts.add_speaker(spk)
        w = multi_tts._speakers[request.name]
        mode = "shared" if not w.is_full_model else "full_model_degraded"

        return {
            "success": True,
            "name": request.name,
            "mode": mode,
            "message": f"Speaker '{request.name}' added ({mode})",
        }
    except ConfigMismatchError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Config mismatch (speaker loaded as full model): {e}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/multi-speaker/remove")
async def multi_remove(request: MultiRemoveSpeakerRequest):
    """Remove a speaker from the MultiSpeakerTTS engine."""
    global multi_tts
    if multi_tts is None:
        raise HTTPException(status_code=400, detail="Not initialized")

    try:
        multi_tts.remove_speaker(request.name)
        return {"success": True, "message": f"Speaker '{request.name}' removed"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/multi-speaker/list")
async def multi_list():
    """List all loaded speakers with their mode."""
    global multi_tts
    if multi_tts is None:
        return {"initialized": False, "speakers": []}

    speakers = []
    for name in multi_tts.speaker_names:
        w = multi_tts._speakers[name]
        speakers.append({
            "name": name,
            "mode": "full_model" if w.is_full_model else "shared",
            "gpt_keys": len(w.gpt_weights) if not w.is_full_model else 0,
            "sovits_keys": len(w.sovits_weights) if not w.is_full_model else 0,
        })

    return {"initialized": True, "speakers": speakers}


@app.post("/multi-speaker/infer")
async def multi_infer(request: MultiInferRequest):
    """Single-speaker inference via MultiSpeakerTTS."""
    global multi_tts
    if multi_tts is None:
        raise HTTPException(status_code=400, detail="Not initialized")

    try:
        audio_clip = multi_tts.infer(
            speaker=request.speaker,
            text=request.text,
            prompt_audio_path=request.prompt_audio_path,
            prompt_audio_text=request.prompt_audio_text,
            text_language=request.text_language,
            prompt_language=request.prompt_language,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            noise_scale=request.noise_scale,
            speed=request.speed,
        )

        filename = f"multi_{uuid.uuid4().hex[:8]}.wav"
        output_path = output_dir / filename
        audio_clip.save(str(output_path))

        return {
            "success": True,
            "speaker": request.speaker,
            "audio_len": audio_clip.audio_len_s,
            "filename": filename,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/multi-speaker/batch")
async def multi_batch(request: MultiBatchRequest):
    """Multi-speaker batch inference.

    Example body:
    {
        "speaker_texts": [
            {"speaker": "alice", "text": "こんにちは"},
            {"speaker": "bob",   "text": "よろしく"}
        ]
    }
    """
    global multi_tts
    if multi_tts is None:
        raise HTTPException(status_code=400, detail="Not initialized")

    try:
        speaker_texts = [(req.speaker, req.text) for req in request.speaker_texts]

        # 语言参数：所有条目相同则传单个 str，否则逐条传 list
        text_languages = [r.text_language for r in request.speaker_texts]
        prompt_languages = [r.prompt_language for r in request.speaker_texts]
        if len(set(text_languages)) == 1:
            text_languages = text_languages[0]
        if len(set(prompt_languages)) == 1:
            prompt_languages = prompt_languages[0]

        batch_kwargs = {
            "text_languages": text_languages,
            "prompt_languages": prompt_languages,
        }

        # 按次 prompt 覆盖：任一条目提供了覆盖参数时逐条透传（自动退化为逐条推理）
        prompt_paths = [r.prompt_audio_path for r in request.speaker_texts]
        prompt_texts = [r.prompt_audio_text for r in request.speaker_texts]
        if any(prompt_paths) or any(prompt_texts):
            batch_kwargs["prompt_audio_paths"] = prompt_paths
            batch_kwargs["prompt_audio_texts"] = prompt_texts

        audio_clips = multi_tts.infer_batched(speaker_texts, **batch_kwargs)

        filenames = []
        for clip in audio_clips:
            filename = f"multi_{uuid.uuid4().hex[:8]}.wav"
            output_path = output_dir / filename
            clip.save(str(output_path))
            filenames.append(filename)

        return {
            "success": True,
            "count": len(audio_clips),
            "filenames": filenames,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/multi-speaker/stream")
async def multi_stream(request: MultiStreamRequest):
    """Multi-speaker streaming inference via SSE.

    Yields `event: audio` chunks (base64 WAV), then `event: done` with the
    total duration, or `event: error` on failure.
    """
    global multi_tts
    if multi_tts is None:
        raise HTTPException(status_code=400, detail="Not initialized")

    async def generate():
        try:
            loop = asyncio.get_running_loop()
            chunk_queue: asyncio.Queue = asyncio.Queue()

            def stream_infer():
                try:
                    for clip in multi_tts.infer_stream(
                        speaker=request.speaker,
                        text=request.text,
                        prompt_audio_path=request.prompt_audio_path,
                        prompt_audio_text=request.prompt_audio_text,
                        text_language=request.text_language,
                        prompt_language=request.prompt_language,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        temperature=request.temperature,
                        repetition_penalty=request.repetition_penalty,
                        noise_scale=request.noise_scale,
                        speed=request.speed,
                        stream_chunk=request.stream_chunk,
                        overlap_len=request.overlap_len,
                        boost_first_chunk=request.boost_first_chunk,
                        debug=False,
                    ):
                        loop.call_soon_threadsafe(chunk_queue.put_nowait, clip)
                except Exception as e:
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, e)
                finally:
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, None)

            loop.run_in_executor(None, stream_infer)

            total_len = 0
            while True:
                clip = await chunk_queue.get()
                if clip is None:
                    break
                if isinstance(clip, Exception):
                    raise clip
                audio_b64 = base64.b64encode(clip.audio_data.tobytes()).decode("utf-8")
                total_len += len(clip.audio_data)
                chunk_data = {
                    "audio": audio_b64,
                    "sample_rate": clip.samplerate,
                    "duration": clip.audio_len_s,
                    "text": clip.orig_text,
                }
                yield f"event: audio\ndata: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

            yield f"event: done\ndata: {json.dumps({'total_duration': total_len / 32000}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
