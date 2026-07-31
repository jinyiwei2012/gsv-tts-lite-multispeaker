"""Multi-speaker TTS engine with shared model backbone and per-speaker weights.

Core idea:
- One shared GPT backbone + one shared SoVITS backbone (loaded once at init)
- Each speaker adds only ~5-15% lightweight weights (predict_layer, ref_enc, etc.)
- At inference time, speaker weights are injected via copy_() (CUDA Graph safe)
- All speakers can be used in a single session without reloading models
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
from typing import Generator, Literal

import numpy as np
import torch

from .TTS import TTS
from .SpeakerWeights import SpeakerConfig, SpeakerWeights
from .Loader import (
    extract_speaker_gpt_weights,
    extract_speaker_sovits_weights,
    load_shared_gpt,
    load_shared_sovits,
    _load_gpt_state_dict,
    _load_sovits_state_dict,
)
from .GPT_SoVITS.SV import ERes2Net
from .Player import AudioClip


logger = logging.getLogger(__name__)

_SHARED_GPT_KEY = "__multi_speaker_shared_gpt__"
_SHARED_SOVITS_KEY = "__multi_speaker_shared_sovits__"

# Config keys that must match between base model and speaker models
_GPT_CRITICAL_KEYS = [
    ("model", "hidden_dim"),
    ("model", "embedding_dim"),
    ("model", "head"),
    ("model", "n_layer"),
    ("model", "vocab_size"),
    ("model", "phoneme_vocab_size"),
]

_SOVITS_CRITICAL_KEYS = [
    ("model", "gin_channels"),
    ("model", "inter_channels"),
    ("model", "hidden_channels"),
    ("model", "filter_channels"),
    ("model", "n_heads"),
    ("model", "n_layers"),
    ("model", "upsample_initial_channel"),
    ("model", "version"),
]

# v2Pro and v2ProPlus share identical code paths in is_v2pro/sv_emb/ge_to512/prelu,
# but may differ in structural dims like upsample_initial_channel.
# The explicit structural key checks above catch those cases.
_SOVITS_COMPATIBLE_VERSIONS = frozenset({"v2Pro", "v2ProPlus"})


class ConfigMismatchError(ValueError):
    """Raised when a speaker model config is incompatible with the base model.

    All speakers must share identical GPT architecture (n_layer, model_dim,
    vocab_size, etc.) and SoVITS architecture (gin_channels, version, etc.)
    for weight injection via copy_() to be safe.
    """
    pass


def _resolve_param(model: torch.nn.Module, param_path: str) -> torch.nn.Parameter:
    """Resolve a dotted parameter path (e.g. 't2s_transformer.blocks.0.qkv.weight')
    to the actual nn.Parameter in the model tree."""
    parts = param_path.split(".")
    obj = model
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    if not isinstance(obj, torch.nn.Parameter):
        raise TypeError(f"Expected nn.Parameter at '{param_path}', got {type(obj)}")
    return obj


_SCRIPT_INLINE_TAG_RE = re.compile(r"<speaker:([^>]+)>(.*?)</speaker>", re.DOTALL)
_SCRIPT_LINE_RE = re.compile(r"^([^:：]{1,32})[:：]\s*(.+)$", re.DOTALL)
_SCRIPT_EMPTY_LINE_RE = re.compile(r"^[^:：]{1,32}[:：]$")


def split_speaker_text(text: str, default_speaker: str) -> list[tuple[str, str]]:
    """Split text into [(speaker, segment), ...] honoring <speaker:name> tags.

    Plain text outside tags uses default_speaker. Empty segments are dropped.
    Used by streaming synthesis (WebUI) to mix speakers per segment.
    """
    segments: list[tuple[str, str]] = []
    pos = 0
    for m in _SCRIPT_INLINE_TAG_RE.finditer(text):
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


def parse_script(script: str) -> list[tuple[str, str]]:
    """Parse a dialogue script into an ordered list of (speaker, text).

    Supported formats:
      - ``speaker: text`` lines (full-width ``：`` also accepted)
      - Inline tags: ``<speaker:name>text</speaker>`` (each tag becomes an entry)

    Raises:
        ValueError: On a non-empty line that matches neither format.
    """
    entries: list[tuple[str, str]] = []
    for raw_line in script.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "<speaker:" in line:
            matched = False
            for m in _SCRIPT_INLINE_TAG_RE.finditer(line):
                seg = m.group(2).strip()
                if seg:
                    entries.append((m.group(1).strip(), seg))
                    matched = True
            if matched:
                continue
        m = _SCRIPT_LINE_RE.match(line)
        if m:
            text = m.group(2).strip()
            if text:
                entries.append((m.group(1).strip(), text))
            continue
        # 形如 "alice:" 的空台词行 → 跳过
        if _SCRIPT_EMPTY_LINE_RE.match(line):
            continue
        raise ValueError(
            f"Cannot parse script line: {raw_line!r} "
            "(expected 'speaker: text' or <speaker:name>text</speaker>)"
        )
    return entries


class MultiSpeakerTTS:
    """Multi-speaker TTS engine — shared backbone + per-speaker lightweight weights.

    Usage:
        speakers = [
            SpeakerConfig(name="alice", gpt_model_path="alice_gpt.ckpt",
                          sovits_model_path="alice_sovits.pth",
                          spk_audio_path="alice_ref.wav"),
            SpeakerConfig(name="bob",   gpt_model_path="bob_gpt.ckpt",
                          sovits_model_path="bob_sovits.pth",
                          spk_audio_path="bob_ref.wav"),
        ]
        tts = MultiSpeakerTTS(speakers=speakers, use_bert=True)
        audio = tts.infer("alice", "Hello world!")
        audio.play()
        tts.audio_queue.wait()
    """

    def __init__(
        self,
        speakers: list[SpeakerConfig],
        base_gpt_path: str | None = None,
        base_sovits_path: str | None = None,
        models_dir: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        use_flash_attn: bool = False,
        use_bert: bool = False,
        shared_gpt_layers: int | None = None,
        gpt_cache: list[tuple[int, int]] | None = None,
        sovits_cache: list[int] | None = None,
    ):
        """Initialize multi-speaker TTS engine.

        Args:
            speakers: List of speaker configurations. Each speaker gets its own
                      fine-tuned prediction head and speaker embedding components.
            base_gpt_path: Path to the base GPT model for shared backbone.
                           Defaults to ~/.cache/gsv/s1v3.ckpt.
            base_sovits_path: Path to the base SoVITS model for shared backbone.
                              Defaults to ~/.cache/gsv/s2Gv2ProPlus.pth.
            models_dir: Override models directory.
            device: Override device (cuda/mps/cpu).
            dtype: Override dtype (float32/float16/bfloat16).
            use_flash_attn: Enable Flash Attention for GPT.
            use_bert: Pre-load Chinese BERT at init.
            shared_gpt_layers: Number of GPT transformer layers to share.
                               None = n_layer - 2 (keep last 2 per speaker).
            gpt_cache: Override GPT CUDA Graph static cache sizes.
            sovits_cache: Override SoVITS CUDA Graph static cache sizes.
        """
        if not speakers:
            raise ValueError("At least one SpeakerConfig is required.")

        # ── Create underlying TTS instance (shared infrastructure) ──
        tts_kwargs = {
            "models_dir": models_dir,
            "device": device,
            "dtype": dtype,
            "use_flash_attn": use_flash_attn,
            "use_bert": use_bert,
        }
        if gpt_cache is not None:
            tts_kwargs["gpt_cache"] = gpt_cache
        if sovits_cache is not None:
            tts_kwargs["sovits_cache"] = sovits_cache
        self._tts = TTS(**{k: v for k, v in tts_kwargs.items() if v is not None})

        # ── Load shared backbones ──
        if base_gpt_path is None:
            base_gpt_path = str(Path(self._tts.models_dir) / "s1v3.ckpt")
        if base_sovits_path is None:
            base_sovits_path = str(Path(self._tts.models_dir) / "s2Gv2ProPlus.pth")

        logger.info(f"Loading shared GPT backbone from: {base_gpt_path}")
        self._shared_gpt = load_shared_gpt(
            base_gpt_path, self._tts.tts_config, shared_layers=shared_gpt_layers
        )
        logger.info(f"Loading shared SoVITS backbone from: {base_sovits_path}")
        self._shared_sovits = load_shared_sovits(base_sovits_path, self._tts.tts_config)

        # ── Store base configs for speaker compatibility validation ──
        _, self._base_gpt_config = _load_gpt_state_dict(base_gpt_path)
        _, self._base_sovits_hps = _load_sovits_state_dict(base_sovits_path)

        # ── Extract per-speaker weights ──
        self._speakers: dict[str, SpeakerWeights] = {}
        self._shared_gpt_layers = shared_gpt_layers  # store for logging
        # ── Weight cache: avoid re-extracting when multiple speakers
        #     share the same checkpoint (multi-speaker model files)
        self._weight_cache: dict[tuple[str, str], tuple[dict, dict]] = {}
        #     key = (gpt_path, sovits_path) → (gpt_weights, sovits_weights)

        for i, spk in enumerate(speakers):
            logger.info(
                f"Extracting speaker weights [{i + 1}/{len(speakers)}]: {spk.name}"
            )
            self._add_speaker(spk)

        # ── Active speaker tracking ──
        self._active_speaker: str | None = None

        # ── Expose shared resources ──
        self.audio_queue = self._tts.audio_queue
        self.samplerate = self._tts.samplerate

        logger.info(
            f"MultiSpeakerTTS ready with {len(self._speakers)} speaker(s). "
            f"Shared GPT layers: {self._shared_gpt_layers or 'n_layer - 2'}"
        )

    # ==================================================================
    # Speaker management
    # ==================================================================

    def _validate_config(
        self,
        spk_name: str,
        gpt_config: dict,
        sovits_hps: dict,
    ):
        """Validate speaker configs against the base model.

        Raises ConfigMismatchError if any critical architecture key differs.
        """
        # Validate GPT config
        for section, key in _GPT_CRITICAL_KEYS:
            base_val = self._base_gpt_config[section][key]
            spk_val = gpt_config[section][key]
            if base_val != spk_val:
                raise ConfigMismatchError(
                    f"Speaker '{spk_name}' GPT config mismatch: "
                    f"{section}.{key}={spk_val}, base={base_val}"
                )

        # Validate SoVITS config
        for section, key in _SOVITS_CRITICAL_KEYS:
            base_val = self._base_sovits_hps[section][key]
            spk_val = sovits_hps[section][key]
            if base_val != spk_val:
                # v2Pro ↔ v2ProPlus are architecturally identical
                if (
                    key == "version"
                    and base_val in _SOVITS_COMPATIBLE_VERSIONS
                    and spk_val in _SOVITS_COMPATIBLE_VERSIONS
                ):
                    logger.info(
                        f"Speaker '{spk_name}': version {spk_val} is "
                        f"compatible with base {base_val} (same architecture)"
                    )
                    continue
                raise ConfigMismatchError(
                    f"Speaker '{spk_name}' SoVITS config mismatch: "
                    f"{section}.{key}={spk_val}, base={base_val}"
                )

        logger.debug(f"Config validation passed for speaker '{spk_name}'")

    def _add_speaker(self, spk: SpeakerConfig):
        """Extract and cache speaker-specific weights and features."""
        # ── Validate config compatibility before anything else ──
        _, spk_gpt_config = _load_gpt_state_dict(spk.gpt_model_path)
        _, spk_sovits_hps = _load_sovits_state_dict(spk.sovits_model_path)
        try:
            self._validate_config(spk.name, spk_gpt_config, spk_sovits_hps)
        except ConfigMismatchError as e:
            logger.warning(
                f"Speaker '{spk.name}' config mismatch — "
                f"falling back to full model loading: {e}"
            )
            return self._add_full_model_speaker(spk)

        weights = SpeakerWeights(name=spk.name)
        weights.spk_audio_path = spk.spk_audio_path
        weights.prompt_audio_path = spk.prompt_audio_path or spk.spk_audio_path
        weights.prompt_audio_text = spk.prompt_audio_text

        # Extract model weights from checkpoints (with caching)
        cache_key = (spk.gpt_model_path, spk.sovits_model_path)
        if cache_key in self._weight_cache:
            cached_gpt, cached_sovits = self._weight_cache[cache_key]
            weights.gpt_weights = cached_gpt
            weights.sovits_weights = cached_sovits
            logger.info(
                f"  Speaker '{spk.name}' reusing cached weights "
                f"from checkpoint (shared with another speaker)"
            )
        else:
            weights.gpt_weights = extract_speaker_gpt_weights(
                spk.gpt_model_path,
                self._tts.tts_config,
                shared_layers=self._shared_gpt_layers,
            )
            weights.sovits_weights = extract_speaker_sovits_weights(
                spk.sovits_model_path,
                self._tts.tts_config,
            )
            self._weight_cache[cache_key] = (
                weights.gpt_weights,
                weights.sovits_weights,
            )

        # Pre-compute speaker embedding (ge) via the SoVITS model
        # We load the full SoVITS model temporarily to get ge, then discard it
        self._tts.load_sovits_model(spk.sovits_model_path)
        self._tts.cache_spk_audio(spk.spk_audio_path, sovits_model=spk.sovits_model_path)
        spk_cache = self._tts.spk_audio_cache[spk.spk_audio_path]
        weights.ge = spk_cache["ge"][spk.sovits_model_path]
        weights.sv_emb = spk_cache.get("sv_emb")

        # Pre-compute prompt features (if prompt audio is configured)
        prompt_audio = spk.prompt_audio_path or spk.spk_audio_path
        prompt_text = spk.prompt_audio_text
        if prompt_text is not None:
            self._tts.cache_prompt_audio(
                prompt_audio_paths=prompt_audio,
                prompt_audio_texts=prompt_text,
            )
            prompt_cache = self._tts.prompt_audio_cache[prompt_audio]
            weights.prompt = prompt_cache["prompt"]
            weights.phones1 = prompt_cache["phones1"]
            weights.bert1 = prompt_cache["bert1"]
        else:
            logger.warning(
                f"Speaker '{spk.name}' has no prompt_audio_text — "
                "prompt features will need to be provided at inference time."
            )

        # Clean up: unload the temporarily loaded full model
        self._tts.unload_sovits_model(spk.sovits_model_path)

        self._speakers[spk.name] = weights
        logger.info(
            f"  Speaker '{spk.name}' ready: "
            f"{len(weights.gpt_weights)} GPT keys, "
            f"{len(weights.sovits_weights)} SoVITS keys"
        )

    def _add_full_model_speaker(self, spk: SpeakerConfig):
        """Load full standalone models for an incompatible speaker.

        Used as degradation fallback when a speaker's model config doesn't
        match the base model. Occupies ~800MB VRAM per speaker instead of
        the ~120MB shared-backbone approach.
        """
        weights = SpeakerWeights(
            name=spk.name,
            is_full_model=True,
            spk_audio_path=spk.spk_audio_path,
            prompt_audio_path=spk.prompt_audio_path or spk.spk_audio_path,
            prompt_audio_text=spk.prompt_audio_text,
            gpt_model_key=spk.gpt_model_path,
            sovits_model_key=spk.sovits_model_path,
        )

        # Load full models into the underlying TTS instance
        self._tts.load_gpt_model(spk.gpt_model_path)
        self._tts.load_sovits_model(spk.sovits_model_path)

        # Pre-compute speaker embedding (ge)
        self._tts.cache_spk_audio(
            spk.spk_audio_path, sovits_model=spk.sovits_model_path
        )
        spk_cache = self._tts.spk_audio_cache[spk.spk_audio_path]
        weights.ge = spk_cache["ge"][spk.sovits_model_path]
        weights.sv_emb = spk_cache.get("sv_emb")

        # Pre-compute prompt features (if configured)
        prompt_audio = spk.prompt_audio_path or spk.spk_audio_path
        prompt_text = spk.prompt_audio_text
        if prompt_text is not None:
            self._tts.cache_prompt_audio(
                prompt_audio_paths=prompt_audio,
                prompt_audio_texts=prompt_text,
            )
            prompt_cache = self._tts.prompt_audio_cache[prompt_audio]
            weights.prompt = prompt_cache["prompt"]
            weights.phones1 = prompt_cache["phones1"]
            weights.bert1 = prompt_cache["bert1"]
        else:
            logger.warning(
                f"Speaker '{spk.name}' has no prompt_audio_text — "
                "prompt features will need to be provided at inference time."
            )

        self._speakers[spk.name] = weights
        logger.info(
            f"  Speaker '{spk.name}' ready (full model — "
            f"config incompatible with base)"
        )

    def add_speaker(self, spk: SpeakerConfig):
        """Add a new speaker at runtime."""
        if spk.name in self._speakers:
            raise ValueError(f"Speaker '{spk.name}' already exists.")
        self._add_speaker(spk)

    def remove_speaker(self, name: str):
        """Remove a speaker at runtime."""
        if name not in self._speakers:
            raise ValueError(f"Speaker '{name}' not found.")
        del self._speakers[name]
        if self._active_speaker == name:
            self._active_speaker = None
        logger.info(f"Removed speaker: {name}")

    @property
    def speaker_names(self) -> list[str]:
        """List all registered speaker names."""
        return list(self._speakers.keys())

    def _spk_cache_key(self, speaker: str) -> str:
        return f"__multi_speaker_spk__:{speaker}"

    def _prompt_cache_key(self, speaker: str) -> str:
        return f"__multi_speaker_prompt__:{speaker}"

    def _require_speaker(self, speaker: str) -> SpeakerWeights:
        if speaker not in self._speakers:
            raise ValueError(f"Speaker '{speaker}' not found.")
        return self._speakers[speaker]

    def _register_cached_features(
        self,
        speaker: str,
        require_prompt: bool = True,
    ) -> tuple[str, str | None, str]:
        w = self._require_speaker(speaker)
        if w.ge is None:
            raise ValueError(f"Speaker '{speaker}' has no cached speaker embedding.")
        if require_prompt and (w.prompt is None or w.phones1 is None or w.bert1 is None):
            raise ValueError(
                f"Speaker '{speaker}' has no cached prompt features. "
                "Provide prompt_audio_path + prompt_audio_text, "
                "or set them in SpeakerConfig."
            )

        spk_key = self._spk_cache_key(speaker)
        self._tts.spk_audio_cache[spk_key] = {
            "ge": {_SHARED_SOVITS_KEY: w.ge},
            "sv_emb": w.sv_emb,
        }
        prompt_key = None
        if w.prompt is not None and w.phones1 is not None and w.bert1 is not None:
            prompt_key = self._prompt_cache_key(speaker)
            self._tts.prompt_audio_cache[prompt_key] = {
                "prompt": w.prompt,
                "phones1": w.phones1,
                "bert1": w.bert1,
            }
        return spk_key, prompt_key, w.prompt_audio_text or ""

    @contextmanager
    def _activate_shared_models(self, speaker: str, require_prompt: bool = True):
        """Expose active shared models and cached features to the underlying TTS API."""
        self._require_speaker(speaker)
        had_gpt = _SHARED_GPT_KEY in self._tts.gpt_models
        had_sovits = _SHARED_SOVITS_KEY in self._tts.sovits_models
        had_spk = self._spk_cache_key(speaker) in self._tts.spk_audio_cache
        had_prompt = self._prompt_cache_key(speaker) in self._tts.prompt_audio_cache
        old_gpt = self._tts.gpt_models.get(_SHARED_GPT_KEY)
        old_sovits = self._tts.sovits_models.get(_SHARED_SOVITS_KEY)
        old_spk = self._tts.spk_audio_cache.get(self._spk_cache_key(speaker))
        old_prompt = self._tts.prompt_audio_cache.get(self._prompt_cache_key(speaker))

        try:
            self._apply_speaker(speaker)
            self._tts.gpt_models[_SHARED_GPT_KEY] = self._shared_gpt
            self._tts.sovits_models[_SHARED_SOVITS_KEY] = self._shared_sovits
            spk_key, prompt_key, prompt_text = self._register_cached_features(
                speaker,
                require_prompt=require_prompt,
            )
            yield spk_key, prompt_key, prompt_text
        finally:
            if had_gpt:
                self._tts.gpt_models[_SHARED_GPT_KEY] = old_gpt
            else:
                self._tts.gpt_models.pop(_SHARED_GPT_KEY, None)

            if had_sovits:
                self._tts.sovits_models[_SHARED_SOVITS_KEY] = old_sovits
            else:
                self._tts.sovits_models.pop(_SHARED_SOVITS_KEY, None)

            if had_spk:
                self._tts.spk_audio_cache[self._spk_cache_key(speaker)] = old_spk
            else:
                self._tts.spk_audio_cache.pop(self._spk_cache_key(speaker), None)

            if had_prompt:
                self._tts.prompt_audio_cache[self._prompt_cache_key(speaker)] = old_prompt
            else:
                self._tts.prompt_audio_cache.pop(self._prompt_cache_key(speaker), None)

    # ==================================================================
    # Weight injection
    # ==================================================================

    def _apply_speaker(self, name: str):
        """Inject speaker-specific weights into the shared backbone via copy_().

        This is safe for CUDA Graphs because copy_() modifies tensor values
        in-place without changing memory addresses.

        Skips injection if the new speaker shares the same model checkpoint
        as the currently active speaker (multi-speaker checkpoint optimization).
        """
        if self._active_speaker == name:
            return

        w = self._speakers[name]
        old_w = self._speakers.get(self._active_speaker) if self._active_speaker else None

        # If both speakers share the same model weights, skip injection
        if old_w is not None and w.gpt_weights is old_w.gpt_weights:
            self._active_speaker = name
            return

        device = self._tts.tts_config.device
        dtype = self._tts.tts_config.dtype

        # Inject GPT weights
        for param_path, tensor in w.gpt_weights.items():
            target = _resolve_param(self._shared_gpt.t2s_model, param_path)
            target.data.copy_(tensor.to(device=device, dtype=dtype))

        # Inject SoVITS weights
        for param_path, tensor in w.sovits_weights.items():
            target = _resolve_param(self._shared_sovits.vq_model, param_path)
            target.data.copy_(tensor.to(device=device, dtype=dtype))

        self._active_speaker = name

    # ==================================================================
    # Inference
    # ==================================================================

    def _register_full_model_cache(
        self,
        speaker: str,
        require_prompt: bool = True,
    ) -> tuple[str, str | None, str, str, str]:
        """Register cached features for a full-model speaker.

        Returns (spk_key, prompt_key, prompt_text, gpt_model_key, sovits_model_key).
        """
        w = self._require_speaker(speaker)
        if w.ge is None:
            raise ValueError(f"Speaker '{speaker}' has no cached speaker embedding.")
        if require_prompt and (w.prompt is None or w.phones1 is None or w.bert1 is None):
            raise ValueError(
                f"Speaker '{speaker}' has no cached prompt features. "
                "Set prompt_audio_text in SpeakerConfig."
            )

        spk_key = self._spk_cache_key(speaker)
        self._tts.spk_audio_cache[spk_key] = {
            "ge": {w.sovits_model_key: w.ge},
            "sv_emb": w.sv_emb,
        }

        prompt_key = None
        if w.prompt is not None and w.phones1 is not None and w.bert1 is not None:
            prompt_key = self._prompt_cache_key(speaker)
            self._tts.prompt_audio_cache[prompt_key] = {
                "prompt": w.prompt,
                "phones1": w.phones1,
                "bert1": w.bert1,
            }

        return (
            spk_key,
            prompt_key,
            w.prompt_audio_text or "",
            w.gpt_model_key,
            w.sovits_model_key,
        )

    @torch.inference_mode()
    def infer(
        self,
        speaker: str,
        text: str,
        prompt_audio_path: str | None = None,
        prompt_audio_text: str | None = None,
        text_language: Literal["auto", "ja", "zh", "en"] = "auto",
        prompt_language: Literal["auto", "ja", "zh", "en"] = "auto",
        return_subtitles: bool = False,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        noise_scale: float = 0.5,
        speed: float = 1.0,
    ) -> AudioClip:
        """Generate speech for a single speaker.

        Args:
            speaker: Speaker name (must match a registered SpeakerConfig.name).
            text: Text to synthesize.
            prompt_audio_path: Override prompt audio for style reference.
            prompt_audio_text: Override prompt audio transcription.
            text_language: Language of the target text ("auto", "ja", "zh", "en").
            prompt_language: Language of the prompt audio text ("auto", "ja", "zh", "en").
            return_subtitles: Return word-level timestamp subtitles.
            top_k, top_p, temperature: GPT sampling parameters.
            repetition_penalty: GPT repetition penalty.
            noise_scale: SoVITS decoder noise scale.
            speed: Playback speed (1.0 = normal).

        Returns:
            AudioClip with generated audio.
        """
        with self._tts._infer_lock:
            w = self._require_speaker(speaker)

            if w.is_full_model:
                # Full-model speaker — no weight injection needed
                require_prompt = (
                    prompt_audio_path is None or prompt_audio_text is None
                )
                (
                    spk_key,
                    prompt_key,
                    cached_prompt_text,
                    gpt_key,
                    sovits_key,
                ) = self._register_full_model_cache(
                    speaker, require_prompt=require_prompt
                )

                if prompt_audio_path is None and prompt_audio_text is None:
                    prompt_audio_path = prompt_key
                    prompt_audio_text = cached_prompt_text
                elif prompt_audio_path is None or prompt_audio_text is None:
                    raise ValueError(
                        "prompt_audio_path and prompt_audio_text "
                        "must be provided together."
                    )

                return self._tts.infer(
                    spk_audio_path=spk_key,
                    prompt_audio_path=prompt_audio_path,
                    prompt_audio_text=prompt_audio_text,
                    text=text,
                    text_language=text_language,
                    prompt_language=prompt_language,
                    return_subtitles=return_subtitles,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    noise_scale=noise_scale,
                    speed=speed,
                    gpt_model=gpt_key,
                    sovits_model=sovits_key,
                )

            # Shared-backbone speaker — inject weights + use shared models
            require_prompt = prompt_audio_path is None or prompt_audio_text is None
            with self._activate_shared_models(speaker, require_prompt=require_prompt) as (
                spk_key,
                prompt_key,
                cached_prompt_text,
            ):
                if prompt_audio_path is None and prompt_audio_text is None:
                    prompt_audio_path = prompt_key
                    prompt_audio_text = cached_prompt_text
                elif prompt_audio_path is None or prompt_audio_text is None:
                    raise ValueError(
                        "prompt_audio_path and prompt_audio_text must be provided together."
                    )

                return self._tts.infer(
                    spk_audio_path=spk_key,
                    prompt_audio_path=prompt_audio_path,
                    prompt_audio_text=prompt_audio_text,
                    text=text,
                    text_language=text_language,
                    prompt_language=prompt_language,
                    return_subtitles=return_subtitles,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    noise_scale=noise_scale,
                    speed=speed,
                    gpt_model=_SHARED_GPT_KEY,
                    sovits_model=_SHARED_SOVITS_KEY,
                )

    def infer_auto(
        self,
        speaker_audio: str,
        text: str,
        min_similarity: float = 0.0,
        **kwargs,
    ) -> tuple[str, AudioClip]:
        """Route to the registered speaker whose timbre is closest to ``speaker_audio``.

        Computes the speaker embedding (SV) of the input audio and compares it
        with every registered speaker's cached timbre embedding (cosine
        similarity). The best match is then used for inference.

        Args:
            speaker_audio: Audio whose voice should be matched to a speaker.
            text: Text to synthesize.
            min_similarity: If the best match is below this similarity (0~1),
                raise ValueError instead of synthesizing.
            **kwargs: Passed through to ``infer`` (language, sampling, etc.).

        Returns:
            (matched_speaker_name, AudioClip).

        Raises:
            KeyError: If no speakers are registered.
            ValueError: If no speaker exceeds ``min_similarity``.
        """
        if not self._speakers:
            raise KeyError(
                "No speakers registered — add speakers before calling infer_auto"
            )

        tts = self._tts
        first = next(iter(self._speakers))
        with self._activate_shared_models(first, require_prompt=False):
            if _SHARED_SOVITS_KEY not in tts.sovits_models:
                tts.sovits_models[_SHARED_SOVITS_KEY] = self._shared_sovits
            model = tts.sovits_models[_SHARED_SOVITS_KEY]
            if tts.sv_model is None:
                tts.sv_model = ERes2Net(tts.sv_path, tts.tts_config)
            _, audio_tensor = tts._get_spec(model.hps, speaker_audio)
            query_emb = tts.sv_model.compute_embedding3(audio_tensor)

            best_name, best_sim = None, -1.0
            for name, w in self._speakers.items():
                tts.cache_spk_audio(w.spk_audio_path, sovits_model=_SHARED_SOVITS_KEY)
                sv_emb = tts.spk_audio_cache[w.spk_audio_path].get("sv_emb")
                if sv_emb is None:
                    continue
                sim = float(
                    torch.cosine_similarity(query_emb, sv_emb, dim=-1, eps=1e-6).item()
                )
                if sim > best_sim:
                    best_sim, best_name = sim, name

        if best_name is None:
            raise ValueError(
                "Could not compute speaker embeddings for any registered speaker"
            )
        if best_sim < min_similarity:
            raise ValueError(
                f"Best speaker '{best_name}' similarity {best_sim:.3f} "
                f"is below min_similarity {min_similarity}"
            )
        logger.info(f"Auto-routed to speaker '{best_name}' (similarity {best_sim:.3f})")
        return best_name, self.infer(best_name, text, **kwargs)

    def infer_batched(
        self,
        speaker_texts: list[tuple[str, str]],
        prompt_audio_paths: str | list[str] | None = None,
        prompt_audio_texts: str | list[str] | None = None,
        text_languages: Literal["auto", "ja", "zh", "en"] | list[str] = "auto",
        prompt_languages: Literal["auto", "ja", "zh", "en"] | list[str] = "auto",
        return_subtitles: bool = False,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        noise_scale: float = 0.5,
        speed: float = 1.0,
        bert_batch_size: int = 20,
        sovits_batch_size: int = 10,
    ) -> list[AudioClip]:
        """Batch inference with true GPU parallelism per speaker.

        Groups texts by speaker, then delegates to TTS.infer_batched() for
        each speaker group. Same-speaker texts share one weight injection
        and one GPU batch — orders of magnitude faster than per-text calls.

        Multi-speaker batches are supported: texts are grouped by speaker
        and each group is processed as an independent batch.

        Args:
            speaker_texts: List of (speaker_name, text) tuples.
            prompt_audio_paths: Optional external prompt audio override.
                If provided, falls back to per-text sequential inference.
            prompt_audio_texts: Optional external prompt transcription override.
            ... (other args, passed through to TTS.infer_batched)

        Returns:
            List of AudioClip results in the same order as speaker_texts.
        """
        if not speaker_texts:
            return []

        has_external_prompts = (
            prompt_audio_paths is not None or prompt_audio_texts is not None
        )

        # ── Group by speaker (preserving original order) ──
        groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for orig_idx, (speaker, text) in enumerate(speaker_texts):
            groups[speaker].append((orig_idx, text))

        all_results: list[AudioClip | None] = [None] * len(speaker_texts)

        if len(groups) > 1:
            logger.info(
                f"infer_batched: {len(speaker_texts)} texts across "
                f"{len(groups)} speakers — batching per speaker group"
            )

        with self._tts._infer_lock:
            for speaker, items in groups.items():
                orig_indices = [idx for idx, _ in items]
                texts = [text for _, text in items]

                if has_external_prompts:
                    # External prompt override → per-text fallback
                    for i, orig_idx in enumerate(orig_indices):
                        pp = (
                            prompt_audio_paths
                            if isinstance(prompt_audio_paths, str)
                            else prompt_audio_paths[orig_idx]
                            if isinstance(prompt_audio_paths, list)
                            else None
                        )
                        pt = (
                            prompt_audio_texts
                            if isinstance(prompt_audio_texts, str)
                            else prompt_audio_texts[orig_idx]
                            if isinstance(prompt_audio_texts, list)
                            else None
                        )
                        all_results[orig_idx] = self.infer(
                            speaker=speaker,
                            text=texts[i],
                            prompt_audio_path=pp,
                            prompt_audio_text=pt,
                            text_language=text_languages
                            if isinstance(text_languages, str)
                            else text_languages[orig_idx],
                            prompt_language=prompt_languages
                            if isinstance(prompt_languages, str)
                            else prompt_languages[orig_idx],
                            return_subtitles=return_subtitles,
                            top_k=top_k, top_p=top_p, temperature=temperature,
                            repetition_penalty=repetition_penalty,
                            noise_scale=noise_scale, speed=speed,
                        )
                else:
                    # Use cached prompt → true GPU batch
                    self._require_speaker(speaker)
                    with self._activate_shared_models(speaker) as (
                        spk_key,
                        prompt_key,
                        prompt_text,
                    ):
                        if prompt_key is None:
                            # Speaker has no cached prompt — fall back
                            logger.warning(
                                f"Speaker '{speaker}' has no cached prompt, "
                                "falling back to per-text inference."
                            )
                            for i, orig_idx in enumerate(orig_indices):
                                all_results[orig_idx] = self.infer(
                                    speaker=speaker,
                                    text=texts[i],
                                    text_language=text_languages
                                    if isinstance(text_languages, str)
                                    else text_languages[orig_idx],
                                    prompt_language=prompt_languages
                                    if isinstance(prompt_languages, str)
                                    else prompt_languages[orig_idx],
                                    return_subtitles=return_subtitles,
                                    top_k=top_k, top_p=top_p, temperature=temperature,
                                    repetition_penalty=repetition_penalty,
                                    noise_scale=noise_scale, speed=speed,
                                )
                        else:
                            audios = self._tts.infer_batched(
                                spk_audio_paths=spk_key,
                                prompt_audio_paths=prompt_key,
                                prompt_audio_texts=prompt_text,
                                texts=texts,
                                text_languages=text_languages
                                if isinstance(text_languages, str)
                                else [text_languages[oi] for oi in orig_indices],
                                prompt_languages=prompt_languages
                                if isinstance(prompt_languages, str)
                                else [prompt_languages[oi] for oi in orig_indices],
                                return_subtitles=return_subtitles,
                                top_k=top_k, top_p=top_p, temperature=temperature,
                                repetition_penalty=repetition_penalty,
                                noise_scale=noise_scale, speed=speed,
                                bert_batch_size=bert_batch_size,
                                sovits_batch_size=sovits_batch_size,
                                gpt_model=_SHARED_GPT_KEY,
                                sovits_model=_SHARED_SOVITS_KEY,
                            )
                            for orig_idx, audio in zip(orig_indices, audios):
                                all_results[orig_idx] = audio

        return all_results  # type: ignore[return-value]

    # ==================================================================
    # Script (dialogue) mode
    # ==================================================================

    def infer_script(
        self,
        script: str,
        text_language: Literal["auto", "ja", "zh", "en"] | list = "auto",
        prompt_language: Literal["auto", "ja", "zh", "en"] | list = "auto",
        return_subtitles: bool = True,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        noise_scale: float = 0.5,
        speed: float = 1.0,
        bert_batch_size: int = 20,
        sovits_batch_size: int = 10,
    ) -> tuple[AudioClip, list[dict]]:
        """Synthesize a multi-speaker dialogue script.

        Script format — one ``speaker: line`` per line (full-width ``：`` also
        accepted), e.g.::

            alice: こんにちは！
            bob: よろしくお願いします。
            alice: 今日も頑張りましょう！

        Inline ``<speaker:name>text</speaker>`` tags are also supported.

        Args:
            script: The dialogue script.
            text_language / prompt_language: A single language or a per-line
                list (must match the number of parsed lines).
            Other args match ``infer_batched``.

        Returns:
            (concatenated AudioClip, timeline). The clip's ``subtitles`` carry
            an extra ``"speaker"`` field per entry; the timeline is a list of
            ``{"speaker", "text", "start_s", "end_s"}`` per line.
        """
        entries = parse_script(script)
        if not entries:
            raise ValueError("Empty script — nothing to synthesize")

        audios = self.infer_batched(
            entries,
            text_languages=text_language,
            prompt_languages=prompt_language,
            return_subtitles=return_subtitles,
            top_k=top_k, top_p=top_p, temperature=temperature,
            repetition_penalty=repetition_penalty,
            noise_scale=noise_scale, speed=speed,
            bert_batch_size=bert_batch_size,
            sovits_batch_size=sovits_batch_size,
        )

        samplerate = audios[0].samplerate
        audio_data = np.concatenate([a.audio_data for a in audios])
        total_len = sum(a.audio_len_s for a in audios)

        timeline: list[dict] = []
        merged_subtitles: list[dict] = []
        offset = 0.0
        for (spk, text), clip in zip(entries, audios):
            end = offset + clip.audio_len_s
            timeline.append({
                "speaker": spk, "text": text,
                "start_s": offset, "end_s": end,
            })
            if clip.subtitles:
                for s in clip.subtitles:
                    shifted = dict(s)
                    shifted["start_s"] = s["start_s"] + offset
                    shifted["end_s"] = s["end_s"] + offset
                    shifted["speaker"] = spk
                    merged_subtitles.append(shifted)
            offset = end

        clip = AudioClip(
            self.audio_queue, audio_data, samplerate, total_len,
            merged_subtitles or None, script,
        )
        return clip, timeline

    def infer_stream(
        self,
        speaker: str,
        text: str,
        prompt_audio_path: str | None = None,
        prompt_audio_text: str | None = None,
        text_language: Literal["auto", "ja", "zh", "en"] = "auto",
        prompt_language: Literal["auto", "ja", "zh", "en"] = "auto",
        return_subtitles: bool = False,
        is_cut_text: bool = True,
        cut_minlen: int = 10,
        cut_mute: float = 0.4,
        cut_mute_scale_map: dict[str, float] | None = None,
        stream_mode: Literal["token", "sentence"] = "token",
        stream_chunk: int = 25,
        overlap_len: int = 5,
        boost_first_chunk: bool = True,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        noise_scale: float = 0.5,
        speed: float = 1.0,
        debug: bool = False,
    ) -> Generator[AudioClip, None, None]:
        """Streaming inference — token-level streaming via the shared backbone.

        Delegates to TTS.infer_stream() after registering speaker resources:
        the shared GPT/SoVITS models are activated and speaker weights are
        injected once (via copy_()) before the stream starts, so every yielded
        chunk is generated token-by-token with the same low latency as TTS.

        Args:
            speaker: Speaker name.
            text: Text to synthesize.
            prompt_audio_path: Override prompt audio (tone/style reference).
            prompt_audio_text: Override prompt audio transcription.
            ... (other args match TTS.infer_stream)

        Yields:
            AudioClip chunks as they are generated (streaming).
        """
        with self._tts._infer_lock:
            w = self._require_speaker(speaker)

            if w.is_full_model:
                # Full-model speaker — no weight injection needed
                require_prompt = (
                    prompt_audio_path is None or prompt_audio_text is None
                )
                (
                    spk_key,
                    prompt_key,
                    cached_prompt_text,
                    gpt_key,
                    sovits_key,
                ) = self._register_full_model_cache(
                    speaker, require_prompt=require_prompt
                )

                if prompt_audio_path is None and prompt_audio_text is None:
                    prompt_audio_path = prompt_key
                    prompt_audio_text = cached_prompt_text
                elif prompt_audio_path is None or prompt_audio_text is None:
                    raise ValueError(
                        "prompt_audio_path and prompt_audio_text "
                        "must be provided together."
                    )

                yield from self._tts.infer_stream(
                    spk_audio_path=spk_key,
                    prompt_audio_path=prompt_audio_path,
                    prompt_audio_text=prompt_audio_text,
                    text=text,
                    text_language=text_language,
                    prompt_language=prompt_language,
                    return_subtitles=return_subtitles,
                    is_cut_text=is_cut_text,
                    cut_minlen=cut_minlen,
                    cut_mute=cut_mute,
                    cut_mute_scale_map=cut_mute_scale_map,
                    stream_mode=stream_mode,
                    stream_chunk=stream_chunk,
                    overlap_len=overlap_len,
                    boost_first_chunk=boost_first_chunk,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    noise_scale=noise_scale,
                    speed=speed,
                    gpt_model=gpt_key,
                    sovits_model=sovits_key,
                    debug=debug,
                )
                return

            # Shared-backbone speaker — inject weights + use shared models
            require_prompt = (
                prompt_audio_path is None or prompt_audio_text is None
            )
            with self._activate_shared_models(
                speaker, require_prompt=require_prompt
            ) as (
                spk_key,
                prompt_key,
                cached_prompt_text,
            ):
                if prompt_audio_path is None and prompt_audio_text is None:
                    prompt_audio_path = prompt_key
                    prompt_audio_text = cached_prompt_text
                elif prompt_audio_path is None or prompt_audio_text is None:
                    raise ValueError(
                        "prompt_audio_path and prompt_audio_text "
                        "must be provided together."
                    )

                yield from self._tts.infer_stream(
                    spk_audio_path=spk_key,
                    prompt_audio_path=prompt_audio_path,
                    prompt_audio_text=prompt_audio_text,
                    text=text,
                    text_language=text_language,
                    prompt_language=prompt_language,
                    return_subtitles=return_subtitles,
                    is_cut_text=is_cut_text,
                    cut_minlen=cut_minlen,
                    cut_mute=cut_mute,
                    cut_mute_scale_map=cut_mute_scale_map,
                    stream_mode=stream_mode,
                    stream_chunk=stream_chunk,
                    overlap_len=overlap_len,
                    boost_first_chunk=boost_first_chunk,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    noise_scale=noise_scale,
                    speed=speed,
                    gpt_model=_SHARED_GPT_KEY,
                    sovits_model=_SHARED_SOVITS_KEY,
                    debug=debug,
                )

    def _empty_cache(self):
        """Release unused GPU memory."""
        self._tts._empty_cache()
