"""Command-line interface: ``python -m gsv_tts`` (or ``gsv-tts`` after install).

Subcommands:
    infer     synthesize text to a wav file (single speaker)
    multi     synthesize a dialogue script with multiple speakers
    models    show default model paths & cache status
    convert   convert a .pth/.ckpt checkpoint to the safetensors directory format
"""

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gsv_tts",
        description="GSV-TTS-Lite MultiSpeaker command-line interface",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── infer ──
    p_infer = sub.add_parser("infer", help="synthesize text to a wav file")
    p_infer.add_argument("--spk", required=True, help="timbre reference audio (whose voice)")
    p_infer.add_argument("--prompt", default=None, help="style reference audio (defaults to --spk)")
    p_infer.add_argument("--prompt-text", required=True, help="transcription of the style reference audio")
    p_infer.add_argument("--text", required=True, help="text to synthesize")
    p_infer.add_argument("--lang", default="auto", choices=["auto", "ja", "zh", "en"], help="text language")
    p_infer.add_argument("--out", default="output.wav", help="output wav path")
    p_infer.add_argument("--models-dir", default=None, help="pretrained models directory")

    # ── multi ──
    p_multi = sub.add_parser("multi", help="synthesize a dialogue script with multiple speakers")
    p_multi.add_argument(
        "--speakers", required=True, metavar="JSON",
        help="path to a JSON file: a list of SpeakerConfig dicts "
             "(name/gpt_model_path/sovits_model_path/spk_audio_path/"
             "prompt_audio_path/prompt_audio_text)",
    )
    p_multi.add_argument("--script", required=True, metavar="FILE", help="dialogue script file ('speaker: text' per line)")
    p_multi.add_argument("--out", default="output.wav", help="output wav path")
    p_multi.add_argument("--srt", default=None, help="optional SRT subtitle output path")
    p_multi.add_argument("--models-dir", default=None, help="pretrained models directory")

    # ── models ──
    p_models = sub.add_parser("models", help="show default model paths & cache status")
    p_models.add_argument("--models-dir", default=None, help="pretrained models directory")

    # ── convert ──
    p_conv = sub.add_parser("convert", help="convert a .pth/.ckpt checkpoint to a safetensors directory")
    p_conv.add_argument("path", help="checkpoint file (.pth or .ckpt)")
    p_conv.add_argument("--models-dir", default=None, help="pretrained models directory")

    return parser


def _cmd_infer(args) -> int:
    from .TTS import TTS

    tts = TTS(use_bert=True, models_dir=args.models_dir)
    audio = tts.infer(
        spk_audio_path=args.spk,
        prompt_audio_path=args.prompt or args.spk,
        prompt_audio_text=args.prompt_text,
        text=args.text,
        text_language=args.lang,
    )
    audio.save(args.out, exist_ok=True)
    print(f"Saved: {args.out} ({audio.audio_len_s:.2f}s)")
    return 0


def _cmd_multi(args) -> int:
    from .MultiSpeaker import MultiSpeakerTTS, parse_script
    from .SpeakerWeights import SpeakerConfig

    with open(args.speakers, encoding="utf-8") as f:
        items = json.load(f)
    if not items:
        print("Error: empty speakers JSON", file=sys.stderr)
        return 1

    speakers = []
    for it in items:
        kwargs = {k: v for k, v in it.items() if v is not None}
        speakers.append(SpeakerConfig(**kwargs))

    tts = MultiSpeakerTTS(speakers=speakers, use_bert=True, models_dir=args.models_dir)
    script = Path(args.script).read_text(encoding="utf-8")
    entries = parse_script(script)
    if not entries:
        print("Error: empty script", file=sys.stderr)
        return 1

    audio, timeline = tts.infer_script(script)
    audio.save(args.out, exist_ok=True)
    print(f"Saved: {args.out} ({audio.audio_len_s:.2f}s, {len(entries)} lines)")
    for t in timeline:
        print(f"  [{t['speaker']}] {t['text']}  ({t['start_s']:.2f}s-{t['end_s']:.2f}s)")
    if args.srt:
        audio.export_subtitles(args.srt)
        print(f"Subtitles: {args.srt}")
    return 0


def _cmd_models(args) -> int:
    from .Download import _DEFAULT_MODEL_FILES, check_pretrained_models

    base = Path(args.models_dir) if args.models_dir else Path.home() / ".cache" / "gsv"
    print(f"models_dir: {base}")
    for name in _DEFAULT_MODEL_FILES:
        p = base / name
        if p.exists():
            print(f"  [OK] {name} ({p.stat().st_size / 1e9:.2f} GB)")
        else:
            print(f"  [--] {name} (missing — will be downloaded on first use)")
    for d in ("chinese-hubert-base", "g2p", "sv", "chinese-roberta-wwm-ext-large"):
        p = base / d
        print(f"  {'[OK]' if p.exists() else '[--]'} {d}")
    return 0


def _cmd_convert(args) -> int:
    from .TTS import TTS

    tts = TTS(models_dir=args.models_dir)
    out = tts.to_safetensors(args.path)
    print(f"Converted: {out}")
    return 0


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "infer":
        return _cmd_infer(args)
    if args.command == "multi":
        return _cmd_multi(args)
    if args.command == "models":
        return _cmd_models(args)
    if args.command == "convert":
        return _cmd_convert(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
