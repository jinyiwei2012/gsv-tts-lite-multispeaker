"""MultiSpeakerTTS vs full-model benchmark (CPU reference environment).

Compares:
  A. MultiSpeakerTTS shared backbone (base + N speaker weights)
  B. Full model loading (N standalone TTS instances, one per speaker)

Metrics: init time, peak RSS (GB), per-speaker warmup + avg inference
latency, RTF.

Models are auto-discovered from the repo (paired .ckpt + .pth by filename
prefix), or explicitly specified via --gpt/--sovits pairs.

Usage:
    # Auto-discover models under the repo root
    python benchmarks/bench_multi_speaker.py

    # Scan custom directory(ies) instead
    python benchmarks/bench_multi_speaker.py --models-dir models

    # Explicit model pairs (paired by order, repeatable; also accepts
    # safetensors directory paths)
    python benchmarks/bench_multi_speaker.py \
        --gpt models/alice_gpt.ckpt   --sovits models/alice_sovits.pth \
        --gpt models/bob_gpt.ckpt     --sovits models/bob_sovits.pth \
        --name alice --name bob

    # Skip the full-loading scenario, only benchmark shared backbone
    python benchmarks/bench_multi_speaker.py --no-full
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from gsv_tts import TTS, MultiSpeakerTTS, SpeakerConfig
from gsv_tts.model_discovery import discover_models, auto_name

DEFAULT_TEXT = "今日も頑張りましょう、一緒に歩いていこう。"
SPK_AUDIO = "examples/laffey.mp3"
PROMPT_AUDIO = "examples/AnAn.ogg"
PROMPT_TEXT = "ちが……ちがう。レイア、貴様は間違っている。"


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def timed(fn, label):
    t0 = time.time()
    result = fn()
    dt = time.time() - t0
    print(f"  {label}: {dt:.1f}s", flush=True)
    return result, dt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="MultiSpeakerTTS vs full-model benchmark"
    )
    parser.add_argument(
        "--models-dir", action="append", default=[], metavar="DIR",
        help="directory to scan for model files (repeatable; default: repo root)",
    )
    parser.add_argument(
        "--gpt", action="append", default=[], metavar="PATH",
        help="explicit GPT model path, paired with --sovits by order (repeatable; "
             "also accepts safetensors directories)",
    )
    parser.add_argument(
        "--sovits", action="append", default=[], metavar="PATH",
        help="explicit SoVITS model path, paired with --gpt by order (repeatable; "
             "also accepts safetensors directories)",
    )
    parser.add_argument(
        "--name", action="append", default=[], metavar="NAME",
        help="speaker name for explicit pairs, by order (optional)",
    )
    parser.add_argument(
        "--text", default=DEFAULT_TEXT, help=f"benchmark text (default: {DEFAULT_TEXT!r})",
    )
    parser.add_argument(
        "--avg", type=int, default=2, metavar="N",
        help="inference repetitions per speaker (default: 2)",
    )
    parser.add_argument(
        "--no-full", action="store_true",
        help="skip scenario B (full model loading)",
    )
    parser.add_argument(
        "--min-prefix", type=int, default=4, metavar="N",
        help="min filename prefix length for auto pairing (default: 4)",
    )
    parser.add_argument(
        "--output", type=str, default=None, metavar="JSON",
        help="write a JSON report of the benchmark results",
    )
    return parser.parse_args(argv)


def resolve_pairs(args):
    """Return list of (name, gpt_path, sovits_path)."""
    if args.gpt or args.sovits:
        if len(args.gpt) != len(args.sovits):
            raise SystemExit(
                "--gpt and --sovits must appear the same number of times "
                f"(got {len(args.gpt)} vs {len(args.sovits)})"
            )
        pairs = []
        for i, (gpt, sovits) in enumerate(zip(args.gpt, args.sovits)):
            name = args.name[i] if i < len(args.name) and args.name[i] else _auto_name(Path(gpt), Path(sovits))
            pairs.append((name, gpt, sovits))
        return pairs

    repo_root = Path(__file__).parent.parent
    dirs = [Path(d) for d in args.models_dir] or [repo_root]
    discovered = discover_models(dirs, args.min_prefix)
    if not discovered:
        print(
            f"No paired .ckpt/.pth models found under: "
            + ", ".join(str(d) for d in dirs),
            file=sys.stderr,
        )
        print(
            "Put your fine-tuned models in the repo (or --models-dir), or specify "
            "them explicitly, e.g.:\n"
            "  python benchmarks/bench_multi_speaker.py "
            "--gpt path/to/gpt.ckpt --sovits path/to/sovits.pth",
            file=sys.stderr,
        )
        sys.exit(1)
    return [(name, str(gpt), str(sovits)) for gpt, sovits, name in discovered]


def main():
    args = parse_args()
    pairs = resolve_pairs(args)
    n = len(pairs)
    text = args.text

    print("=" * 60, flush=True)
    print("MultiSpeakerTTS vs Full Model Benchmark (CPU)", flush=True)
    print("=" * 60, flush=True)
    print(f"Benchmarking {n} speaker pair(s):", flush=True)
    for name, gpt, sovits in pairs:
        print(f"  - {name}: {gpt} + {sovits}", flush=True)

    # ── Scenario A: MultiSpeakerTTS shared backbone ──
    print("\n[A] MultiSpeakerTTS shared backbone", flush=True)
    gc.collect()
    rss_before = rss_gb()

    def build_shared():
        speakers = [
            SpeakerConfig(
                name=name,
                gpt_model_path=gpt,
                sovits_model_path=sovits,
                spk_audio_path=SPK_AUDIO,
                prompt_audio_path=PROMPT_AUDIO,
                prompt_audio_text=PROMPT_TEXT,
            )
            for name, gpt, sovits in pairs
        ]
        return MultiSpeakerTTS(speakers=speakers, use_bert=True)

    mtts, init_a = timed(build_shared, f"init {n} speakers (shared)")

    for name, _, _ in pairs:
        w = mtts._speakers[name]
        mode = "shared" if not w.is_full_model else "FULL-DEGRADED"
        print(f"  speaker '{name}': {mode}", flush=True)

    rss_a = rss_gb() - rss_before
    print(f"  RSS delta: {rss_a:.2f} GB", flush=True)

    results_a = {}
    for name, _, _ in pairs:
        _, w = timed(lambda n=name: mtts.infer(n, text), f"warmup infer '{name}'")
        times = []
        for _ in range(args.avg):
            t0 = time.time()
            mtts.infer(name, text)
            times.append(time.time() - t0)
        avg = sum(times) / len(times)
        results_a[name] = (w, avg)
        print(f"  infer '{name}': warmup {w:.1f}s, avg {avg:.1f}s", flush=True)

    # ── Scenario B: full model per speaker ──
    if args.no_full:
        print("\n[Skipped] Scenario B (--no-full)", flush=True)
        results_b, instances, rss_b = None, {}, 0.0
    else:
        print(f"\n[B] Full model loading ({n} standalone TTS)", flush=True)
        gc.collect()
        rss_before = rss_gb()

        instances = {}
        for name, gpt, sovits in pairs:
            t = TTS(use_bert=True)

            def load(t=t, gpt=gpt, sovits=sovits):
                t.load_gpt_model(gpt)
                t.load_sovits_model(sovits)

            _, dt = timed(load, f"load full models '{name}'")
            instances[name] = (t, dt)

        rss_b = rss_gb() - rss_before
        print(f"  RSS delta: {rss_b:.2f} GB", flush=True)

        results_b = {}
        for name, _, _ in pairs:
            t = instances[name][0]
            _, w = timed(
                lambda n=name: t.infer(
                    spk_audio_path=SPK_AUDIO, prompt_audio_path=PROMPT_AUDIO,
                    prompt_audio_text=PROMPT_TEXT, text=text,
                ),
                f"warmup infer '{name}' (full)",
            )
            times = []
            for _ in range(args.avg):
                t0 = time.time()
                t.infer(
                    spk_audio_path=SPK_AUDIO, prompt_audio_path=PROMPT_AUDIO,
                    prompt_audio_text=PROMPT_TEXT, text=text,
                )
                times.append(time.time() - t0)
            avg = sum(times) / len(times)
            results_b[name] = (w, avg)
            print(f"  infer '{name}': warmup {w:.1f}s, avg {avg:.1f}s", flush=True)

    # ── Summary ──
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"{'':30s} {'Shared (A)':>12s} {'Full (B)':>12s} {'Speedup':>10s}", flush=True)
    print(f"{f'Init ({n} speakers)':30s} {init_a:>10.1f}s "
          f"{sum(v[1] for v in instances.values()) if not args.no_full else 0:>10.1f}s", flush=True)
    if not args.no_full and rss_b > 0:
        print(f"{'Peak RSS delta':30s} {rss_a:>10.2f}G {rss_b:>10.2f}G "
              f"{'saved ' + f'{100*(1-rss_a/rss_b):.0f}%':>10s}", flush=True)
    else:
        print(f"{'Peak RSS delta':30s} {rss_a:>10.2f}G", flush=True)
    for name, _, _ in pairs:
        wa, aa = results_a[name]
        if args.no_full:
            print(f"{'infer avg ' + name:30s} {aa:>10.1f}s", flush=True)
        else:
            wb, ab = results_b[name]
            print(f"{'infer avg ' + name:30s} {aa:>10.1f}s {ab:>10.1f}s {ab/aa:>8.2f}x", flush=True)

    # RTF for first speaker (audio len ~ text duration)
    clip = mtts.infer(pairs[0][0], text)
    audio_len = clip.audio_len_s
    rtf_a = results_a[pairs[0][0]][1] / audio_len
    print(f"\nAudio length: {audio_len:.2f}s", flush=True)
    if not args.no_full:
        rtf_b = results_b[pairs[0][0]][1] / audio_len
        print(f"RTF shared: {rtf_a:.3f} | RTF full: {rtf_b:.3f}", flush=True)
    else:
        rtf_b = None
        print(f"RTF shared: {rtf_a:.3f}", flush=True)

    # ── JSON report ──
    if args.output:
        report = {
            "meta": {
                "n_speakers": n,
                "avg_reps": args.avg,
                "text": text,
                "no_full": args.no_full,
            },
            "init_s": {
                "shared": init_a,
                "full": sum(v[1] for v in instances.values()) if not args.no_full else None,
            },
            "peak_rss_gb": {
                "shared": rss_a,
                "full": rss_b if not args.no_full else None,
            },
            "speakers": {},
            "rtf": {"shared": rtf_a, "full": rtf_b},
            "audio_len_s": audio_len,
        }
        for name, gpt, sovits in pairs:
            wa, aa = results_a[name]
            entry = {
                "gpt_model": gpt,
                "sovits_model": sovits,
                "mode": "shared" if not mtts._speakers[name].is_full_model else "full_degraded",
                "warmup_s": wa,
                "avg_s": aa,
            }
            if not args.no_full:
                wb, ab = results_b[name]
                entry["full_avg_s"] = ab
                entry["speedup_x"] = round(ab / aa, 3) if aa > 0 else None
            report["speakers"][name] = entry
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport written to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
