import os
import json
import queue
import logging
import numpy as np
import soundfile as sf
import threading
try:
    import sounddevice as sd
except ImportError:
    sd = None
    logging.warning("sounddevice not available — audio playback disabled")


_SENTINEL = object()


def _format_srt_ts(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm."""
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_ass_ts(seconds: float) -> str:
    """Format seconds as ASS timestamp: H:MM:SS.cc."""
    cs = max(0, int(round(seconds * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


class AudioQueue:
    def __init__(self, samplerate):
        self.samplerate = samplerate
        self.q = queue.Queue()
        self.t = None
        self.playback_finished = threading.Event()
        self.playback_finished.set()

        try:
            self.stream = sd.OutputStream(
                samplerate=self.samplerate,
                channels=1,
                dtype='float32'
            )
            self.stream.start()
        except Exception:
            self.stream = None

    def put(self, data):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        self.q.put(data)
        
        if self.t is None or not self.t.is_alive():
            self.playback_finished.clear()
            self.t = threading.Thread(target=self._run_playback, daemon=True)
            self.t.start()

    def _run_playback(self):
        while True:
            data = self.q.get()
            if data is _SENTINEL:
                break
            if self.stream is not None:
                try:
                    self.stream.write(data)
                except Exception as e:
                    logging.warning(f"Audio playback error: {e}")
                    break
        
        self.playback_finished.set()

    def stop(self):
        """
        Immediately stops playback and clears all audio data in the queue.
        """
        # Drain queue using sentinel
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.q.put(_SENTINEL)

        if self.t is not None and self.t.is_alive():
            self.t.join(timeout=5.0)

        if self.stream is not None:
            self.stream.stop()
            self.stream.start()
        
        self.playback_finished.set()

    def wait(self, timeout: float = 30.0):
        """
        Waits until all audio currently in the queue has finished playing.

        Args:
            timeout: Maximum time to wait in seconds. Defaults to 30.0.
        """
        if not self.playback_finished.wait(timeout=timeout):
            logging.warning("Audio playback did not finish within timeout")

    def close(self):
        """Clean up resources."""
        self.stop()
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class AudioClip:
    def __init__(self, audio_queue, audio_data, samplerate, audio_len_s, subtitles, orig_text):
        self.audio_queue: AudioQueue = audio_queue
        self.audio_data = audio_data
        self.samplerate = samplerate
        self.audio_len_s = audio_len_s
        self.subtitles = subtitles
        self.orig_text = orig_text
    
    def play(self, volume: float = 1.0):
        """
        Adds the audio data to the playback queue for sequential output.
        """
        audio = self.audio_data
        if volume != 1.0:
            audio = audio * volume
            audio = np.clip(audio, -1.0, 1.0)

        self.audio_queue.put(audio)
    
    def save(self, save_path: str, is_save_subtitles: bool = False, exist_ok: bool = False):
        """
        Saves the audio data to a file and optionally exports subtitles as a JSON file.

        Args:
            save_path: Output file path.
            is_save_subtitles: Whether to also save a .json subtitle file.
            exist_ok: If False (default), raises FileExistsError when the file already exists.
        """
        if not exist_ok and os.path.exists(save_path):
            raise FileExistsError(f"File already exists: {save_path}")

        sf.write(save_path, self.audio_data, self.samplerate)

        if is_save_subtitles:
            subtitles_path, _ = os.path.splitext(save_path)
            subtitles_path = subtitles_path + ".json"
            with open(subtitles_path, 'w', encoding='utf-8') as f:
                json.dump({"orig_text":self.orig_text, "subtitles":self.subtitles}, f, indent=4, ensure_ascii=False)

    def export_subtitles(self, save_path: str, fmt: str = "srt"):
        """Exports character-level subtitles to an SRT or ASS file.

        Args:
            save_path: Output subtitle file path (e.g. "out.srt" / "out.ass").
            fmt: "srt" or "ass".

        Returns:
            The path the subtitles were written to.

        Raises:
            ValueError: If this clip has no subtitles (inference was run
                without ``return_subtitles=True``), or the format is unknown.
        """
        if not self.subtitles:
            raise ValueError(
                "No subtitles available. Re-run inference with return_subtitles=True."
            )

        subs = sorted(self.subtitles, key=lambda s: s.get("start_s", 0.0))

        if fmt == "srt":
            lines = []
            for i, s in enumerate(subs, 1):
                lines.append(
                    f"{i}\n"
                    f"{_format_srt_ts(s['start_s'])} --> {_format_srt_ts(s['end_s'])}\n"
                    f"{s['text']}\n"
                )
            content = "\n".join(lines)
        elif fmt == "ass":
            header = (
                "[Script Info]\n"
                "ScriptType: v4.00+\n"
                "PlayResX: 384\n"
                "PlayResY: 288\n"
                "WrapStyle: 0\n"
                "\n"
                "[V4+ Styles]\n"
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding\n"
                "Style: Default,Microsoft YaHei,36,&H00FFFFFF,&H000000FF,"
                "&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,0,2,20,20,24,1\n"
                "\n"
                "[Events]\n"
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            )
            events = []
            for s in subs:
                text = str(s["text"]).replace("\n", "\\N")
                events.append(
                    "Dialogue: 0,"
                    f"{_format_ass_ts(s['start_s'])},{_format_ass_ts(s['end_s'])},"
                    f"Default,,0,0,0,,{text}"
                )
            content = header + "\n".join(events) + "\n"
        else:
            raise ValueError(f"Unsupported subtitle format: {fmt}")

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return save_path
