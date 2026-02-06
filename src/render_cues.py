import tempfile
from pathlib import Path
from typing import Any, Dict, List

from pydub import AudioSegment
from src.tts_piper import piper_tts


def render_cue_track(
    duration_ms: int,
    cues: List[Dict[str, Any]],
    piper_model_path: str,
    cue_gain_db: float = 0.0,
) -> AudioSegment:
    base = AudioSegment.silent(duration=duration_ms)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for idx, cue in enumerate(cues):
            wav = td / f"cue_{idx}.wav"
            piper_tts(cue["text"], str(wav), piper_model_path)

            a = AudioSegment.from_wav(str(wav))
            if cue_gain_db:
                a = a.apply_gain(float(cue_gain_db))

            base = base.overlay(a, position=int(cue["t_ms"]))

    return base
