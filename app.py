import io
import json
import zipfile
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import streamlit as st
import librosa
import soundfile as sf
from pydub import AudioSegment
from gtts import gTTS


# ----------------------------
# Helpers
# ----------------------------

@dataclass
class CueEvent:
    t_sec: float
    text: str


def load_audio_any(in_bytes: bytes, filename: str, target_sr: int = 44100) -> Tuple[np.ndarray, int, float]:
    """
    Decode audio (mp3/wav/etc) using pydub (ffmpeg), return mono float32 waveform at target_sr.
    """
    seg = AudioSegment.from_file(io.BytesIO(in_bytes), format=filename.split(".")[-1].lower())
    seg = seg.set_channels(1).set_frame_rate(target_sr)
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    # normalize based on sample width
    peak = float(1 << (8 * seg.sample_width - 1))
    y = samples / peak
    duration = len(seg) / 1000.0
    return y, target_sr, duration


def estimate_bpm(y: np.ndarray, sr: int) -> float:
    # librosa tempo estimate (rough but ok for v1)
    tempo = librosa.feature.tempo(y=y, sr=sr, aggregate=np.median)
    bpm = float(tempo)
    # keep in a sane range
    if bpm < 60:
        bpm *= 2
    if bpm > 200:
        bpm /= 2
    return float(np.clip(bpm, 60, 200))


def make_click_wav(duration_sec: float, sr: int, bpm: float, click_hz: float = 2000.0) -> np.ndarray:
    """
    Simple click: short sine burst every beat.
    """
    beat_period = 60.0 / bpm
    n = int(duration_sec * sr)
    out = np.zeros(n, dtype=np.float32)

    click_len = int(0.012 * sr)  # 12 ms burst
    t = np.arange(click_len) / sr
    burst = 0.7 * np.sin(2 * np.pi * click_hz * t) * np.hanning(click_len).astype(np.float32)

    time = 0.0
    while time < duration_sec:
        i = int(time * sr)
        j = min(i + click_len, n)
        out[i:j] += burst[: (j - i)]
        time += beat_period

    # prevent clipping
    out = np.clip(out, -1.0, 1.0)
    return out


def detect_silences(y: np.ndarray, sr: int, win_ms: int = 50, min_silence_ms: int = 350) -> List[Tuple[float, float]]:
    """
    Very simple silence detector based on RMS threshold.
    Returns list of (start_sec, end_sec) for silent regions.
    """
    hop = int(sr * win_ms / 1000)
    frame = hop
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-10, ref=np.max)

    # Threshold: below -30 dB relative to max is "silence" (tweakable)
    silent = rms_db < -30.0

    silences = []
    start = None
    for idx, is_silent in enumerate(silent):
        if is_silent and start is None:
            start = idx
        if (not is_silent) and (start is not None):
            end = idx
            s = start * hop / sr
            e = end * hop / sr
            if (e - s) * 1000 >= min_silence_ms:
                silences.append((s, e))
            start = None

    # tail
    if start is not None:
        s = start * hop / sr
        e = len(y) / sr
        if (e - s) * 1000 >= min_silence_ms:
            silences.append((s, e))

    # remove early/late tiny artifacts
    cleaned = []
    for s, e in silences:
        if e - s >= 0.35:
            cleaned.append((s, e))
    return cleaned


def tts_to_mp3_bytes(text: str, lang: str = "ru") -> bytes:
    mp3_fp = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(mp3_fp)
    return mp3_fp.getvalue()


def overlay_cues(duration_sec: float, cue_events: List[CueEvent], lang: str = "ru") -> AudioSegment:
    """
    Build an audio track (mp3) with cues at timestamps.
    """
    base = AudioSegment.silent(duration=int(duration_sec * 1000))

    for ev in cue_events:
        t_ms = int(max(ev.t_sec, 0) * 1000)
        cue_mp3 = tts_to_mp3_bytes(ev.text, lang=lang)
        cue_seg = AudioSegment.from_file(io.BytesIO(cue_mp3), format="mp3")
        base = base.overlay(cue_seg, position=t_ms)

    return base


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="Drummer Cues v1", page_icon="🥁", layout="centered")
st.title("🥁 Drummer Cues — Web v1")
st.caption("v1: BPM + click.wav + подсказки по паузам (cues.mp3)")

uploaded = st.file_uploader("Загрузи трек (mp3/wav/m4a/flac)", type=["mp3", "wav", "m4a", "aac", "flac"])

lang = st.selectbox("Язык подсказок", ["ru", "en"], index=0)
silence_db_hint = st.slider("Чувствительность тишины (пока простой параметр)", 20, 45, 30)
# (в v1 это просто UI; реальный порог в detect_silences фиксированный — можно связать позже)

generate = st.button("Generate", type="primary", disabled=uploaded is None)

if generate and uploaded:
    with st.spinner("Анализируем и генерим файлы..."):
        audio_bytes = uploaded.read()
        y, sr, duration = load_audio_any(audio_bytes, uploaded.name, target_sr=44100)

        bpm = estimate_bpm(y, sr)
        st.write(f"Оценка BPM: **{bpm:.1f}**")
        st.write(f"Длина трека: **{duration:.1f} сек**")

        # 1) click
        click = make_click_wav(duration, sr, bpm)

        # 2) silences -> cues
        silences = detect_silences(y, sr)
        st.write(f"Найдено пауз/стопов (по тишине): **{len(silences)}**")

        cue_events: List[CueEvent] = []
        for (s, e) in silences:
            # предупреждение о паузе чуть заранее
            cue_events.append(CueEvent(t_sec=max(s - 0.7, 0), text="Пауза" if lang == "ru" else "Break"))
            # сигнал входа чуть до окончания тишины
            cue_events.append(CueEvent(t_sec=max(e - 0.25, 0), text="Вход" if lang == "ru" else "In"))

        cue_events = sorted(cue_events, key=lambda x: x.t_sec)

        # build cue track
        cues_seg = overlay_cues(duration, cue_events, lang=lang)

        # 3) markers.json
        markers = []
        for (s, e) in silences:
            markers.append({"type": "silence_start", "t_sec": float(s)})
            markers.append({"type": "silence_end", "t_sec": float(e)})

        markers_json = {
            "song": uploaded.name,
            "duration_sec": float(duration),
            "sr": int(sr),
            "bpm_estimate": float(bpm),
            "events": [{"t_sec": ev.t_sec, "text": ev.text} for ev in cue_events],
            "markers": markers,
        }

        # 4) zip export
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
            # click.wav
            click_wav = io.BytesIO()
            sf.write(click_wav, click, sr, format="WAV", subtype="PCM_16")
            z.writestr("click.wav", click_wav.getvalue())

            # cues.mp3
            cues_mp3 = io.BytesIO()
            cues_seg.export(cues_mp3, format="mp3", bitrate="192k")
            z.writestr("cues.mp3", cues_mp3.getvalue())

            # markers.json
            z.writestr("markers.json", json.dumps(markers_json, ensure_ascii=False, indent=2))

            # source
            z.writestr(f"input/{uploaded.name}", audio_bytes)

        mem.seek(0)

    st.success("Готово ✅")
    st.download_button(
        "Download result.zip",
        data=mem,
        file_name="result.zip",
        mime="application/zip",
    )

    # debug preview
    with st.expander("Показать найденные паузы"):
        for i, (s, e) in enumerate(silences, 1):
            st.write(f"{i}. {s:.2f}s → {e:.2f}s  (длительность {e-s:.2f}s)")
