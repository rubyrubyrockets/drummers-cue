# streamlit_app.py
# Cloud-safe версия: загрузка трека -> конвертация в WAV -> анализ структуры -> генерация базовых барабанных подсказок
# Требования: streamlit, pydub, soundfile, librosa, numpy, scipy, requests (не обязателен)
#
# ВАЖНО:
# 1) положи файл src/allin1fix.py (который я дал ранее)
# 2) в packages.txt должен быть ffmpeg (ты уже ставишь)
# 3) этот streamlit_app.py НЕ требует твоих старых модулей src/structure_allin1.py и т.п.

import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from pydub import AudioSegment
import soundfile as sf
import librosa

from src.allin1fix import analyze_structure


APP_TITLE = "Drummer's Cue (Cloud-safe)"
TARGET_SR = 44100


# -------------------------
# Utils
# -------------------------
def to_wav_44100_mono(input_path: str, out_wav_path: str) -> str:
    """
    Конвертирует любой аудиофайл (mp3/wav/m4a/ogg/...) в WAV 44.1kHz mono.
    Использует pydub (ffmpeg должен быть доступен).
    """
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(TARGET_SR).set_channels(1)
    audio.export(out_wav_path, format="wav")
    return out_wav_path


def read_duration_seconds(wav_path: str) -> float:
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    return float(librosa.get_duration(y=y, sr=sr))


def format_time(t: float) -> str:
    mm = int(t // 60)
    ss = int(t % 60)
    return f"{mm:02d}:{ss:02d}"


def build_basic_drummer_cues(segments, lead_seconds: float = 4.0):
    """
    Делает простые подсказки по структуре.
    Пример: за 4 секунды до начала секции говорим "CHORUS".
    """
    cues = []
    for s in segments:
        start = float(s["start"])
        label = str(s["label"]).upper()

        cue_time = max(0.0, start - lead_seconds)
        text = label

        cues.append(
            {
                "time": cue_time,
                "text": text,
                "type": "section",
                "target_section_start": start,
            }
        )

    # Сортировка по времени
    cues.sort(key=lambda x: x["time"])
    return cues


def segments_to_table(segments):
    rows = []
    for i, s in enumerate(segments, start=1):
        start = float(s["start"])
        end = float(s["end"])
        label = str(s["label"])
        rows.append(
            {
                "#": i,
                "Section": label,
                "Start": format_time(start),
                "End": format_time(end),
                "Length (s)": round(end - start, 1),
            }
        )
    return rows


def cues_to_table(cues):
    rows = []
    for i, c in enumerate(cues, start=1):
        rows.append(
            {
                "#": i,
                "Cue time": format_time(float(c["time"])),
                "Text": c["text"],
                "Type": c["type"],
                "Target section": format_time(float(c["target_section_start"])),
            }
        )
    return rows


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Загрузи трек → получи структуру (intro/verse/chorus/bridge/outro) → получи базовые подсказки барабанщику.")

with st.sidebar:
    st.header("Настройки")
    lead_seconds = st.slider("За сколько секунд предупреждать о секции", 1.0, 12.0, 4.0, 0.5)
    min_segment_seconds = st.slider("Минимальная длина секции (сек)", 4.0, 20.0, 8.0, 1.0)
    st.divider()
    st.write("Поддержка форматов: mp3 / wav / m4a / ogg (если ffmpeg установлен).")

uploaded = st.file_uploader("Загрузи аудиофайл", type=["mp3", "wav", "m4a", "ogg", "flac", "aac"])

if not uploaded:
    st.info("Загрузи файл, чтобы начать.")
    st.stop()

# Работаем во временной папке
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)

    input_path = tmpdir / uploaded.name
    input_path.write_bytes(uploaded.getbuffer())

    wav_path = tmpdir / "input_44100_mono.wav"

    with st.spinner("Конвертирую в WAV 44.1kHz mono..."):
        try:
            to_wav_44100_mono(str(input_path), str(wav_path))
        except Exception as e:
            st.error("Не удалось конвертировать аудио. Проверь, что ffmpeg доступен в packages.txt.")
            st.exception(e)
            st.stop()

    st.success("Конвертация готова ✅")

    # Плеер исходника
    st.subheader("Прослушивание")
    st.audio(str(input_path), format=uploaded.type if hasattr(uploaded, "type") else None)

    # Анализ структуры
    with st.spinner("Анализирую структуру трека..."):
        try:
            segments = analyze_structure(
                audio_path=str(wav_path),
                sr=22050,  # внутри анализатора
                min_segment_seconds=float(min_segment_seconds),
            )
        except Exception as e:
            st.error("Ошибка анализа структуры.")
            st.exception(e)
            st.stop()

    if not segments:
        st.warning("Секции не найдены (попробуй уменьшить 'Минимальная длина секции').")
        st.stop()

    duration = read_duration_seconds(str(wav_path))

    # Генерация подсказок
    cues = build_basic_drummer_cues(segments, lead_seconds=float(lead_seconds))

    # Вывод
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Структура")
        st.write(f"Длина трека: **{format_time(duration)}**")
        st.dataframe(segments_to_table(segments), use_container_width=True)

    with col2:
        st.subheader("Подсказки (базовые)")
        st.dataframe(cues_to_table(cues), use_container_width=True)

        # Экспорт подсказок в .txt
        lines = []
        for c in cues:
            lines.append(f"{format_time(float(c['time']))}\t{c['text']}")
        cues_txt = "\n".join(lines)

        st.download_button(
            label="Скачать подсказки .txt",
            data=cues_txt.encode("utf-8"),
            file_name="drummer_cues.txt",
            mime="text/plain",
        )

    st.divider()
    st.subheader("JSON (для дальнейшей генерации cue-track / TTS / кликов)")
    st.json(
        {
            "duration_seconds": duration,
            "segments": segments,
            "cues": cues,
            "settings": {
                "lead_seconds": float(lead_seconds),
                "min_segment_seconds": float(min_segment_seconds),
            },
        }
    )
