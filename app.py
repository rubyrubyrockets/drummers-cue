import os
import tempfile
from pathlib import Path

import streamlit as st
from pydub import AudioSegment

from src.audio_utils import to_wav_44100_mono
from src.structure_allin1 import analyze_structure_allin1
from src.transcribe_adtof import run_adtof_to_midi
from src.drum_features import load_drum_notes, compute_section_features, compute_diffs
from src.cue_builder import build_cues_with_drum_info
from src.render_cues import render_cue_track
from src.constants import DEFAULT_RU_PIPER_MODEL


st.set_page_config(page_title="Drum Cues Generator", layout="wide")


def ensure_models_hint(path: str):
    p = Path(path)
    if not p.exists():
        st.warning(
            "Piper модель не найдена по указанному пути. "
            "Положи .onnx (и желательно .onnx.json) в папку ./models/ и укажи путь."
        )


def main():
    st.title("🎧 Drum Cues Generator (Streamlit)")
    st.caption("Загрузи трек → анализ структуры + транскрипция барабанов → mp3 с подсказками заранее.")

    colL, colR = st.columns([1.1, 1])

    with colL:
        uploaded = st.file_uploader("Аудио файл", type=["mp3", "wav", "m4a", "flac", "aac", "ogg"])
        lead_bars = st.slider("Предупреждать за (тактов)", 1, 8, 2)
        assume_44 = st.checkbox("Считать размер 4/4 (для расчёта тактов)", value=True)

        use_drums_stem = st.checkbox("Использовать drums stem (Demucs) для транскрипции", value=True)
        min_hits_per_class = st.slider("Мин. ударов класса в секции, чтобы считать что он 'присутствует'", 1, 20, 3)

        density_threshold_silence = st.slider("Порог 'почти без барабанов' (notes/sec)", 0.0, 3.0, 0.25, 0.05)

    with colR:
        st.subheader("Голос подсказок")
        piper_model_path = st.text_input("Piper .onnx путь", value=DEFAULT_RU_PIPER_MODEL)
        ensure_models_hint(piper_model_path)
        cue_gain_db = st.slider("Громкость подсказок (dB)", -18, 18, 0)

        st.subheader("Экспорт")
        out_bitrate = st.selectbox("MP3 bitrate", ["128k", "192k", "256k"], index=1)

    if not uploaded:
        st.info("Загрузи трек, выбери параметры и нажми “Сгенерировать”.")
        return

    if st.button("🚀 Сгенерировать", type="primary"):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src_path = td / uploaded.name
            src_path.write_bytes(uploaded.getbuffer())

            wav_path = td / "input.wav"
            with st.spinner("Конвертация в WAV 44.1k mono..."):
                to_wav_44100_mono(str(src_path), str(wav_path))

            audio = AudioSegment.from_wav(str(wav_path))
            st.write(f"Длина трека: **{audio.duration_seconds:.1f} сек**")

            with st.spinner("Анализ структуры (all-in-one-fix)..."):
                analysis = analyze_structure_allin1(str(wav_path))

            segments = analysis["segments"]
            tempo = analysis["tempo"]

            st.success(f"Найден темп: **{tempo:.1f} BPM**, сегментов: **{len(segments)}**")

            # Для транскрипции используем либо исходник, либо drums stem (если доступен)
            transcribe_wav = wav_path
            if use_drums_stem and analysis.get("drums_stem_wav"):
                transcribe_wav = Path(analysis["drums_stem_wav"])
                st.write("Транскрипция будет по **drums stem** (Demucs).")
            else:
                st.write("Транскрипция будет по **исходному** аудио (mono).")

            midi_path = td / "drums.mid"
            with st.spinner("Транскрипция барабанов (ADTOF → MIDI)..."):
                run_adtof_to_midi(audio_wav=str(transcribe_wav), out_midi=str(midi_path))

            notes = load_drum_notes(str(midi_path))

            with st.spinner("Извлечение фич по секциям..."):
                feats = compute_section_features(
                    notes=notes,
                    segments=segments,
                    min_hits_per_class=min_hits_per_class
                )
                diffs = compute_diffs(feats)

            with st.spinner("Генерация подсказок..."):
                cues = build_cues_with_drum_info(
                    segments=segments,
                    tempo=tempo,
                    lead_bars=lead_bars,
                    diffs=diffs,
                    feats=feats,
                    density_threshold_silence=density_threshold_silence,
                    assume_44=assume_44,
                )

            st.subheader("Подсказки (таймкоды)")
            st.dataframe(
                [{"t (сек)": round(c["t_ms"] / 1000, 2), "text": c["text"]} for c in cues],
                use_container_width=True
            )

            with st.spinner("Рендер mp3 подсказок (Piper TTS + overlay)..."):
                cue_track = render_cue_track(
                    duration_ms=len(audio),
                    cues=cues,
                    piper_model_path=piper_model_path,
                    cue_gain_db=cue_gain_db
                )

            out_mp3 = td / "drum_cues.mp3"
            cue_track.export(str(out_mp3), format="mp3", bitrate=out_bitrate)

            st.success("Готово ✅")
            st.audio(str(out_mp3))
            st.download_button(
                "⬇️ Скачать drum_cues.mp3",
                data=out_mp3.read_bytes(),
                file_name="drum_cues.mp3",
                mime="audio/mpeg"
            )


if __name__ == "__main__":
    main()
