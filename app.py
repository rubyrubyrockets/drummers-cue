import io
import zipfile
from datetime import datetime

import streamlit as st

st.set_page_config(page_title="Drummer Cues MVP", page_icon="🥁", layout="centered")

st.title("🥁 Drummer Cues — MVP")
st.caption("Загрузи трек → получи ZIP (пока без анализа). Следующий шаг — click + cue track.")

uploaded = st.file_uploader("Загрузи аудио (mp3/wav)", type=["mp3", "wav", "m4a", "aac", "flac"])

preset = st.selectbox(
    "Пресет подсказок (пока влияет только на метаданные)",
    ["Rock/Pop (default)", "Worship", "EDM", "Hip-Hop"],
)

generate = st.button("Generate ZIP", type="primary", disabled=uploaded is None)

if generate and uploaded:
    # читаем входной файл
    audio_bytes = uploaded.read()

    # создаём ZIP в памяти
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        # кладём оригинальный файл (как источник)
        z.writestr(f"input/{uploaded.name}", audio_bytes)

        # placeholder файлы, чтобы показать формат результата
        z.writestr(
            "README.txt",
            "MVP output.\n\nNext versions will include:\n- click.wav\n- cues.wav\n- markers.json\n"
        )
        z.writestr(
            "markers.json",
            f"""{{
  "song": "{uploaded.name}",
  "preset": "{preset}",
  "created_at": "{datetime.utcnow().isoformat()}Z",
  "markers": []
}}"""
        )

    mem.seek(0)

    st.success("Готово! Скачай ZIP.")
    st.download_button(
        "Download result.zip",
        data=mem,
        file_name="result.zip",
        mime="application/zip",
    )
