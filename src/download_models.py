import os
from pathlib import Path

import requests
import streamlit as st

MODEL_DIR = Path("models")
ONNX_PATH = MODEL_DIR / "ru.onnx"
JSON_PATH = MODEL_DIR / "ru.onnx.json"

# HuggingFace (Rhasspy Piper voices) — ru_RU-irina-medium
ONNX_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx"
JSON_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json"


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dst, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


@st.cache_resource
def ensure_piper_model() -> str:
    """
    Streamlit Cloud: скачиваем модель на первом запуске и кешируем.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not ONNX_PATH.exists() or not JSON_PATH.exists():
        with st.spinner("Скачиваю голосовую модель Piper (первый запуск)..."):
            if not ONNX_PATH.exists():
                _download(ONNX_URL, ONNX_PATH)
            if not JSON_PATH.exists():
                _download(JSON_URL, JSON_PATH)

    return str(ONNX_PATH)
