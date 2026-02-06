import numpy as np
import librosa


def analyze_structure(
    audio_path: str,
    sr: int = 22050,
    min_segment_seconds: float = 8.0,
):
    """
    Cloud-safe структурный анализ трека.
    Возвращает список сегментов:
    [
        {
            "label": "intro|verse|chorus|bridge|outro",
            "start": float (seconds),
            "end": float (seconds)
        }
    ]
    """

    # Загружаем аудио
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # ===== Feature extraction =====
    # Хрома + спектральная энергия → хорошо работает для структуры
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]

    # Усредняем фичи
    features = np.vstack([
        chroma,
        rms[np.newaxis, :]
    ])

    # ===== Self-similarity / novelty =====
    # Косинусная дистанция между окнами
    S = librosa.segment.recurrence_matrix(
        features,
        mode="affinity",
        metric="cosine",
        sym=True
    )

    # Novelty curve
    novelty = librosa.segment.novelty(
        S,
        kernel_size=32,
        lag=1
    )

    # ===== Segment boundaries =====
    peaks = librosa.util.peak_pick(
        novelty,
        pre_max=8,
        post_max=8,
        pre_avg=8,
        post_avg=8,
        delta=0.05,
        wait=8
    )

    # Переводим в секунды
    times = librosa.frames_to_time(peaks, sr=sr)

    # Добавляем начало и конец
    boundaries = [0.0] + times.tolist() + [duration]

    # ===== Build segments =====
    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end - start < min_segment_seconds:
            continue

        segments.append({
            "start": float(start),
            "end": float(end),
        })

    # ===== Label heuristics =====
    labeled = []
    for i, seg in enumerate(segments):
        if i == 0:
            label = "intro"
        elif i == len(segments) - 1:
            label = "outro"
        elif i % 4 == 0:
            label = "chorus"
        elif i % 4 == 2:
            label = "bridge"
        else:
            label = "verse"

        labeled.append({
            "label": label,
            "start": seg["start"],
            "end": seg["end"]
        })

    return labeled
