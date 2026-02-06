from pathlib import Path
import allin1fix


def analyze_structure_allin1(wav_path: str) -> dict:
    """
    Возвращает:
      tempo: float
      segments: list[{label,start,end}]
      drums_stem_wav: str|None
    """
    result = allin1fix.analyze(wav_path)

    # возможные ключи сегментов
    segments = None
    for key in ("segments", "functional_segments", "sections"):
        if key in result and result[key]:
            segments = result[key]
            break
    if segments is None:
        segments = []

    norm_segments = []
    for s in segments:
        norm_segments.append({
            "label": (s.get("label") or s.get("name") or "part"),
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
        })

    tempo = float(result.get("tempo") or result.get("bpm") or 120.0)

    # drums stem (если all-in-one-fix вернул путь)
    drums_stem_wav = None
    for key in ("drums_stem_wav", "drums_wav", "drums_stem", "drums"):
        if key in result and result[key]:
            drums_stem_wav = result[key]
            break

    if drums_stem_wav and not Path(str(drums_stem_wav)).exists():
        drums_stem_wav = None

    return {
        "tempo": tempo,
        "segments": norm_segments,
        "drums_stem_wav": drums_stem_wav,
    }
