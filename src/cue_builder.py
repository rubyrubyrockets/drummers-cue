from typing import Any, Dict, List

RU_PART = {
    "intro": "интро",
    "verse": "куплет",
    "chorus": "припев",
    "bridge": "бридж",
    "outro": "аутро",
    "silence": "тишина",
    "instrumental": "инструментал",
    "inst": "инструментал",
    "transition": "переход",
    "break": "брейк",
    "drop": "дроп",
}

RU_INST = {
    "kick": "бочка",
    "snare": "рабочий",
    "hihat": "хай-хэт",
    "toms": "томы",
    "cymbals": "тарелки",
    "other": "прочее",
}


def ru_label(lbl: str) -> str:
    l = (lbl or "").strip().lower()
    return RU_PART.get(l, lbl or "часть")


def ru_inst_set(s: set) -> str:
    return ", ".join(RU_INST.get(x, x) for x in sorted(s))


def _lead_seconds_from_bars(tempo_bpm: float, lead_bars: int, assume_44: bool) -> float:
    if not assume_44:
        # fallback: просто ~2с на такт
        return float(lead_bars) * 2.0
    beat_s = 60.0 / max(1e-6, float(tempo_bpm))
    bar_s = 4.0 * beat_s
    return float(lead_bars) * bar_s


def _section_phrase(label: str, diff: Dict[str, Any], feat: Dict[str, Any], density_threshold_silence: float) -> str:
    parts = []

    if feat["density"] <= density_threshold_silence:
        parts.append("почти без барабанов")

    if diff["density_delta"] > 0.9:
        parts.append("плотнее")
    elif diff["density_delta"] < -0.9:
        parts.append("реже")

    if diff["added"]:
        parts.append(f"входит: {ru_inst_set(diff['added'])}")
    if diff["removed"]:
        parts.append(f"уходит: {ru_inst_set(diff['removed'])}")

    tail = (". " + "; ".join(parts)) if parts else ""
    return f"{ru_label(label)}{tail}"


def build_cues_with_drum_info(
    segments: List[dict],
    tempo: float,
    lead_bars: int,
    diffs: List[dict],
    feats: List[dict],
    density_threshold_silence: float = 0.25,
    assume_44: bool = True,
) -> List[Dict[str, Any]]:
    cues: List[Dict[str, Any]] = []
    if not segments:
        return cues

    lead_s = _lead_seconds_from_bars(tempo, lead_bars, assume_44)

    first_phrase = _section_phrase(segments[0]["label"], diffs[0], feats[0], density_threshold_silence)
    cues.append({"t_ms": 0, "text": f"Старт. {first_phrase}."})

    for i in range(1, len(segments)):
        seg = segments[i]
        t_boundary = float(seg["start"])
        t_cue = max(0.0, t_boundary - lead_s)

        phrase = _section_phrase(seg["label"], diffs[i], feats[i], density_threshold_silence)
        cues.append({"t_ms": int(t_cue * 1000), "text": f"Через {lead_bars} такта — {phrase}."})

    cues.sort(key=lambda x: x["t_ms"])

    # небольшой дедуп по времени
    out = []
    last_t = -10_000
    for c in cues:
        if c["t_ms"] - last_t >= 250:
            out.append(c)
            last_t = c["t_ms"]
    return out
