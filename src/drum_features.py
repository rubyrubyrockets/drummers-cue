from typing import Any, Dict, List, Tuple

import numpy as np
import pretty_midi

# GM drum pitches (минимальная, но полезная группировка)
KICK = {35, 36}
SNARE = {38, 40}
HAT = {42, 44, 46}
TOMS = {41, 43, 45, 47, 48, 50}
CYMS = {49, 51, 52, 53, 55, 57, 59}


def drum_class(pitch: int) -> str:
    if pitch in KICK:
        return "kick"
    if pitch in SNARE:
        return "snare"
    if pitch in HAT:
        return "hihat"
    if pitch in TOMS:
        return "toms"
    if pitch in CYMS:
        return "cymbals"
    return "other"


def load_drum_notes(midi_path: str) -> List[Tuple[float, float, int, int]]:
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            for n in inst.notes:
                notes.append((float(n.start), float(n.end), int(n.pitch), int(n.velocity)))
    notes.sort(key=lambda x: x[0])
    return notes


def _section_features(notes, t0: float, t1: float) -> Dict[str, Any]:
    sec = [n for n in notes if (t0 <= n[0] < t1)]
    dur = max(1e-6, t1 - t0)

    counts = {"kick": 0, "snare": 0, "hihat": 0, "toms": 0, "cymbals": 0, "other": 0}
    vels = []
    for s, e, p, v in sec:
        counts[drum_class(p)] += 1
        vels.append(v)

    density = len(sec) / dur
    mean_vel = float(np.mean(vels)) if vels else 0.0

    return {
        "count": len(sec),
        "duration": dur,
        "density": float(density),
        "mean_vel": mean_vel,
        "counts": counts,
    }


def compute_section_features(notes, segments: List[dict], min_hits_per_class: int = 3) -> List[Dict[str, Any]]:
    feats = []
    for seg in segments:
        f = _section_features(notes, float(seg["start"]), float(seg["end"]))
        present_stable = set(k for k, c in f["counts"].items() if c >= min_hits_per_class)
        f["present_stable"] = present_stable
        feats.append(f)
    return feats


def compute_diffs(feats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    diffs = []
    prev = None
    for cur in feats:
        if prev is None:
            diffs.append({"added": set(), "removed": set(), "density_delta": 0.0, "vel_delta": 0.0})
        else:
            added = cur["present_stable"] - prev["present_stable"]
            removed = prev["present_stable"] - cur["present_stable"]
            diffs.append({
                "added": added,
                "removed": removed,
                "density_delta": float(cur["density"] - prev["density"]),
                "vel_delta": float(cur["mean_vel"] - prev["mean_vel"]),
            })
        prev = cur
    return diffs
