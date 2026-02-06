import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def run_adtof_to_midi(audio_wav: str, out_midi: str) -> None:
    """
    ADTOF-based transcription → MIDI.
    В разных окружениях CLI может отличаться, поэтому пробуем несколько вариантов.

    Если у тебя в логах будет "command not found" или "unrecognized arguments",
    просто пришли лог — я поправлю candidates под твою сборку.
    """
    out_midi = str(out_midi)
    Path(out_midi).parent.mkdir(parents=True, exist_ok=True)

    candidates = [
        ["python", "-m", "adtof_plus_drum_transcription", "--audio", audio_wav, "--out", out_midi],
        ["python", "-m", "adtof_plus_drum_transcription", "--input", audio_wav, "--output", out_midi],
        ["adtof_plus_drum_transcription", "--audio", audio_wav, "--out", out_midi],
        ["adtof_plus_drum_transcription", "--input", audio_wav, "--output", out_midi],
    ]

    last_err = None
    for cmd in candidates:
        try:
            _run(cmd)
            return
        except Exception as e:
            last_err = e

    raise RuntimeError(
        "Не удалось запустить ADTOF транскрипцию. "
        "Проверь установку adtof_plus_drum_transcription и поправь команду в src/transcribe_adtof.py.\n"
        f"Последняя ошибка: {last_err}"
    )
