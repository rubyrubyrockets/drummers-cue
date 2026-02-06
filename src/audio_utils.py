import subprocess
from pathlib import Path


def to_wav_44100_mono(src_path: str, dst_path: str) -> None:
    src = Path(src_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ac", "1",
        "-ar", "44100",
        "-f", "wav",
        str(dst),
    ]
    subprocess.check_call(cmd)
