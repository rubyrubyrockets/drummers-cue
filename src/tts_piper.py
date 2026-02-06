import subprocess
from pathlib import Path


def piper_tts(text: str, out_wav: str, model_path: str) -> None:
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "piper",
        "--model", model_path,
        "--output_file", str(out),
        "--text", text,
    ]
    subprocess.check_call(cmd)
