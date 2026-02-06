import subprocess
def to_wav_44100_mono(src, dst):
    subprocess.check_call(["ffmpeg","-y","-i",src,"-ac","1","-ar","44100",dst])
