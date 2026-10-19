import wave
from piper import PiperVoice
import requests as r
import httpx
import sounddevice as sd
from piper import SynthesisConfig



base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high"
files = ["en_US-lessac-high.onnx", "en_US-lessac-high.onnx.json"]
base_path = "src/rough/"
with httpx.Client(follow_redirects=True, timeout=120) as c:
    for name in files:
        url = f"{base}/{name}?download=true"
        with c.stream("GET", url) as r:
            r.raise_for_status()
            with open(base_path+name, "wb") as f:
                for chunk in r.iter_bytes(1024 * 64):
                    f.write(chunk)
        print(f"saved {name}")



cfg = SynthesisConfig(
    speaker_id=0,        # multi-speaker voices only (e.g. libritts_r 904 speakers)
    length_scale=1.0,    # <1 faster, >1 slower. 0.9-1.1 useful range. Same as demo's Length Scale
    noise_scale=0.667,   # voice variability / flat vs expressive. Default 0.667
    noise_w_scale=0.8,    # phoneme width jitter. Default 0.8. Lower = steadier pace
    sentence_silence=0.2 # pause between sentences, seconds
)

voice.synthesize(text, wav_file, syn_config=cfg)

voice = PiperVoice.load(base_path+"en_US-lessac-high.onnx",
                        config_path=base_path+"en_US-lessac-high.onnx.json")


text = """Alice was beginning to get very tired of sitting by her 
    sister on the bank, and of having nothing to do. Once or twice she had peeped into the book her sister was reading,
      but it had no pictures or conversations in it. And what is the use of a book, thought Alice,
        without pictures or conversations?"""

stream = sd.OutputStream(samplerate=voice.config.sample_rate, channels=1, dtype="float32")
stream.start()
for chunk in voice.synthesize(text):
    stream.write(chunk.audio_float_array)
stream.stop()
stream.close()