import os
import platform
import subprocess
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()
# 1. Initialize the Groq Raw Client (required for STT + TTS)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def transcribe_audio_turbo(file_path: str) -> str:
    """Transcribes audio file using ultra-fast whisper-large-v3-turbo."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at: {file_path}")
        
    with open(file_path, "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
            file=(os.path.basename(file_path), file.read()),
            model="whisper-large-v3-turbo",
            response_format="text" # Returns raw string directly
        )
    return transcription


def _chunk_text(text: str, limit: int = 180) -> list[str]:
    """Split text into <=limit chunks for Orpheus (200 char max per request)."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= limit:
            current = f"{current} {s}".strip()
        else:
            if current:
                chunks.append(current)
            # hard-split very long single sentence
            while len(s) > limit:
                chunks.append(s[:limit])
                s = s[limit:]
            current = s
    if current:
        chunks.append(current)
    return chunks or [text[:limit]]


def text_to_speech(text: str, output_path: str = "ai_response.wav",
                   model: str = "canopylabs/orpheus-v1-english",
                   voice: str = "troy") -> str:
    """Synthesizes text to speech via Groq Orpheus TTS. Returns output file path."""
    import wave
    chunks = _chunk_text(text)
    tmp_files = []
    try:
        for i, chunk in enumerate(chunks):
            part = f".tts_part_{i}.wav" if len(chunks) > 1 else output_path
            resp = groq_client.audio.speech.create(
                input=chunk,
                model=model,
                voice=voice,
                response_format="wav",
            )
            resp.write_to_file(part)
            tmp_files.append(part)

        if len(tmp_files) > 1:
            # Concatenate wav parts (same params) via stdlib
            with wave.open(tmp_files[0], 'rb') as w0:
                params = w0.getparams()
            with wave.open(output_path, 'wb') as out:
                out.setparams(params)
                for f in tmp_files:
                    with wave.open(f, 'rb') as w:
                        out.writeframes(w.readframes(w.getnframes()))
            for f in tmp_files:
                os.remove(f)
        return output_path
    finally:
        # cleanup leftovers on error path only if combined file exists
        pass


def play_audio(file_path: str):
    """Plays an audio file using OS-native player. Falls back to just saving."""
    # NOTE: winsound below is Windows-only stdlib. It's imported lazily
    # inside the Windows branch, so macOS/Linux never touch it.
    # macOS -> afplay, Windows -> winsound, Linux -> aplay/mpg123/ffplay.
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["afplay", file_path], check=False)
        elif system == "Windows":
            # winsound is Windows-only, import lazily
            import winsound
            winsound.PlaySound(file_path, winsound.SND_FILENAME)
        else:  # Linux and others
            # Try common CLI players in order
            for player in (["aplay", file_path], ["mpg123", file_path], ["ffplay", "-nodisp", "-autoexit", file_path]):
                try:
                    subprocess.run(player, check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
            print(f"⚠️ No audio player found. Saved to: {file_path}")
    except Exception as e:
        print(f"⚠️ Could not auto-play audio ({e}). Saved to: {file_path}")

def speak_fallback_mac(text: str, output_path: str = "ai_response.aiff") -> str:
    """Offline macOS fallback using `say` (no API key / terms needed)."""
    subprocess.run(["say", "-o", output_path, text], check=True)
    subprocess.run(["afplay", output_path], check=False)
    return output_path


# 2. Setup LangChain LLM and Prompt Harness
# We pair turbo STT with a fast Llama model to keep latency low.
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.5
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise voice assistant. Give brief, natural answers suitable for speech."),
    ("user", "{user_speech_text}")
])

# 3. Create the LangChain Expression Language (LCEL) Chain
chain = prompt | llm | StrOutputParser()

# 4. Execution Pipeline: text in -> voice out
if __name__ == "__main__":
    import sys

    # Allow: python voice_test.py "your text here" OR interactive input
    if len(sys.argv) > 1:
        user_text = " ".join(sys.argv[1:])
    else:
        user_text = input("💬 You (text): ").strip()

    if not user_text:
        print("❌ No input text provided.")
        raise SystemExit(1)

    try:
        print(f"\n🤖 Processing with LLM...")
        ai_response = chain.invoke({"user_speech_text": user_text})

        print(f"\n📢 AI Response: {ai_response}")

        print("\n🔊 Synthesizing voice with orpheus...")
        try:
            out_file = text_to_speech(ai_response)
            print(f"💾 Saved to: {out_file}")
            play_audio(out_file)
        except Exception as tts_e:
            # Most common: org hasn't accepted Orpheus terms at
            # https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english
            print(f"⚠️ Groq TTS failed ({tts_e}). Falling back to macOS `say`...")
            out_file = speak_fallback_mac(ai_response)
            print(f"💾 Saved to: {out_file}")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
