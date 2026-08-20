"""Speech-to-Text module using Groq Whisper API."""

import io
import os
import time
import wave
import numpy as np
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# A blocklist of known Whisper silence hallucinations
GHOST_PHRASES = {
    "thank you.", "thank you", "thanks for watching.", "thanks for watching!",
    "you", "thanks.", "subtitles by the amara.org community", "subscribe."
}

def transcribe_audio(audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
    if audio_chunk is None or len(audio_chunk) == 0:
        return ""

    audio_int16 = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    buffer.name = "audio.wav"
    buffer.seek(0)

    try:
        # Start the real network latency timer
        start_time = time.perf_counter()
        
        response = client.audio.transcriptions.create(
            file=("audio.wav", buffer.read()),
            model="whisper-large-v3-turbo",
            language="en",
            # The prompt steers the model away from YouTube subtitles and toward our domain
            prompt="Customer ordering food from a restaurant menu.",
        )
        
        latency = time.perf_counter() - start_time
        text = response.text.strip()
        
        # Check if the output is an exact match for a known hallucination
        if text.lower() in GHOST_PHRASES:
            return ""
            
        # Print the ACTUAL time it took Groq to process the request
        if text:
            print(f"  [API Latency: {latency:.2f} seconds]")
            
        return text

    except Exception as e:
        print(f"\n[STT Error] {e}")
        return ""