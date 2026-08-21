"""Speech-to-Text translation using Deepgram Nova-2."""

import os
import io
import wave
import numpy as np
from dotenv import load_dotenv
from deepgram import DeepgramClient

load_dotenv() 

def transcribe_audio(audio_data: np.ndarray, sample_rate: int = 16000) -> tuple[str, float]:
    """Returns a tuple containing the (transcript, confidence_score)."""
    try:
        audio_int16 = (audio_data * 32767).astype(np.int16).tobytes()
        
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16)
            
        wav_bytes = wav_io.getvalue()
        
        deepgram = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))

        # Add the keywords parameter with our exact menu vocabulary
        response = deepgram.listen.v1.media.transcribe_file(
            request=wav_bytes,
            model="nova-2",
            language="en",
            smart_format=True,
            keywords=[
                "Classic Beef Burger", 
                "Crispy Chicken Burger", 
                "Margherita Pizza", 
                "French Fries", 
                "Coca-Cola", 
                "Fresh Lemonade",
                "Abdullah's Cafe"
            ]
        )
        
        alternative = response.results.channels[0].alternatives[0]
        return alternative.transcript, alternative.confidence
        
    except Exception as e:
        print(f"\n[STT Error] Deepgram failed: {e}")
        return "", 0.0