"""Text-to-Speech generation using Microsoft Edge neural voices."""

import asyncio
import edge_tts

DEFAULT_VOICE = "en-US-AriaNeural"  # Natural, conversational voice


async def _generate_audio_async(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    audio_stream = bytearray()
    
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_stream.extend(chunk["data"])
            
    return bytes(audio_stream)


def synthesize_speech(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """
    Synchronous wrapper for generating speech bytes from text.
    Handles running the async event loop safely.
    """
    if not text.strip():
        return b""
        
    try:
        # Create a new event loop if called within a worker thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_bytes = loop.run_until_complete(_generate_audio_async(text, voice))
        loop.close()
        return audio_bytes
    except Exception as e:
        print(f"\n[TTS Error] Speech synthesis failed: {e}")
        return b""