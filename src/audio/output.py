"""Audio output player using Pygame mixer."""

import io
import pygame
from src.agent.state import is_agent_speaking, interruption_event

pygame.mixer.init()

def play_audio_bytes(audio_data: bytes) -> None:
    """Loads in-memory MP3 bytes and plays them, listening for barge-in signals."""
    if not audio_data:
        return

    try:
        buffer = io.BytesIO(audio_data)
        pygame.mixer.music.load(buffer)
        
        # Lock the state: Agent is now talking
        is_agent_speaking.set()
        interruption_event.clear()
        
        pygame.mixer.music.play()

        # Tick loop block: continuously check the cross-thread cancellation token
        while pygame.mixer.music.get_busy():
            if interruption_event.is_set():
                pygame.mixer.music.stop()
                print("\n  [ 🛑 AUDIO HALTED - BARGE-IN EXECUTED ]")
                break
            pygame.time.Clock().tick(20)

    except Exception as e:
        print(f"\n[Playback Error] Failed to play audio: {e}")
    finally:
        # Release the state lock
        is_agent_speaking.clear()