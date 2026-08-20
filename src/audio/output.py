"""Audio output player using Pygame mixer."""

import io
import time
import pygame

# Initialize pygame mixer once at import time
pygame.mixer.init()


def play_audio_bytes(audio_data: bytes) -> None:
    """Loads in-memory MP3 audio bytes and blocks until playback finishes."""
    if not audio_data:
        return

    try:
        buffer = io.BytesIO(audio_data)
        pygame.mixer.music.load(buffer)
        pygame.mixer.music.play()

        # Block the worker thread until the sentence has finished playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(20)

    except Exception as e:
        print(f"\n[Playback Error] Failed to play audio: {e}")