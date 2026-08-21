"""Global concurrency synchronization states for the Voice Agent."""

import threading

# Signaled when the Pygame mixer is actively outputting audio
is_agent_speaking = threading.Event()

# Signaled by the STT worker to forcefully halt active generation/playback
interruption_event = threading.Event()

# Holds the exact text the agent is currently speaking
current_agent_script = ""