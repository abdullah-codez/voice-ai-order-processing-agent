import queue
import threading
import numpy as np
import sounddevice as sd

from src.voice.vad import VoiceActivityDetector
from src.voice.segmenter import SpeechSegmenter
from src.speech.stt import transcribe_audio
from src.agent.llm import generate_agent_response
from src.speech.tts import synthesize_speech
from src.audio.output import play_audio_bytes

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 256

vad = VoiceActivityDetector()
segmenter = SpeechSegmenter(silence_duration_ms=650)  # Snappier turn detection
utterance_queue = queue.Queue()

# State flag to prevent acoustic feedback
is_agent_speaking = threading.Event()


def audio_callback(indata, frames, time, status):
    if status:
        print(f"[Audio Status] {status}")

    # Ignore microphone input while the agent is speaking
    if is_agent_speaking.is_set():
        return

    audio_chunk = indata[:, 0]
    probability, is_speech = vad.process(audio_chunk)

    utterance = segmenter.process(audio_chunk, is_speech)

    if utterance is not None:
        duration = len(utterance) / SAMPLE_RATE
        print(f"\n[Detected Speech] {duration:.2f}s — Transcribing...")
        utterance_queue.put(utterance)


def voice_pipeline_worker():
    """Background worker executing the full audio-to-speech loop."""
    while True:
        audio = utterance_queue.get()
        if audio is None:
            break

        # 1. Speech to Text
        transcript = transcribe_audio(audio, sample_rate=SAMPLE_RATE)
        
        if transcript:
            print(f"> Customer: {transcript}")
            
            # 2. LLM Decision & Tool Execution
            print("  [Agent is thinking...]")
            agent_reply = generate_agent_response(transcript)
            print(f"> Agent: {agent_reply}")
            
            # 3. Text to Speech Synthesis
            print("  [Synthesizing voice...]")
            audio_bytes = synthesize_speech(agent_reply)
            
            # 4. Audio Playback with Feedback Prevention
            if audio_bytes:
                print("  [Playing audio...]\n" + "-" * 40)
                is_agent_speaking.set()  # Mute mic
                try:
                    play_audio_bytes(audio_bytes)
                finally:
                    is_agent_speaking.clear()  # Unmute mic
        else:
            print("> [Empty or Unclear Speech]\n")

        utterance_queue.task_done()


def start_microphone():
    worker_thread = threading.Thread(target=voice_pipeline_worker, daemon=True)
    worker_thread.start()

    print("========================================")
    print(" Voice AI Restaurant Agent (Optimized)")
    print(" Speak into your microphone...")
    print(" Press Ctrl+C to stop.")
    print("========================================\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.float32,
        blocksize=CHUNK_SIZE,
        callback=audio_callback,
    ):
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("\nStopping voice agent...")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    start_microphone()