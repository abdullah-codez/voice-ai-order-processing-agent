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
import src.agent.state as state  # Import the whole module to modify the string

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 256

vad = VoiceActivityDetector()
segmenter = SpeechSegmenter(silence_duration_ms=600)
utterance_queue = queue.Queue()
text_queue = queue.Queue()

FILLER_WORDS = {"um", "uh", "ah", "hmm", "hm", "oh", "like"}


def audio_callback(indata, frames, time, status):
    audio_chunk = indata[:, 0]
    probability, is_speech = vad.process(audio_chunk)
    utterance = segmenter.process(audio_chunk, is_speech)

    if utterance is not None:
        utterance_queue.put(utterance)


def is_semantic_echo(transcript: str, active_script: str) -> bool:
    """Calculates if the transcribed words overlap heavily with the agent's active script."""
    if not active_script:
        return False
        
    trans_words = set(transcript.lower().replace(".", "").replace(",", "").split())
    script_words = set(active_script.lower().replace(".", "").replace(",", "").split())
    
    if not trans_words:
        return False
        
    # If 40% or more of the words heard are currently being spoken by the agent, it's an echo
    overlap = trans_words.intersection(script_words)
    overlap_ratio = len(overlap) / len(trans_words)
    
    return overlap_ratio >= 0.40


def stt_evaluator_worker():
    while True:
        audio = utterance_queue.get()
        if audio is None: break

        transcript = transcribe_audio(audio, sample_rate=SAMPLE_RATE)
        
        if not transcript:
            utterance_queue.task_done()
            continue
            
        transcript_clean = transcript.lower().strip(".,!?")

        # --- THE BARGE-IN GATE WITH SEMANTIC ECHO CANCELLATION ---
        if state.is_agent_speaking.is_set():
            if transcript_clean in FILLER_WORDS or len(transcript_clean) <= 2:
                print(f"  [Ignored Background/Filler: '{transcript}']")
            elif is_semantic_echo(transcript, state.current_agent_script):
                print(f"  [Ignored Acoustic Bleed (Echo): '{transcript}']")
            else:
                print(f"\n> [⚡ INTERRUPT] Customer barged in: '{transcript}'")
                state.interruption_event.set()
                
                with text_queue.mutex:
                    text_queue.queue.clear()
                    
                text_queue.put(transcript)
        else:
            print(f"> Customer: {transcript}")
            text_queue.put(transcript)

        utterance_queue.task_done()


def intelligence_and_playback_worker():
    while True:
        transcript = text_queue.get()
        if transcript is None: break

        state.interruption_event.clear()

        print("  [Agent is thinking...]")
        agent_reply = generate_agent_response(transcript)

        if state.interruption_event.is_set():
            print("  [Thought process aborted due to late interruption]")
            text_queue.task_done()
            continue

        print(f"> Agent: {agent_reply}")
        
        # Load the agent's text into the global state for the STT thread to read
        state.current_agent_script = agent_reply
        
        print("  [Synthesizing voice...]")
        audio_bytes = synthesize_speech(agent_reply)

        if audio_bytes and not state.interruption_event.is_set():
            print("  [Playing audio...]\n" + "-" * 40)
            play_audio_bytes(audio_bytes)
            
        # Clear the script once playback finishes
        state.current_agent_script = ""

        text_queue.task_done()


def start_microphone():
    # Spawn both worker threads
    threading.Thread(target=stt_evaluator_worker, daemon=True).start()
    threading.Thread(target=intelligence_and_playback_worker, daemon=True).start()

    print("========================================")
    print(" Voice Agent Live (Full-Duplex Barge-in)")
    print(" Speak to interrupt the agent at any time.")
    print("========================================\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.float32,
        blocksize=CHUNK_SIZE, callback=audio_callback,
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