"""FastAPI WebSocket Server for real-time Voice AI."""

import json
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from src.voice.vad import VoiceActivityDetector
from src.voice.segmenter import SpeechSegmenter
from src.speech.stt import transcribe_audio
from src.agent.llm import generate_agent_response_stream
from src.speech.tts import synthesize_speech

load_dotenv()
app = FastAPI()

@app.get("/")
async def get():
    with open("src/web/index.html", "r") as f:
        return HTMLResponse(f.read())

async def intelligence_consumer(websocket: WebSocket, text_queue: asyncio.Queue):
    """Background worker with deterministic STT filtering and streaming."""
    FILLER_WORDS = {"uh", "umm", "um", "ah", "eh", "oh", "mmm", "yeah", "mmh"}
    
    while True:
        transcript, confidence = await text_queue.get()
        
        if not transcript:
            text_queue.task_done()
            continue

        # 1. The Confidence Gate: Drop hallucinations (e.g., coughs parsed as "You.")
        if confidence < 0.65:
            await websocket.send_text(json.dumps({
                "action": "log", 
                "message": f"<i>[Ignored Low Confidence ({confidence:.2f}): '{transcript}']</i>"
            }))
            text_queue.task_done()
            continue

        # 2. The Filler Gate: Drop isolated hesitation sounds
        clean_transcript = transcript.strip().lower().replace(".", "").replace(",", "")
        if clean_transcript in FILLER_WORDS or len(clean_transcript) <= 2:
            await websocket.send_text(json.dumps({
                "action": "log", 
                "message": f"<i>[Ignored Filler: '{transcript}']</i>"
            }))
            text_queue.task_done()
            continue
            
        # 3. Valid Intent Confirmed -> Interrupt audio playback
        await websocket.send_text(json.dumps({"action": "interrupt"}))
        await websocket.send_text(json.dumps({"action": "log", "message": f"<b>Customer:</b> {transcript}"}))
        await websocket.send_text(json.dumps({"action": "log", "message": "<b>[System]</b> Agent thinking..."}))

        # 4. Stream the LLM and TTS sequentially
        async for sentence in generate_agent_response_stream(transcript):
            
            # Send text instantly to the UI
            await websocket.send_text(json.dumps({"action": "log", "message": f"<b>Agent:</b> {sentence}"}))
            
            # Synthesize and send just that sentence
            audio_bytes = await asyncio.to_thread(synthesize_speech, sentence)
            if audio_bytes:
                await websocket.send_bytes(audio_bytes)
                
        text_queue.task_done()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    vad = VoiceActivityDetector()
    # 1.2-second pause allowance to fix sentence splitting
    segmenter = SpeechSegmenter(silence_duration_ms=1200, min_speech_duration_ms=200)
    VAD_CHUNK_SIZE = 256 
    
    text_queue = asyncio.Queue()
    consumer_task = asyncio.create_task(intelligence_consumer(websocket, text_queue))
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer = np.frombuffer(data, dtype=np.float32)

            for i in range(0, len(audio_buffer), VAD_CHUNK_SIZE):
                small_chunk = audio_buffer[i:i + VAD_CHUNK_SIZE]
                if len(small_chunk) < VAD_CHUNK_SIZE:
                    small_chunk = np.concatenate((small_chunk, np.zeros(VAD_CHUNK_SIZE - len(small_chunk), dtype=np.float32)))

                probability, is_speech = vad.process(small_chunk)
                utterance = segmenter.process(small_chunk, is_speech)

                if utterance is not None:
                    # Offload Deepgram STT to a thread
                    transcript, confidence = await asyncio.to_thread(transcribe_audio, utterance, 16000)
                    if transcript:
                        # Push the tuple to the queue
                        await text_queue.put((transcript, confidence))

    except WebSocketDisconnect:
        print("Client disconnected.")
        consumer_task.cancel()