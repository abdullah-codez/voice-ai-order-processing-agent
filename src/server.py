"""FastAPI WebSocket Server for Real-Time Voice AI (Pure Streaming)."""

import json
import os
import asyncio
import numpy as np
import io
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

from src.agent.llm import generate_agent_response_stream

load_dotenv()
app = FastAPI()

# --- Color Constants for Terminal ---
C_STT = '\033[92m'   # Green
C_TTS = '\033[95m'   # Magenta
C_SYS = '\033[94m'   # Blue
C_ERR = '\033[91m'   # Red
C_END = '\033[0m'    # Reset

# Initialize the async Deepgram client
deepgram = AsyncDeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))

@app.get("/")
async def get():
    with open("src/web/index.html", "r") as f:
        return HTMLResponse(f.read())

async def process_intelligence(transcript: str, websocket: WebSocket):
    """Streams text from Groq, generates TTS, and sends complete sentence MP3s."""
    try:
        # Stream the LLM sentence by sentence
        async for sentence in generate_agent_response_stream(transcript):
            
            # 1. Send the text to the frontend immediately
            await websocket.send_text(json.dumps({"action": "log", "message": f"<b>Agent:</b> {sentence}"}))
            
            tts_start = time.time()
            print(f"  {C_TTS}[TTS]{C_END} 🎙️ Synthesizing sentence: '{sentence[:30]}...'")
            
            # 2. Call the Deepgram TTS API
            stream = deepgram.speak.v1.audio.generate(
                text=sentence,
                model="aura-asteria-en",
                encoding="mp3"
            )
            
            # 3. Assemble the audio chunks on the server
            audio_buffer = io.BytesIO()
            async for chunk in stream:
                if isinstance(chunk, bytes):
                    audio_buffer.write(chunk)
            
            # 4. Send the complete, uncorrupted MP3 sentence to the browser
            complete_audio = audio_buffer.getvalue()
            if complete_audio:
                print(f"  {C_TTS}[TTS]{C_END} 🚀 Audio dispatched to browser in {time.time() - tts_start:.3f}s")
                await websocket.send_bytes(complete_audio)
                
    except asyncio.CancelledError:
        print(f"\n{C_SYS}[System]{C_END} 🛑 LLM/TTS stream forcefully halted (Barge-in).")
    except Exception as e:
        print(f"{C_ERR}[Intelligence Error]{C_END} {e}")
        
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"{C_SYS}🟢 Browser connected to FastAPI.{C_END}")
    
    # Tracks the active LLM/TTS stream so we can cancel it if the user interrupts
    current_intelligence_task = None 

    # --- THE PROACTIVE GREETING TRIGGER ---
    async def trigger_proactive_greeting():
        # We wait 0.5 seconds to ensure the browser's audio context has fully initialized
        await asyncio.sleep(0.2)
        print(f"{C_SYS}[System]{C_END} 🎙️ Triggering Proactive Greeting...")
        nonlocal current_intelligence_task
        
        # We simulate the user saying a hidden trigger phrase. 
        initial_trigger = "[SYSTEM_EVENT: Call connected. Initiate greeting protocol.]"
        
        # Fire off the first intelligence stream automatically!
        current_intelligence_task = asyncio.create_task(process_intelligence(initial_trigger, websocket))
        
    # Start the greeting timer without blocking the STT connection setup
    asyncio.create_task(trigger_proactive_greeting())
    # --------------------------------------

    # 1. Connect to Deepgram STT (Live streaming)
    async with deepgram.listen.v1.connect(
        model="nova-2",
        language="en",
        smart_format=True,
        encoding="linear16",
        sample_rate=16000,
        channels=1,
        endpointing=300, 
        keywords=[
            "Classic Beef Burger", 
            "Crispy Chicken Burger", 
            "Margherita Pizza", 
            "French Fries", 
            "Coca-Cola", 
            "Fresh Lemonade",
            "Abdullah's Cafe"
        ]
    ) as dg_connection:

        async def on_message(message):
            nonlocal current_intelligence_task
            try:
                sentence = message.channel.alternatives[0].transcript
                
                # Only process finalized thoughts from the user
                if len(sentence) > 0 and message.is_final:
                    confidence = message.channel.alternatives[0].confidence
                    
                    # Deterministic gating: Ignore coughs and filler words instantly
                    clean_sentence = sentence.strip().lower().replace(".", "").replace(",", "")
                    FILLER_WORDS = {"uh", "umm", "um", "ah", "eh", "oh", "mmm", "yeah", "mmh"}
                    
                    if confidence > 0.65 and clean_sentence not in FILLER_WORDS and len(clean_sentence) > 2:
                        print(f"\n{C_STT}🗣️ Customer:{C_END} {sentence}")
                        
                        # 1. Cancel the backend task if it's still generating
                        if current_intelligence_task and not current_intelligence_task.done():
                            print(f"{C_SYS}[System]{C_END} ⚡ Interrupting current agent speech...")
                            current_intelligence_task.cancel()
                        
                        # 2. ALWAYS tell the frontend to clear its audio queue!
                        await websocket.send_text(json.dumps({"action": "interrupt"}))
                        
                        # 3. Log the customer's text to the browser UI
                        await websocket.send_text(json.dumps({"action": "log", "message": f"<b>Customer:</b> {sentence}"}))
                        await websocket.send_text(json.dumps({"action": "log", "message": "<b>[System]</b> Agent thinking..."}))
                        
                        # 4. Fire off the new intelligence stream in the background
                        current_intelligence_task = asyncio.create_task(process_intelligence(sentence, websocket))
                        
            except Exception:
                # Silently ignore partial/empty chunks from Deepgram
                pass 

        async def on_error(error):
            print(f"{C_ERR}[Deepgram STT Error]{C_END} {error}")

        # Bind the event listeners
        dg_connection.on(EventType.MESSAGE, on_message)
        dg_connection.on(EventType.ERROR, on_error)

        # Start the Deepgram listening loop in the background
        listen_task = asyncio.create_task(dg_connection.start_listening())

        try:
            while True:
                # Receive raw float32 audio from browser
                data = await websocket.receive_bytes()
                
                # Convert float32 to int16 PCM instantly
                audio_buffer = np.frombuffer(data, dtype=np.float32)
                audio_int16 = (audio_buffer * 32767).astype(np.int16).tobytes()
                
                # Stream the audio chunk directly to Deepgram
                await dg_connection.send_media(audio_int16) 
                
        except WebSocketDisconnect:
            print(f"{C_SYS}🔴 Browser disconnected.{C_END}")
            listen_task.cancel()
            if current_intelligence_task:
                current_intelligence_task.cancel()
        except Exception as e:
            print(f"{C_ERR}WebSocket Error:{C_END} {e}")
            listen_task.cancel()