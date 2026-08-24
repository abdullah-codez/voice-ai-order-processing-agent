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
from src.agent.state import AgentState

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

async def process_intelligence(transcript: str, websocket: WebSocket, session_state: AgentState):
    """Streams true real-time PCM audio chunks directly to the browser."""
    try:
        async for sentence in generate_agent_response_stream(transcript, session_state):
            
            await websocket.send_text(json.dumps({"action": "log", "message": f"<b>Agent:</b> {sentence}"}))
            
            print(f"  {C_TTS}[TTS]{C_END} 🎙️ Synthesizing: '{sentence[:30]}...'")
            
            # Request Raw PCM Audio instead of MP3
            stream = deepgram.speak.v1.audio.generate(
                text=sentence,
                model="aura-asteria-en",
                encoding="linear16", 
                sample_rate=24000
            )
            
            # Instantly pipe the raw bytes to the WebSocket as they arrive!
            async for chunk in stream:
                if isinstance(chunk, bytes):
                    await websocket.send_bytes(chunk)
                
    except asyncio.CancelledError:
        # Expected behavior when the user interrupts
        pass
    except Exception as e:
        print(f"{C_ERR}[Intelligence Error]{C_END} {e}")
        
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"{C_SYS}🟢 Browser connected to FastAPI.{C_END}")
    
    # 1. CREATE ISOLATED STATE FOR THIS CONNECTION
    session_state: AgentState = {
        "conversation_history": [],
        "order_items": [],
        "payment_method": None,
        "fulfillment_type": None,
        "delivery_address": None,
        "customer_name": None,
        "customer_phone": None,
        "current_phase": "greeting",
        "total_amount": 0.0
    }
    
    current_intelligence_task = None 
    task_lock = asyncio.Lock()  # Protects task reassignment against race conditions

    async def trigger_proactive_greeting():
        await asyncio.sleep(0.2)
        print(f"{C_SYS}[System]{C_END} 🎙️ Triggering Proactive Greeting...")
        initial_trigger = "[SYSTEM_EVENT: Call connected. Initiate greeting protocol.]"
        
        nonlocal current_intelligence_task
        async with task_lock:
            current_intelligence_task = asyncio.create_task(process_intelligence(initial_trigger, websocket, session_state))
        
    asyncio.create_task(trigger_proactive_greeting())

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
                
                if len(sentence) > 0:
                    clean_sentence = sentence.strip().lower().replace(".", "").replace(",", "")
                    FILLER_WORDS = {"uh", "umm", "um", "ah", "eh", "oh", "mmm", "yeah", "mmh"}
                    
                    # --- 1. INSTANT AUDIO BARGE-IN (Interim Results) ---
                    if not message.is_final:
                        if len(clean_sentence) > 2 and clean_sentence not in FILLER_WORDS:
                            # Instantly mute the browser
                            await websocket.send_text(json.dumps({"action": "interrupt"}))
                            
                            # Cancel the backend task to stop TTS and save tokens
                            async with task_lock:
                                if current_intelligence_task and not current_intelligence_task.done():
                                    print(f"{C_SYS}[System]{C_END} ⚡ Instant Barge-in! Agent silenced.")
                                    current_intelligence_task.cancel()

                    # --- 2. LLM LOGIC PROCESSING (Final Results) ---
                    elif message.is_final:
                        confidence = message.channel.alternatives[0].confidence
                        
                        if confidence > 0.65 and clean_sentence not in FILLER_WORDS and len(clean_sentence) > 2:
                            print(f"\n{C_STT}🗣️ Customer:{C_END} {sentence}")
                            
                            async with task_lock:
                                # Double check cancellation
                                if current_intelligence_task and not current_intelligence_task.done():
                                    current_intelligence_task.cancel()
                                
                                # Fire the complete sentence to LangGraph
                                current_intelligence_task = asyncio.create_task(process_intelligence(sentence, websocket, session_state))
                            
                            await websocket.send_text(json.dumps({"action": "interrupt"}))
                            await websocket.send_text(json.dumps({"action": "log", "message": f"<b>Customer:</b> {sentence}"}))
                            await websocket.send_text(json.dumps({"action": "log", "message": "<b>[System]</b> Agent thinking..."}))
                            
            except Exception:
                pass 

        async def on_error(error):
            print(f"{C_ERR}[Deepgram STT Error]{C_END} {error}")

        dg_connection.on(EventType.MESSAGE, on_message)
        dg_connection.on(EventType.ERROR, on_error)

        listen_task = asyncio.create_task(dg_connection.start_listening())

        try:
            while True:
                data = await websocket.receive_bytes()
                audio_buffer = np.frombuffer(data, dtype=np.float32)
                audio_int16 = (audio_buffer * 32767).astype(np.int16).tobytes()
                await dg_connection.send_media(audio_int16) 
                
        except WebSocketDisconnect:
            print(f"{C_SYS}🔴 Browser disconnected.{C_END}")
            listen_task.cancel()
            if current_intelligence_task:
                current_intelligence_task.cancel()
        except Exception as e:
            print(f"{C_ERR}WebSocket Error:{C_END} {e}")
            listen_task.cancel()