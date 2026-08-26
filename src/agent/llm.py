"""Language Model and AI processing logic using Groq API (Synchronized LangGraph)."""

import json
import os
import re
import time
import asyncio
from dotenv import load_dotenv
from groq import AsyncGroq

# Import our new graph, state, and phase-based prompts
from src.agent.prompts import get_phase_prompt
from src.agent.graph import graph_app
from src.agent.state import AgentState

load_dotenv()
client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_NAME = "openai/gpt-oss-120b"

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?,])\s+(?=[A-Za-z0-9\"'])")

# --- Color Constants for Terminal ---
C_LLM = '\033[96m'   
C_TOOL = '\033[93m'  
C_ERR = '\033[91m'   
C_END = '\033[0m'    

async def generate_agent_response_stream(transcript: str, session_state: AgentState):
    """Processes the LangGraph state FIRST, then streams audio based on the NEW state."""
    
    # 1. Append user audio transcript to the graph's history
    session_state["conversation_history"].append({"role": "user", "content": transcript})
    
    print(f"\n{C_LLM}[Pipeline]{C_END} ⚙️ User spoke. Updating State Machine first...")
    stream_start = time.time()

    # 2. RUN THE BRAIN FIRST (Wait for it to finish!)
    try:
        new_state = await graph_app.ainvoke(session_state)
        session_state.update(new_state)
        print(f"  {C_TOOL}[LangGraph]{C_END} ✅ Graph pass complete in {time.time() - stream_start:.2f}s. Next Phase: [{session_state['current_phase'].upper()}]")
    except Exception as e:
        print(f"  {C_ERR}[LangGraph Error]{C_END} {e}")

    # 3. Fetch prompt based on the newly updated state and COMPLETE cart details
    state_summary = (
        f"Cart: {session_state.get('order_items', [])} | Total: ${session_state.get('total_amount', 0.0):.2f}\n"
        f"Payment Method: {session_state.get('payment_method', 'Not provided yet')}\n"
        f"Fulfillment: {session_state.get('fulfillment_type', 'Not provided yet')}\n"
        f"Address: {session_state.get('delivery_address', 'Not provided yet')}\n"
        f"Customer Contact: {session_state.get('customer_name', 'No Name')} / {session_state.get('customer_phone', 'No Phone')}"
    )
    
    system_instruction = get_phase_prompt(session_state["current_phase"], state_summary)
    
    messages_for_voice = [{"role": "system", "content": system_instruction}] + session_state["conversation_history"]

    print(f"  {C_LLM}[LLM Stream]{C_END} 🎙️ Requesting Speech (Phase: {session_state['current_phase'].upper()})...")
    first_token_received = False

    # 4. INSTANTLY STREAM THE VERBAL RESPONSE
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages_for_voice,
        temperature=0.3,
        max_tokens=350,
        stream=True,
    )

    buffer = ""
    full_response = ""

    try:
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                if not first_token_received:
                    first_token_received = True
                    print(f"  {C_LLM}[LLM]{C_END} ⚡ Time-to-First-Token: {time.time() - stream_start:.3f}s")
                    
                buffer += token
                full_response += token

                parts = SENTENCE_SPLIT_REGEX.split(buffer)
                if len(parts) > 1:
                    for sentence in parts[:-1]:
                        if sentence.strip():
                            yield sentence.strip()
                    buffer = parts[-1]

        if buffer.strip():
            yield buffer.strip()

        print(f"  {C_LLM}[LLM]{C_END} 🏁 Finished generating sentence. (Total LLM time: {time.time() - stream_start:.2f}s)")
        
        # 5. Save the COMPLETE agent's spoken response back into the history
        session_state["conversation_history"].append({"role": "assistant", "content": full_response})

    except asyncio.CancelledError:
        # 🛑 BARGE-IN DETECTED!
        # The user interrupted the agent. We save the partial thought so the LLM remembers getting cut off.
        if full_response.strip():
            session_state["conversation_history"].append({"role": "assistant", "content": full_response.strip() + "--"})
        raise  # Pass the error back up to server.py to formally kill the TTS process