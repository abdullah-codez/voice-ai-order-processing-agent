"""Language Model and AI processing logic using Groq API (Parallel Architecture)."""

import json
import os
import re
import asyncio
import time
import inspect
from dotenv import load_dotenv
from groq import AsyncGroq
from src.agent.prompts import get_system_prompt
from src.agent.tools import TOOL_MAP, TOOLS

load_dotenv()

client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_NAME = "openai/gpt-oss-20b"

conversation_history = [{"role": "system", "content": get_system_prompt()}]
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

# --- Color Constants for Terminal ---
C_LLM = '\033[96m'   # Cyan
C_TOOL = '\033[93m'  # Yellow
C_ERR = '\033[91m'   # Red
C_END = '\033[0m'    # Reset

async def execute_background_json_tools(history_snapshot: list):
    """Uses official, robust JSON tool calling in the background to execute actions."""
    start_time = time.time()
    print(f"  {C_TOOL}[Background Tools]{C_END} 🔍 Checking for required actions...")
    
    try:
        tool_response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=history_snapshot,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1, 
        )
        
        message = tool_response.choices[0].message
        
        if message.tool_calls:
            successful_actions = []
            
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    raw_args = tool_call.function.arguments
                    arguments = json.loads(raw_args) if raw_args and raw_args.strip() else {}
                    
                    print(f"  {C_TOOL}[Executing]{C_END} ⚙️ {tool_name}({arguments})")
                    
                    function_to_call = TOOL_MAP[tool_name]
                    sig = inspect.signature(function_to_call)
                    
                    # Execute safely
                    if not sig.parameters:
                        function_response = function_to_call()
                    else:
                        function_response = function_to_call(**arguments)
                        
                    successful_actions.append(f"{tool_name} returned: {function_response}")
                    print(f"  {C_TOOL}[Success]{C_END} ✅ Action completed in {time.time() - start_time:.2f}s")
                    
                except Exception as e:
                    print(f"  {C_ERR}[Tool Error]{C_END} Failed to execute {tool_name}: {e}")

            if successful_actions:
                memory_update = f"System Update: The following background actions occurred: {', '.join(successful_actions)}"
                conversation_history.append({"role": "system", "content": memory_update})
        else:
            print(f"  {C_TOOL}[Background Tools]{C_END} 💤 No actions needed (Took {time.time() - start_time:.2f}s)")
                
    except Exception as e:
        print(f"  {C_ERR}[Background Task Error]{C_END} {e}")


async def generate_agent_response_stream(transcript: str):
    """Instantly streams conversational audio while robustly extracting tools in parallel."""
    conversation_history.append({"role": "user", "content": transcript})
    
    print(f"\n{C_LLM}[LLM]{C_END} 🧠 Requesting response from Groq...")
    stream_start = time.time()
    first_token_received = False

    asyncio.create_task(execute_background_json_tools(list(conversation_history)))

    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=conversation_history,
        temperature=0.3,
        max_tokens=150,
        stream=True,
    )

    buffer = ""
    full_response = ""

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
    conversation_history.append({"role": "assistant", "content": full_response})