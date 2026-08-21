"""Language Model and AI processing logic using Groq API."""

import json
import os
import re
from dotenv import load_dotenv
from groq import AsyncGroq
from src.agent.prompts import get_system_prompt
from src.agent.tools import TOOL_MAP, TOOLS

load_dotenv()

client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_NAME = "openai/gpt-oss-20b"

conversation_history = [{"role": "system", "content": get_system_prompt()}]

# Matches sentence terminators (.!?) that are followed by whitespace, ignoring decimal numbers (e.g., $8.99)
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


async def generate_agent_response_stream(transcript: str):
  """Generates the response and yields complete sentences without breaking decimals."""
  conversation_history.append({"role": "user", "content": transcript})

  # Step 1: Decision / Tool Execution Phase
  tool_response = await client.chat.completions.create(
      model=MODEL_NAME,
      messages=conversation_history,
      tools=TOOLS,
      tool_choice="auto",
      temperature=0.3,
      max_tokens=150,
  )

  response_message = tool_response.choices[0].message

  if response_message.tool_calls:
    conversation_history.append(response_message)

    for tool_call in response_message.tool_calls:
      tool_name = tool_call.function.name
      try:
        arguments = json.loads(tool_call.function.arguments)
        print(f"  [System] Executing Tool: {tool_name}({arguments})")
        function_to_call = TOOL_MAP[tool_name]
        function_response = function_to_call(**arguments)
      except Exception as e:
        print(f"\n[LLM Error] Error code: 400 - {e}")
        function_response = {"error": str(e)}

      conversation_history.append({
          "tool_call_id": tool_call.id,
          "role": "tool",
          "name": tool_name,
          "content": json.dumps(function_response),
      })

  # Step 2: Conversational Streaming Phase
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
      buffer += token
      full_response += token

      # Check if buffer contains at least one complete sentence
      parts = SENTENCE_SPLIT_REGEX.split(buffer)
      if len(parts) > 1:
        # Yield all completed sentences except the last partial chunk
        for sentence in parts[:-1]:
          if sentence.strip():
            yield sentence.strip()
        buffer = parts[-1]

  # Yield remaining tail
  if buffer.strip():
    yield buffer.strip()

  conversation_history.append({"role": "assistant", "content": full_response})