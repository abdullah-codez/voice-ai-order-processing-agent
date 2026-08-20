"""Orchestrates the Groq LLM tool-calling loop and manages conversation history."""

import os
import json
from dotenv import load_dotenv
from groq import Groq

from src.agent.prompts import get_system_prompt
from src.agent.tools import TOOLS, execute_tool

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Use the fast 8B model for real-time voice latency
MODEL_NAME = "llama-3.1-8b-instant"

conversation_history = [
    {"role": "system", "content": get_system_prompt()}
]

def generate_agent_response(user_text: str) -> str:
    """
    Sends the user's transcript to Groq, executes any tools requested,
    and returns the AI's final spoken response.
    """
    if not user_text.strip():
        return ""
        
    conversation_history.append({"role": "user", "content": user_text})
    
    try:
        # Step 1: Tool Decision Phase
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=conversation_history,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=150
        )
        
        response_message = response.choices[0].message
        
        # Step 2: Tool Execution Phase
        if response_message.tool_calls:
            conversation_history.append(response_message)
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"\n  [System] Executing Tool: {function_name}({function_args})")
                
                tool_result = execute_tool(function_name, function_args)
                
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result
                })
                
            # Step 3: Conversational Response Phase
            final_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation_history,
                temperature=0.3,
                max_tokens=150
            )
            final_text = final_response.choices[0].message.content
        else:
            final_text = response_message.content
            
        if final_text:
            conversation_history.append({"role": "assistant", "content": final_text})
            
        return final_text
        
    except Exception as e:
        print(f"\n[LLM Error] {e}")
        return "I'm sorry, my system encountered an error while processing that."