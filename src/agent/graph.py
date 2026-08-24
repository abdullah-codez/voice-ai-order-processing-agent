"""LangGraph State Machine for Abdullah's Cafe Voice Agent."""

import json
import os
from groq import AsyncGroq
from langgraph.graph import StateGraph, START, END
from src.agent.state import AgentState
from src.agent.tools import TOOLS, TOOL_MAP

client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_NAME = "openai/gpt-oss-20b"

# --- Helper Function for Node Sub-Agents ---
async def extract_with_llm(state: AgentState, phase_tools: list, system_instruction: str):
    """Fires a focused Groq request to extract specific state data in the background."""
    messages = [{"role": "system", "content": system_instruction}] + state.get("conversation_history", [])
    
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=phase_tools,
        tool_choice="auto",
        temperature=0.1,
    )
    return response.choices[0].message

# --- Node Definitions ---

async def greeting_node(state: AgentState) -> dict:
    """Handles the initial greeting and instantly transitions to order collection."""
    print("  [Graph] ➡️ Entering Greeting Phase")
    return {"current_phase": "order"}

async def collect_order_node(state: AgentState) -> dict:
    """Extracts food items and updates the cart."""
    print("  [Graph] 🍔 Processing Order Items")
    
    instruction = "You are updating the cart. Use tools to add/remove items. If the user explicitly says they are done ordering, or says 'that's it', use the 'advance_phase' tool to move to payment."
    
    # Define the missing tool schema
    advance_tool = {
        "type": "function",
        "function": {
            "name": "advance_phase",
            "description": "Call this tool to move to the next phase when the customer is done ordering food.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
    
    # Combine our standard tools with the new phase transition tool
    phase_tools = TOOLS + [advance_tool]
    
    # Pass the updated tool list to the LLM
    message = await extract_with_llm(state, phase_tools, instruction)
    
    if message.tool_calls:
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            
            # Catch the transition tool!
            if tool_name == "advance_phase":
                print("  [Graph] ⏭️ Customer finished ordering. Moving to Payment.")
                return {"current_phase": "payment"}
            
            try:
                args = json.loads(tool_call.function.arguments)
                print(f"  [Graph Executing]: {tool_name}({args})")
                TOOL_MAP[tool_name](**args)
            except Exception as e:
                print(f"  [Graph Tool Error] {e}")

    return {}

async def payment_node(state: AgentState) -> dict:
    """Extracts 'Cash' or 'Card'."""
    print("  [Graph] 💳 Processing Payment Method")
    instruction = "Extract the payment method. If the user says Cash or Card, use the 'set_payment' tool."
    
    tools = [{
        "type": "function",
        "function": {
            "name": "set_payment",
            "description": "Call this to save the payment method.",
            "parameters": {
                "type": "object",
                "properties": {"method": {"type": "string", "enum": ["Cash", "Card", "Cash on Delivery"]}},
                "required": ["method"]
            }
        }
    }]
    
    message = await extract_with_llm(state, tools, instruction)
    if message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call.function.name == "set_payment":
                args = json.loads(tool_call.function.arguments)
                print(f"  [Graph] ⏭️ Payment set to {args.get('method')}. Moving to Fulfillment.")
                return {"payment_method": args.get("method"), "current_phase": "fulfillment"}
                
    return {} # Stay in payment phase if nothing was extracted

async def fulfillment_node(state: AgentState) -> dict:
    """Extracts Delivery Address."""
    print("  [Graph] 🚗 Processing Fulfillment")
    instruction = "Extract the delivery address from the user. Use the 'set_address' tool."
    
    tools = [{
        "type": "function",
        "function": {
            "name": "set_address",
            "description": "Call this to save the delivery address.",
            "parameters": {
                "type": "object",
                "properties": {"address": {"type": "string"}},
                "required": ["address"]
            }
        }
    }]
    
    message = await extract_with_llm(state, tools, instruction)
    if message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call.function.name == "set_address":
                args = json.loads(tool_call.function.arguments)
                print(f"  [Graph] ⏭️ Address set. Moving to Info.")
                return {"delivery_address": args.get("address"), "current_phase": "info"}
                
    return {}

async def customer_info_node(state: AgentState) -> dict:
    """Extracts Name and Phone Number."""
    print("  [Graph] 👤 Processing Customer Info")
    instruction = "Extract the customer's phone number. Use the 'set_info' tool."
    
    tools = [{
        "type": "function",
        "function": {
            "name": "set_info",
            "description": "Call this to save customer contact info.",
            "parameters": {
                "type": "object",
                "properties": {"phone": {"type": "string"}},
                "required": ["phone"]
            }
        }
    }]
    
    message = await extract_with_llm(state, tools, instruction)
    if message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call.function.name == "set_info":
                args = json.loads(tool_call.function.arguments)
                print(f"  [Graph] ⏭️ Info set. Moving to Finalize.")
                return {"customer_phone": args.get("phone"), "current_phase": "finalize"}
                
    return {}

async def finalize_order_node(state: AgentState) -> dict:
    """Commits to PostgreSQL and ends the conversation."""
    print("  [Graph] ✅ Finalizing Order")
    return {"current_phase": "complete"}

# --- The Router & Graph Construction ---

def route_based_on_phase(state: AgentState) -> str:
    phase = state.get("current_phase", "greeting")
    return phase if phase in ["greeting", "order", "payment", "fulfillment", "info", "finalize"] else END

def build_order_graph():
    builder = StateGraph(AgentState) 

    builder.add_node("greeting", greeting_node)
    builder.add_node("order", collect_order_node)
    builder.add_node("payment", payment_node)
    builder.add_node("fulfillment", fulfillment_node)
    builder.add_node("info", customer_info_node)
    builder.add_node("finalize", finalize_order_node)

    builder.add_conditional_edges(START, route_based_on_phase)

    for node in ["greeting", "order", "payment", "fulfillment", "info", "finalize"]:
        builder.add_edge(node, END)

    return builder.compile()

graph_app = build_order_graph()