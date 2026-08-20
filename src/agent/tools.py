"""Tool schemas mapping LLM intents to the Restaurant Order execution layer."""

import json
from src.restaurant.order import Order

# We instantiate a global order state for the current session MVP
current_order = Order()

# Define the tools exactly as Groq/Llama 3 requires
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_item",
            "description": "Add a specific quantity of a menu item to the customer's order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "The name of the menu item (e.g. 'Classic Beef Burger', 'Coke')"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "How many of this item to add"
                    },
                    "special_instructions": {
                        "type": "string",
                        "description": "Any modifications like 'no pickles' or 'extra cheese'"
                    }
                },
                "required": ["item_name", "quantity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_item",
            "description": "Remove an item entirely from the customer's order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "The name of the menu item to remove"
                    }
                },
                "required": ["item_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_summary",
            "description": "Retrieve the current items in the cart and the total price to read back to the customer.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

def execute_tool(tool_name: str, arguments: dict) -> str:
    """Routes the LLM tool call to the correct Order method and returns the JSON string response."""
    
    if tool_name == "add_item":
        result = current_order.add_item(
            item_name=arguments.get("item_name"),
            quantity=arguments.get("quantity", 1),
            special_instructions=arguments.get("special_instructions", "")
        )
        return json.dumps(result)
        
    elif tool_name == "remove_item":
        result = current_order.remove_item(item_name=arguments.get("item_name"))
        return json.dumps(result)
        
    elif tool_name == "get_order_summary":
        result = current_order.get_summary()
        return json.dumps(result)
        
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})