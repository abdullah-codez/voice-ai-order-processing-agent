"""Tool schemas mapping LLM intents to the Restaurant Order execution layer."""

from src.restaurant.order import Order

# Define the tools schema for Groq / Llama 3
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

# --- Stateless Wrapper Execution Functions ---
# Notice how they now accept 'state' as the first argument to isolate memory per caller.

def add_item(state: dict, item_name: str, quantity: int = 1, special_instructions: str = "") -> dict:
    order = Order.from_state(state.get("order_items", []))
    result = order.add_item(item_name, quantity, special_instructions)
    
    # Sync the cart and total directly back into LangGraph's session state
    summary = order.get_summary()
    state["order_items"] = summary["items"]
    state["total_amount"] = summary["total_amount"]
    return result

def remove_item(state: dict, item_name: str) -> dict:
    order = Order.from_state(state.get("order_items", []))
    result = order.remove_item(item_name)
    
    summary = order.get_summary()
    state["order_items"] = summary["items"]
    state["total_amount"] = summary["total_amount"]
    return result

def get_order_summary(state: dict) -> dict:
    order = Order.from_state(state.get("order_items", []))
    return order.get_summary()

# Router dictionary for llm.py
TOOL_MAP = {
    "add_item": add_item,
    "remove_item": remove_item,
    "get_order_summary": get_order_summary
}