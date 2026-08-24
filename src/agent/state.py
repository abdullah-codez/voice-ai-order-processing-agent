"""LangGraph State Definition for the Order Pipeline."""

from typing import List, Dict, Optional
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """The central state machine for the voice agent."""
    conversation_history: List[Dict[str, str]]
    
    # --- The Conversational Checklist ---
    order_items: List[Dict[str, str]]  # e.g., [{"item": "Classic Beef Burger", "quantity": 1}]
    payment_method: Optional[str]      # "Cash" or "Card"
    fulfillment_type: Optional[str]    # "Pickup" or "Delivery"
    delivery_address: Optional[str]    
    customer_name: Optional[str]
    customer_phone: Optional[str]
    
    # --- Flow Control ---
    current_phase: str                 # "greeting", "order", "payment", "fulfillment", "info", "finalize"
    total_amount: float