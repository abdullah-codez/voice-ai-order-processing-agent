"""Order state management and strict validation logic."""

from dataclasses import dataclass, field
from typing import Any
from src.restaurant.menu import find_item_by_name

@dataclass
class OrderItem:
    item_id: str
    name: str
    price: float
    quantity: int = 1
    special_instructions: str = ""

@dataclass
class Order:
    items: list[OrderItem] = field(default_factory=list)
    customer_name: str = ""
    is_confirmed: bool = False

    @classmethod
    def from_state(cls, state_items: list) -> "Order":
        """Reconstructs the Order object dynamically from the LangGraph session state."""
        order = cls()
        for item in state_items:
            order.items.append(
                OrderItem(
                    item_id=item.get("name"), # Using name as fallback ID
                    name=item.get("name"),
                    price=item.get("price", 0.0),
                    quantity=item.get("quantity", 1),
                    special_instructions=item.get("notes", "")
                )
            )
        return order

    def add_item(self, item_name: str, quantity: Any = 1, special_instructions: str = "") -> dict[str, Any]:
        # STRICT VALIDATION: Prevents LLM hallucinations (e.g. quantity="two") from crashing the cart
        try:
            quantity = int(quantity)
            if quantity <= 0:
                raise ValueError
        except (ValueError, TypeError):
            return {
                "success": False,
                "message": f"Invalid quantity '{quantity}'. Must be a positive integer."
            }

        matched = find_item_by_name(item_name)
        if not matched:
            return {"success": False, "message": f"'{item_name}' was not found on the menu."}

        if not matched.get("available", True):
            return {"success": False, "message": f"Sorry, {matched['name']} is currently out of stock."}

        # Update quantity if identical item exists
        for existing in self.items:
            if existing.name == matched["name"] and existing.special_instructions == special_instructions:
                existing.quantity += quantity
                return {"success": True, "message": f"Updated {matched['name']} quantity."}

        # Add new item
        self.items.append(
            OrderItem(
                item_id=matched["id"],
                name=matched["name"],
                price=matched["price"],
                quantity=quantity,
                special_instructions=special_instructions,
            )
        )
        return {"success": True, "message": f"Added {quantity}x {matched['name']}."}

    def remove_item(self, item_name: str) -> dict[str, Any]:
        matched = find_item_by_name(item_name)
        if not matched:
            return {"success": False, "message": f"'{item_name}' was not found."}

        initial_len = len(self.items)
        self.items = [i for i in self.items if i.name != matched["name"]]

        if len(self.items) < initial_len:
            return {"success": True, "message": f"Removed {matched['name']}."}
        return {"success": False, "message": f"{matched['name']} is not in your order."}

    def calculate_total(self) -> float:
        return sum(item.price * item.quantity for item in self.items)

    def get_summary(self) -> dict[str, Any]:
        item_list = [
            {
                "name": item.name,
                "quantity": item.quantity,
                "price": item.price,
                "subtotal": round(item.price * item.quantity, 2),
                "notes": item.special_instructions,
            }
            for item in self.items
        ]
        return {
            "items": item_list,
            "total_items_count": sum(i.quantity for i in self.items),
            "total_amount": round(self.calculate_total(), 2),
            "is_confirmed": self.is_confirmed,
        }