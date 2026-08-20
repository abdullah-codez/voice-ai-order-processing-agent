"""Order state management and validation logic."""

from dataclasses import dataclass, field
from typing import Any
from src.restaurant.menu import find_item_by_name, MENU_ITEMS


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

    def add_item(self, item_name: str, quantity: int = 1, special_instructions: str = "") -> dict[str, Any]:
        matched = find_item_by_name(item_name)
        if not matched:
            return {
                "success": False,
                "message": f"'{item_name}' was not found on the menu.",
            }

        if not matched["available"]:
            return {
                "success": False,
                "message": f"Sorry, {matched['name']} is currently out of stock.",
            }

        # Check if item already exists with identical instructions
        for existing in self.items:
            if existing.item_id == matched["id"] and existing.special_instructions == special_instructions:
                existing.quantity += quantity
                return {
                    "success": True,
                    "message": f"Updated {matched['name']} quantity to {existing.quantity}.",
                    "current_order": self.get_summary(),
                }

        self.items.append(
            OrderItem(
                item_id=matched["id"],
                name=matched["name"],
                price=matched["price"],
                quantity=quantity,
                special_instructions=special_instructions,
            )
        )
        return {
            "success": True,
            "message": f"Added {quantity}x {matched['name']} to the order.",
            "current_order": self.get_summary(),
        }

    def remove_item(self, item_name: str) -> dict[str, Any]:
        matched = find_item_by_name(item_name)
        if not matched:
            return {"success": False, "message": f"'{item_name}' was not found in the menu."}

        initial_len = len(self.items)
        self.items = [i for i in self.items if i.item_id != matched["id"]]

        if len(self.items) < initial_len:
            return {
                "success": True,
                "message": f"Removed {matched['name']} from your order.",
                "current_order": self.get_summary(),
            }
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
        total = round(self.calculate_total(), 2)
        return {
            "items": item_list,
            "total_items_count": sum(i.quantity for i in self.items),
            "total_amount": total,
            "is_confirmed": self.is_confirmed,
        }

    def reset(self):
        self.items.clear()
        self.customer_name = ""
        self.is_confirmed = False