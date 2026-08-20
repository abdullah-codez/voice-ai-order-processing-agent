"""In-memory menu database and query utilities."""

MENU_ITEMS = {
    "classic_beef_burger": {
        "id": "classic_beef_burger",
        "name": "Classic Beef Burger",
        "category": "burgers",
        "price": 8.99,
        "description": "Juicy beef patty with lettuce, tomato, pickles, and house sauce.",
        "available": True,
        "customizations": ["no pickles", "extra cheese", "gluten-free bun"],
    },
    "crispy_chicken_burger": {
        "id": "crispy_chicken_burger",
        "name": "Crispy Chicken Burger",
        "category": "burgers",
        "price": 7.99,
        "description": "Crispy fried chicken breast with spicy mayo and slaw.",
        "available": True,
        "customizations": ["extra spicy", "sauce on the side"],
    },
    "margherita_pizza": {
        "id": "margherita_pizza",
        "name": "Margherita Pizza",
        "category": "pizza",
        "price": 12.50,
        "description": "Fresh mozzarella, tomato sauce, and basil on thin crust.",
        "available": True,
        "customizations": ["extra cheese", "thin crust"],
    },
    "french_fries": {
        "id": "french_fries",
        "name": "French Fries",
        "category": "sides",
        "price": 3.49,
        "description": "Crispy golden salted fries.",
        "available": True,
        "customizations": ["extra salt", "truffle seasoning"],
    },
    "coke": {
        "id": "coke",
        "name": "Coca-Cola",
        "category": "drinks",
        "price": 1.99,
        "description": "Can of chilled Coca-Cola.",
        "available": True,
        "customizations": ["diet", "zero", "regular"],
    },
    "lemonade": {
        "id": "lemonade",
        "name": "Fresh Lemonade",
        "category": "drinks",
        "price": 2.99,
        "description": "House-made fresh lemonade with mint.",
        "available": False,  # Useful for testing unavailable item handling
        "customizations": ["less sugar"],
    },
}


def get_menu_summary() -> str:
    """Returns a concise string representation of available menu items for prompts."""
    lines = []
    for item in MENU_ITEMS.values():
        status = "Available" if item["available"] else "Out of Stock"
        lines.append(
            f"- {item['name']} (${item['price']:.2f}) [{status}]: {item['description']}"
        )
    return "\n".join(lines)


def find_item_by_name(query: str) -> dict | None:
    """Fuzzy-matches or substring-matches a menu item."""
    query_clean = query.lower().strip()
    for item_id, item in MENU_ITEMS.items():
        if query_clean in item["name"].lower() or item_id in query_clean:
            return item
    return None