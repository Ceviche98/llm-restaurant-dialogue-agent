"""Structured tool definitions and deterministic execution handlers for LLM function calling."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Set

from .database import RestaurantDatabase


TOOL_ACTIONS: Set[str] = {
    "search_restaurants",
    "get_menu",
    "book_restaurant",
    "create_order",
    "update_order",
    "modify_order",
    "check_order",
    "get_order_status",
    "cancel_order",
}

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "search_restaurants",
        "description": "Search restaurants by area, food/cuisine, pricerange, or dietary constraints.",
        "parameters": {
            "type": "object",
            "properties": {
                "area": {
                    "type": "string",
                    "enum": ["centre", "north", "south", "east", "west", "riverside"],
                },
                "food": {
                    "type": "string",
                    "enum": ["chinese", "italian", "indian", "japanese", "vegetarian", "british"],
                },
                "pricerange": {
                    "type": "string",
                    "enum": ["cheap", "moderate", "expensive"],
                },
                "limit": {"type": "integer", "default": 5},
            },
        },
    },
    {
        "name": "get_menu",
        "description": "Retrieve the complete menu (item_id, name, price) for a given restaurant_id.",
        "parameters": {
            "type": "object",
            "properties": {"restaurant_id": {"type": "string"}},
            "required": ["restaurant_id"],
        },
    },
    {
        "name": "book_restaurant",
        "description": "Create a restaurant booking and optional dish pre-order, returning an ORD-XXXXXX order_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "restaurant_id": {"type": "string"},
                "day": {"type": "string"},
                "time": {"type": "string"},
                "people": {"type": "integer"},
                "customer_name": {"type": "string"},
                "dishes": {
                    "type": "object",
                    "description": "Mapping of item_id to quantity, e.g. {'c02': 3}",
                },
            },
            "required": ["restaurant_id", "day", "time", "people", "customer_name"],
        },
    },
    {
        "name": "update_order",
        "description": "Replace the pre-ordered dishes on an existing active order_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "dishes": {"type": "object"},
            },
            "required": ["order_id", "dishes"],
        },
    },
    {
        "name": "check_order",
        "description": "Check full order details by order_id (works for both confirmed and canceled orders).",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "cancel_order",
        "description": "Non-destructively cancel an order by order_id while preserving audit history.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
]


def extract_json_action(text: str) -> Dict[str, Any]:
    """Parse a single JSON action object from raw LLM output, stripping optional markdown fences."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    obj = json.loads(cleaned)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be object/dict, got {type(obj).__name__}")
    return obj


class ToolExecutor:
    """Deterministic tool router binding JSON tool calls to a RestaurantDatabase instance."""

    def __init__(self, db: Optional[RestaurantDatabase] = None) -> None:
        self.db = db or RestaurantDatabase()
        self.tool_fn: Dict[str, Callable[..., Dict[str, Any]]] = {
            "search_restaurants": self.db.search_restaurants,
            "get_menu": self.db.get_menu,
            "book_restaurant": self.db.book_restaurant,
            "create_order": self.db.book_restaurant,
            "update_order": self.db.update_order,
            "modify_order": self.db.update_order,
            "check_order": self.db.check_order,
            "get_order_status": self.db.check_order,
            "cancel_order": self.db.cancel_order,
        }

    def execute(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Dispatch a tool call by name with keyword arguments."""
        fn = self.tool_fn.get(tool_name)
        if fn is None:
            return {"ok": False, "error": f"Unknown tool: {tool_name}"}
        safe_args = args if isinstance(args, dict) else {}
        return fn(**safe_args)
