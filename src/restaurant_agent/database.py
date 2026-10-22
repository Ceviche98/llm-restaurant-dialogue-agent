"""Transactional data access layer over restaurant, menu, and order JSON databases."""

from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

VALID_AREAS = {"centre", "north", "south", "east", "west", "riverside"}
VALID_FOODS = {"chinese", "italian", "indian", "japanese", "vegetarian", "british"}
VALID_PRICES = {"cheap", "moderate", "expensive"}

AREA_MAP = {
    "centre": "centre",
    "center": "centre",
    "city centre": "centre",
    "city center": "centre",
    "downtown": "centre",
    "central": "centre",
    "north": "north",
    "south": "south",
    "east": "east",
    "west": "west",
    "riverside": "riverside",
    "river side": "riverside",
}

FOOD_MAP = {
    "chinese": "chinese",
    "italian": "italian",
    "indian": "indian",
    "japanese": "japanese",
    "vegetarian": "vegetarian",
    "veggie": "vegetarian",
    "vegan": "vegetarian",
    "british": "british",
    "english": "british",
}

PRICE_MAP = {
    "cheap": "cheap",
    "inexpensive": "cheap",
    "budget": "cheap",
    "low": "cheap",
    "low-priced": "cheap",
    "moderate": "moderate",
    "mid": "moderate",
    "medium": "moderate",
    "moderately priced": "moderate",
    "expensive": "expensive",
    "pricey": "expensive",
    "high": "expensive",
    "high-end": "expensive",
}


def normalize_slot(value: Optional[str], mapping: Dict[str, str]) -> Optional[str]:
    """Normalize a slot string against a canonical synonym dictionary."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    s = re.sub(r"\s+", " ", s)
    if s in mapping:
        return mapping[s]
    for k, v in mapping.items():
        if k in s:
            return v
    return None


class RestaurantDatabase:
    """Transactional data access layer over restaurants.json, menus.json, and orders.json."""

    def __init__(
        self,
        restaurants_path: Optional[os.PathLike | str] = None,
        menus_path: Optional[os.PathLike | str] = None,
        orders_path: Optional[os.PathLike | str] = None,
    ) -> None:
        self.restaurants_path = Path(restaurants_path or (DEFAULT_DATA_DIR / "restaurants.json"))
        self.menus_path = Path(menus_path or (DEFAULT_DATA_DIR / "menus.json"))
        self.orders_path = Path(orders_path or (DEFAULT_DATA_DIR / "orders.json"))

        self.restaurants: List[Dict[str, Any]] = self._load_json(self.restaurants_path, default=[])
        self.menus: Dict[str, List[Dict[str, Any]]] = self._load_json(self.menus_path, default={})

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default

    def load_orders(self) -> Dict[str, Dict[str, Any]]:
        """Read all persisted orders from orders.json."""
        return self._load_json(self.orders_path, default={})

    def save_orders(self, orders: Dict[str, Dict[str, Any]]) -> None:
        """Persist orders dictionary atomically to orders.json."""
        self.orders_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.orders_path, "w", encoding="utf-8") as f:
            json.dump(orders, f, indent=4)

    def clear_orders(self) -> None:
        """Reset orders store to an empty dictionary."""
        self.save_orders({})

    def get_restaurant(self, restaurant_id: str) -> Optional[Dict[str, Any]]:
        """Return a restaurant record by its ID (e.g., 'r001')."""
        rid = str(restaurant_id).strip()
        return next((r for r in self.restaurants if r["id"] == rid), None)

    def calculate_total(self, restaurant_id: str, dishes: Dict[str, int]) -> float:
        """Compute total pre-order cost for a dictionary of {item_id: quantity}."""
        menu_items = self.menus.get(str(restaurant_id).strip(), [])
        price_lookup = {item["item_id"]: float(item["price"]) for item in menu_items}
        total = sum(price_lookup.get(iid, 0.0) * int(qty) for iid, qty in (dishes or {}).items())
        return round(total, 2)

    def search_restaurants(
        self,
        area: Optional[str] = None,
        food: Optional[str] = None,
        pricerange: Optional[str] = None,
        dietary: Optional[str] = None,
        name: Optional[str] = None,
        limit: int = 5,
        **_: Any,
    ) -> Dict[str, Any]:
        """Search restaurants by area, cuisine/food, pricerange, dietary constraint, or name."""
        a = normalize_slot(area, AREA_MAP)
        f = normalize_slot(food or dietary, FOOD_MAP)
        p = normalize_slot(pricerange, PRICE_MAP)
        name_q = str(name).strip().lower() if name and str(name).strip() else None

        results: List[Dict[str, Any]] = []
        for r in self.restaurants:
            r_cuisine = r.get("cuisine") or r.get("food")
            if a and r.get("area") != a:
                continue
            if f and r_cuisine != f:
                continue
            if p and r.get("pricerange") != p:
                continue
            if name_q and name_q not in str(r.get("name", "")).lower():
                continue
            results.append(r)

        capped_limit = max(1, min(int(limit), 10))
        return {
            "count": len(results),
            "results": results[:capped_limit],
        }

    def get_menu(self, restaurant_id: str, **_: Any) -> Dict[str, Any]:
        """Return the full menu for a restaurant."""
        rid = str(restaurant_id).strip()
        items = self.menus.get(rid)
        if items is None:
            return {"error": f"No menu found for restaurant_id '{restaurant_id}'."}
        return {"restaurant_id": rid, "menu": items}

    def book_restaurant(
        self,
        restaurant_id: str,
        day: str,
        time: str,
        people: int,
        customer_name: str,
        dishes: Optional[Dict[str, int]] = None,
        order_id: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Create a confirmed reservation and optional dish pre-order."""
        restaurant = self.get_restaurant(restaurant_id)
        if not restaurant:
            return {"error": f"Unknown restaurant_id: {restaurant_id}"}

        try:
            people_int = int(people)
        except (TypeError, ValueError):
            return {"error": "'people' must be a positive integer."}
        if people_int < 1 or people_int > 20:
            return {"error": "'people' must be between 1 and 20."}

        day_s = str(day or "").strip()
        time_s = str(time or "").strip()
        name_s = str(customer_name or "").strip()
        if not day_s or not time_s or not name_s:
            return {"error": "day, time, and customer_name are all required."}

        clean_dishes: Dict[str, int] = {}
        if dishes:
            valid_item_ids = {item["item_id"] for item in self.menus.get(restaurant["id"], [])}
            for iid, qty in dishes.items():
                if iid not in valid_item_ids:
                    return {"error": f"Unknown menu item_id '{iid}' for restaurant '{restaurant['id']}'."}
                qty_int = int(qty)
                if qty_int > 0:
                    clean_dishes[iid] = qty_int

        generated_id = order_id or f"ORD-{random.randint(100000, 999999)}"
        order_data = {
            "order_id": generated_id,
            "restaurant_id": restaurant["id"],
            "restaurant_name": restaurant["name"],
            "address": restaurant["address"],
            "day": day_s,
            "time": time_s,
            "people": people_int,
            "customer_name": name_s,
            "dishes": clean_dishes,
            "total_price": self.calculate_total(restaurant["id"], clean_dishes),
            "status": "confirmed",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

        orders = self.load_orders()
        orders[generated_id] = order_data
        self.save_orders(orders)
        return {"success": True, "order_id": generated_id, "details": order_data}

    def update_order(self, order_id: str, dishes: Dict[str, int], **_: Any) -> Dict[str, Any]:
        """Update the pre-ordered dishes on an existing active order."""
        oid = str(order_id or "").strip()
        orders = self.load_orders()
        if oid not in orders:
            return {"error": "Order not found."}
        if orders[oid].get("status") == "canceled":
            return {"error": "Cannot update a canceled order."}

        rid = orders[oid]["restaurant_id"]
        valid_item_ids = {item["item_id"] for item in self.menus.get(rid, [])}
        clean_dishes: Dict[str, int] = {}
        for iid, qty in (dishes or {}).items():
            if iid not in valid_item_ids:
                return {"error": f"Unknown menu item_id '{iid}' for restaurant '{rid}'."}
            qty_int = int(qty)
            if qty_int > 0:
                clean_dishes[iid] = qty_int

        orders[oid]["dishes"] = clean_dishes
        orders[oid]["total_price"] = self.calculate_total(rid, clean_dishes)
        self.save_orders(orders)
        return {"success": True, "message": "Order updated.", "details": orders[oid]}

    def check_order(self, order_id: str, **_: Any) -> Dict[str, Any]:
        """Retrieve an order by ID (works for both confirmed and canceled orders)."""
        oid = str(order_id or "").strip()
        orders = self.load_orders()
        if oid not in orders:
            return {"error": "Order not found."}
        return {"success": True, "order": orders[oid]}

    def cancel_order(self, order_id: str, **_: Any) -> Dict[str, Any]:
        """Non-destructively cancel an order so it remains auditable afterward."""
        oid = str(order_id or "").strip()
        orders = self.load_orders()
        if oid not in orders:
            return {"error": "Order not found."}
        orders[oid]["status"] = "canceled"
        self.save_orders(orders)
        return {
            "success": True,
            "message": f"Order {oid} canceled.",
            "details": orders[oid],
        }
