"""Automated multi-turn scenario simulator and evaluation harness over test_scenarios.json."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .agent import RestaurantDialogueAgent
from .database import DEFAULT_DATA_DIR, VALID_AREAS, VALID_FOODS, VALID_PRICES, RestaurantDatabase


class RuleBasedMockLLM:
    """Deterministic state-tracking policy emulating the 6-Phase LLM JSON output for offline testing."""

    def __init__(self, db: RestaurantDatabase) -> None:
        self.db = db
        self.last_restaurant_id: Optional[str] = None
        self.last_restaurant_name: Optional[str] = None
        self.pending_booking: Dict[str, Any] = {}
        self.last_order_id: Optional[str] = None

    def __call__(self, messages: List[Dict[str, str]]) -> Tuple[str, int]:
        last_msg = messages[-1]["content"]

        # Check if the previous message is a tool_result injection
        if last_msg.startswith("{") and '"tool_result"' in last_msg:
            payload = json.loads(last_msg)
            res = payload["tool_result"]

            if "results" in res:
                results = res.get("results", [])
                if not results:
                    return (
                        json.dumps(
                            {
                                "action": "final",
                                "response": "No matching restaurants found. Please relax your area, food, or pricerange constraints.",
                            }
                        ),
                        5,
                    )
                r0 = results[0]
                self.last_restaurant_id = r0["id"]
                self.last_restaurant_name = r0["name"]
                return (
                    json.dumps(
                        {
                            "action": "final",
                            "response": (
                                f"We found {r0['name']} ({r0['id']}) in the {r0['area']} area "
                                f"serving {r0.get('cuisine') or r0.get('food')} food ({r0['pricerange']}). "
                                "Would you like to book this restaurant?"
                            ),
                        }
                    ),
                    5,
                )

            if "menu" in res:
                items_str = ", ".join(
                    f"{i['name']} ({i['item_id']}, £{i['price']:.2f})" for i in res["menu"]
                )
                return (
                    json.dumps(
                        {
                            "action": "final",
                            "response": f"Menu: {items_str}. Would you like to pre-order any dishes?",
                        }
                    ),
                    5,
                )

            if res.get("error") == "Order not found.":
                return (
                    json.dumps(
                        {
                            "action": "final",
                            "response": "Sorry, that order number does not exist in our records.",
                        }
                    ),
                    5,
                )

            if res.get("success") and "order_id" in res:
                oid = res["order_id"]
                self.last_order_id = oid
                det = res["details"]
                return (
                    json.dumps(
                        {
                            "action": "final",
                            "response": (
                                f"Your booking is confirmed! Order ID: {oid} at {det['restaurant_name']} "
                                f"on {det['day']} at {det['time']} for {det['people']} people under {det['customer_name']} "
                                f"(Total: £{det['total_price']:.2f}). Is there anything else I can help you with, "
                                "or would you like to end the session?"
                            ),
                        }
                    ),
                    5,
                )

            if res.get("success") and res.get("message") == "Order updated.":
                det = res["details"]
                return (
                    json.dumps(
                        {
                            "action": "final",
                            "response": (
                                f"Order {det['order_id']} has been updated with dishes {det['dishes']} "
                                f"(Total: £{det['total_price']:.2f}). Is there anything else I can help you with, "
                                "or would you like to end the session?"
                            ),
                        }
                    ),
                    5,
                )

            if res.get("success") and "canceled" in str(res.get("message", "")):
                det = res["details"]
                return (
                    json.dumps(
                        {
                            "action": "final",
                            "response": f"Your order {det['order_id']} has been canceled.",
                        }
                    ),
                    5,
                )

            if res.get("success") and "order" in res:
                ord_obj = res["order"]
                return (
                    json.dumps(
                        {
                            "action": "final",
                            "response": (
                                f"Order {ord_obj['order_id']} details: {ord_obj['restaurant_name']} "
                                f"for {ord_obj['customer_name']}, dishes={ord_obj['dishes']}, "
                                f"total=£{ord_obj['total_price']:.2f}, status={ord_obj['status']}."
                            ),
                        }
                    ),
                    5,
                )

        text_lower = last_msg.lower()

        # Check order by explicit ID or last order ID
        ord_match = re.search(r"ord-\d{6}", text_lower)
        if "check" in text_lower and ("order" in text_lower or ord_match):
            oid = ord_match.group(0).upper() if ord_match else (self.last_order_id or "ORD-000000")
            return json.dumps({"action": "check_order", "args": {"order_id": oid}}), 5

        if "cancel" in text_lower and ("order" in text_lower or ord_match):
            oid = ord_match.group(0).upper() if ord_match else (self.last_order_id or "ORD-000000")
            return json.dumps({"action": "cancel_order", "args": {"order_id": oid}}), 5

        if "update" in text_lower and "order" in text_lower:
            oid = ord_match.group(0).upper() if ord_match else (self.last_order_id or "ORD-000000")
            dishes = self._parse_dishes(last_msg)
            return (
                json.dumps({"action": "update_order", "args": {"order_id": oid, "dishes": dishes}}),
                5,
            )

        # Search restaurants trigger
        found_area = next((a for a in VALID_AREAS if a in text_lower), None)
        found_food = next((f for f in VALID_FOODS if f in text_lower), None)
        found_price = next((p for p in VALID_PRICES if p in text_lower), None)
        if found_area or found_food or found_price:
            args = {}
            if found_area:
                args["area"] = found_area
            if found_food:
                args["food"] = found_food
            if found_price:
                args["pricerange"] = found_price
            return json.dumps({"action": "search_restaurants", "args": args}), 5

        if "recommend" in text_lower and self.last_restaurant_name:
            return (
                json.dumps(
                    {
                        "action": "final",
                        "response": f"I recommend {self.last_restaurant_name} ({self.last_restaurant_id}). Would you like to book it?",
                    }
                ),
                5,
            )

        # Booking slot extraction
        if "book" in text_lower and "place the booking" not in text_lower:
            day_m = re.search(
                r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text_lower
            )
            time_m = re.search(r"\b(\d{1,2}:\d{2})\b", text_lower)
            ppl_m = re.search(r"for\s+(\d+)\s+(?:people|person)", text_lower)
            name_m = re.search(r"name\s+([A-Za-z]+)", last_msg)
            self.pending_booking = {
                "restaurant_id": self.last_restaurant_id or "r001",
                "day": day_m.group(1).capitalize() if day_m else "Monday",
                "time": time_m.group(1) if time_m else "19:00",
                "people": int(ppl_m.group(1)) if ppl_m else 2,
                "customer_name": name_m.group(1) if name_m else "Guest",
                "dishes": {},
            }
            return (
                json.dumps(
                    {"action": "get_menu", "args": {"restaurant_id": self.pending_booking["restaurant_id"]}}
                ),
                5,
            )

        if "preorder" in text_lower or "pre-order" in text_lower:
            dishes = self._parse_dishes(last_msg)
            self.pending_booking["dishes"] = dishes
            b = self.pending_booking
            return (
                json.dumps(
                    {
                        "action": "final",
                        "response": (
                            f"Please confirm: {self.last_restaurant_name} on {b.get('day')} at {b.get('time')} "
                            f"for {b.get('people')} people, under {b.get('customer_name')}. "
                            f"Pre-order: {dishes}. Is this correct?"
                        ),
                    }
                ),
                5,
            )

        if "place the booking" in text_lower or ("yes" in text_lower and "end" not in text_lower):
            if self.pending_booking:
                return json.dumps({"action": "book_restaurant", "args": self.pending_booking}), 5

        if "end the conversation" in text_lower or "that would be all" in text_lower:
            return (
                json.dumps(
                    {
                        "action": "final",
                        "response": "Thank you for using our restaurant booking service! Goodbye and enjoy your meal.",
                    }
                ),
                5,
            )

        # Off-topic fallback (Phase 0)
        return (
            json.dumps(
                {
                    "action": "final",
                    "response": (
                        "I am a restaurant booking and ordering assistant. You can search by "
                        "area (centre, north, south, east, west, riverside), "
                        "cuisine (chinese, italian, indian, japanese, vegetarian, british), or "
                        "price range (cheap, moderate, expensive)."
                    ),
                }
            ),
            5,
        )

    @staticmethod
    def _parse_dishes(text: str) -> Dict[str, int]:
        dishes: Dict[str, int] = {}
        pattern = re.compile(r"(\d+)x\s+[^(]+\(([a-z]\d{2})\)", re.IGNORECASE)
        for qty_str, code in pattern.findall(text):
            dishes[code.lower()] = int(qty_str)
        return dishes


class ScenarioEvaluator:
    """Multi-turn scenario runner and check verifier over test_scenarios.json."""

    def __init__(
        self,
        db: Optional[RestaurantDatabase] = None,
        scenarios_path: Optional[os.PathLike | str] = None,
    ) -> None:
        self.db = db or RestaurantDatabase()
        self.scenarios_path = Path(scenarios_path or (DEFAULT_DATA_DIR / "test_scenarios.json"))
        with open(self.scenarios_path, "r", encoding="utf-8") as f:
            self.scenarios: List[Dict[str, Any]] = json.load(f)

    def evaluate_scenario(
        self,
        scenario: Dict[str, Any],
        llm_fn: Optional[Callable[[List[Dict[str, str]]], Tuple[str, int]]] = None,
    ) -> Dict[str, Any]:
        """Run all turns in a scenario and verify all scenario checks."""
        self.db.clear_orders()
        policy = llm_fn or RuleBasedMockLLM(self.db)
        agent = RestaurantDialogueAgent(db=self.db, llm_fn=policy)

        turn_logs: List[Dict[str, Any]] = []
        all_terminated = True

        for user_utterance in scenario["turns"]:
            reply, trace, terminated = agent.step(user_utterance)
            if not terminated:
                all_terminated = False
            turn_logs.append(
                {
                    "user": user_utterance,
                    "reply": reply,
                    "trace": trace,
                    "terminated": terminated,
                }
            )

        orders_after = self.db.load_orders()
        checks_passed: Dict[str, bool] = {}
        all_replies_lower = " ".join(t["reply"].lower() for t in turn_logs)
        all_tools_called = [
            ev["tool"] for t in turn_logs for ev in t["trace"] if ev.get("tool")
        ]

        for chk in scenario.get("checks", []):
            if chk == "has_order_after_end":
                checks_passed[chk] = len(orders_after) > 0
            elif chk == "final_terminated":
                checks_passed[chk] = all_terminated
            elif chk == "offtopic_lists_ranges":
                has_areas = "centre" in all_replies_lower and "riverside" in all_replies_lower
                has_foods = "chinese" in all_replies_lower and "italian" in all_replies_lower
                has_prices = "cheap" in all_replies_lower and "expensive" in all_replies_lower
                checks_passed[chk] = bool(has_areas and has_foods and has_prices)
            elif chk == "query_found":
                checks_passed[chk] = "check_order" in all_tools_called
            elif chk == "update_applied":
                checks_passed[chk] = "update_order" in all_tools_called and any(
                    o.get("dishes", {}).get("j04") == 2 and o.get("dishes", {}).get("j05") == 1
                    for o in orders_after.values()
                )
            elif chk == "canceled_status":
                checks_passed[chk] = "cancel_order" in all_tools_called and any(
                    o.get("status") == "canceled" for o in orders_after.values()
                )
            elif chk == "not_found_message":
                checks_passed[chk] = (
                    "not exist" in all_replies_lower or "no order" in all_replies_lower
                )
            else:
                checks_passed[chk] = True

        passed = all(checks_passed.values())
        return {
            "scenario_id": scenario["scenario_id"],
            "description": scenario.get("description", ""),
            "passed": passed,
            "checks": checks_passed,
            "num_turns": len(turn_logs),
            "turns": turn_logs,
        }

    def evaluate_all(
        self,
        llm_fn: Optional[Callable[[List[Dict[str, str]]], Tuple[str, int]]] = None,
    ) -> Dict[str, Any]:
        """Run all test scenarios and return aggregate metrics."""
        results = [self.evaluate_scenario(sc, llm_fn=llm_fn) for sc in self.scenarios]
        num_passed = sum(1 for r in results if r["passed"])
        total = max(len(results), 1)
        return {
            "total_scenarios": len(results),
            "passed_scenarios": num_passed,
            "pass_rate": round((num_passed / total) * 100.0, 2),
            "scenarios": results,
        }
