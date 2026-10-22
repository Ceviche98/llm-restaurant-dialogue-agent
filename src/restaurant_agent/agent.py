"""Stateful Conversational Restaurant Agent orchestrating multi-turn dialogue, tool calls, and error recovery."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from .database import RestaurantDatabase
from .tools import TOOL_ACTIONS, ToolExecutor, extract_json_action


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"

SYSTEM_PROMPT = """You are a friendly restaurant booking and order assistant.
You MUST output ONLY ONE valid JSON object per turn — no markdown, no extra text.

═══════════════════════════════════════════════
 ALLOWED ACTIONS (exact JSON shapes)
═══════════════════════════════════════════════
{"action": "search_restaurants", "args": {"area": "...", "food": "...", "pricerange": "..."}}
{"action": "get_menu",           "args": {"restaurant_id": "r001"}}
{"action": "book_restaurant",    "args": {"restaurant_id": "r001", "day": "Monday", "time": "19:00", "people": 4, "customer_name": "Oscar", "dishes": {"c02": 3}}}
{"action": "update_order",       "args": {"order_id": "ORD-123456", "dishes": {"c02": 2, "c04": 1}}}
{"action": "check_order",        "args": {"order_id": "ORD-123456"}}
{"action": "cancel_order",       "args": {"order_id": "ORD-123456"}}
{"action": "final",              "response": "...English message to the user..."}

NOTE: Order IDs always look like ORD-XXXXXX (6 digits, e.g. ORD-605454).

═══════════════════════════════════════════════
 SLOT ENUMERATIONS (use these exact values only)
═══════════════════════════════════════════════
area:       centre | north | south | east | west | riverside
food:       chinese | italian | indian | japanese | vegetarian | british
pricerange: cheap | moderate | expensive

Never guess or invent a value. Omit the key entirely if the user has not specified it.

═══════════════════════════════════════════════
 STATE MACHINE  (follow in order — never skip backward)
═══════════════════════════════════════════════

PHASE 0 — OFF-TOPIC
  If the user's message is unrelated to restaurants, use action=final to explain
  you are a restaurant assistant and list the available options:
  areas (centre, north, south, east, west, riverside),
  cuisines (chinese, italian, indian, japanese, vegetarian, british),
  price ranges (cheap, moderate, expensive).

PHASE 1 — SEARCHING
  Call search_restaurants with whatever slots the user provided.
  After receiving the tool_result:
    count == 0 → action=final, ask user to relax constraints. Do NOT search again.
    count > 0  → action=final, list the restaurant(s) and ask which they prefer.
                 Do NOT call search_restaurants again after this point.

PHASE 2 — RESTAURANT CHOSEN (COLLECTING BOOKING SLOTS)
  Once the user confirms a restaurant, enter this phase immediately.
  Do NOT call search_restaurants again for any reason.
  Collect missing slots ONE group at a time in this strict order:
    Step A: day
    Step B: time
    Step C: number of people
    Step D: customer name
  If the user provides multiple slots at once, accept them all and ask only for what is still missing.
  IMPORTANT: customer_name must be an explicit name given by the user (e.g. 'Oscar', 'John').
  Do NOT infer or assume the name from context. If the user has not stated a name,
  ask: 'What name should the reservation be under?'
  Do NOT mention food or dishes until ALL of steps A-D are complete.

PHASE 3 — OPTIONAL FOOD PRE-ORDER
  Once all booking slots (day, time, people, name) are confirmed:
    1. Call get_menu with the chosen restaurant_id.
    2. Use action=final to display the menu — show item_id, name and price for each item.
    3. Ask: "Would you like to pre-order any dishes? (optional — you can also order at the restaurant)"
    4. If the user names a dish by name, match it to the correct item_id yourself.
    5. If the user says no or skips, proceed with an empty dishes dict {}.

PHASE 4 — BOOKING CONFIRMATION
  Present a full summary and ask for confirmation before calling book_restaurant:
    "Please confirm: [Restaurant Name] on [day] at [time] for [N] people, under [name].
     Pre-order: [dish summary or 'none']. Is this correct?"
  Only call book_restaurant AFTER the user explicitly confirms.
  NEVER call book_restaurant as your first response after receiving food items.
  Always show the summary with action=final first, wait for confirmation, then book.
  Never call book_restaurant with empty strings, 0, or null values.

PHASE 5 — BOOKING CONFIRMED
  After a successful book_restaurant call, use action=final to report:
    Restaurant Name, Address, Day, Time, People, Customer Name,
    Order ID (ORD-XXXXXX), and pre-ordered dishes with total price if any.
  Then ask: "Is there anything else I can help you with, or would you like to end the session?"

═══════════════════════════════════════════════
 ORDER MANAGEMENT (available at any time)
═══════════════════════════════════════════════
- check_order  → show all order details (works even for canceled orders).
- cancel_order → cancel immediately and confirm to the user.
- update_order → replace dishes; format is {"item_id": quantity} e.g. {"c02": 2}.
    After a successful update, always confirm the new order summary and ask:
    'Is there anything else I can help you with, or would you like to end the session?'
- If the tool returns {"error": "Order not found."} → use action=final to say that order number does not exist.
"""


def call_ollama(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.0,
    timeout: int = 180,
) -> Tuple[str, int]:
    """Invoke a local Ollama chat completion endpoint and return (content, latency_ms)."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    t0 = time.time()
    r = requests.post(ollama_url, json=payload, timeout=timeout)
    latency_ms = int((time.time() - t0) * 1000)
    r.raise_for_status()
    return r.json()["message"]["content"], latency_ms


class RestaurantDialogueAgent:
    """Stateful 6-phase restaurant booking & order management conversational agent."""

    def __init__(
        self,
        db: Optional[RestaurantDatabase] = None,
        llm_fn: Optional[Callable[[List[Dict[str, str]]], Tuple[str, int]]] = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_hops: int = 12,
    ) -> None:
        self.db = db or RestaurantDatabase()
        self.executor = ToolExecutor(self.db)
        self.llm_fn = llm_fn or call_ollama
        self.system_prompt = system_prompt
        self.max_hops = max_hops
        self.messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        self.restaurant_chosen: bool = False

    def reset(self) -> None:
        """Reset conversation state and restaurant_chosen guardrail flag."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.restaurant_chosen = False

    def _build_dynamic_instruction(self, action: str, result: Dict[str, Any]) -> str:
        """Construct dynamic state-machine guardrail instructions injected alongside tool results."""
        if action == "search_restaurants":
            if result.get("count", 0) > 0:
                base_inst = (
                    "Search returned results. Use action=final to list the restaurants found "
                    "and ask the user which one they would like to book. "
                    "Do NOT assume the user has chosen yet. Wait for their confirmation."
                )
            else:
                base_inst = (
                    "Search returned 0 results. Use action=final to ask the user to relax "
                    "their constraints(area, food, pricerange). Do NOT search again."
                )
        elif self.restaurant_chosen:
            base_inst = (
                "A restaurant has already been chosen and booked. "
                "Do NOT call search_restaurants again."
            )
        else:
            base_inst = (
                "Continue helping the user collect booking slots "
                "(day, time, people, name). "
                "Do NOT call search_restaurants again under any circumstances."
            )

        return (
            base_inst
            + " If error=='Order not found.', tell the user that order number does not exist."
            + " If book_restaurant succeeded, report the full booking summary including order_id."
        )

    def step(self, user_text: str, max_hops: Optional[int] = None) -> Tuple[str, List[Dict[str, Any]], bool]:
        """Process a single user utterance through the multi-hop JSON action loop."""
        hops_limit = max_hops or self.max_hops
        trace: List[Dict[str, Any]] = []
        self.messages.append({"role": "user", "content": user_text})

        for hop in range(1, hops_limit + 1):
            assistant_text, latency_ms = self.llm_fn(self.messages)
            self.messages.append({"role": "assistant", "content": assistant_text})

            ev: Dict[str, Any] = {
                "hop": hop,
                "latency_ms": latency_ms,
                "raw": assistant_text,
                "action": None,
                "tool": None,
                "tool_args": None,
                "tool_ok": None,
                "tool_result": None,
                "error": None,
            }

            try:
                obj = extract_json_action(assistant_text)
            except Exception as e:
                ev["error"] = f"json_parse_error: {repr(e)}"
                trace.append(ev)
                self.messages.append(
                    {"role": "user", "content": "Invalid format. Output ONLY one JSON object."}
                )
                continue

            action = obj.get("action")
            ev["action"] = action

            if action == "final":
                trace.append(ev)
                return str(obj.get("response", "")), trace, True

            if action in TOOL_ACTIONS:
                args = obj.get("args") or {}
                if not isinstance(args, dict):
                    args = {}
                ev["tool"] = action
                ev["tool_args"] = args
                result = self.executor.execute(action, args)
                ev["tool_result"] = result
                ev["tool_ok"] = bool(result.get("success") or result.get("count") is not None)
                trace.append(ev)

                if action in ("book_restaurant", "create_order") and result.get("success"):
                    self.restaurant_chosen = True

                instruction = self._build_dynamic_instruction(action, result)
                self.messages.append(
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"tool_result": result, "instruction": instruction},
                            ensure_ascii=False,
                        ),
                    }
                )
                continue

            ev["error"] = f"unknown_action: {action}"
            trace.append(ev)
            self.messages.append(
                {"role": "user", "content": "Unknown action. Use only allowed actions."}
            )

        trace.append({"hop": hops_limit, "error": "loop_limit"})
        return "[Agent loop limit reached]", trace, False
