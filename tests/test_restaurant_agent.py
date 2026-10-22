"""Comprehensive unit tests for the LLM Restaurant Dialogue Agent with Clarification DM."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from restaurant_agent import (
    AREA_MAP,
    FOOD_MAP,
    PRICE_MAP,
    RestaurantDatabase,
    RestaurantDialogueAgent,
    ScenarioEvaluator,
    ToolExecutor,
    build_clarification_response,
    decide_clarification,
    dm_preprocess,
    extract_booking_fields,
    extract_json_action,
    normalize_slot,
    normalize_slot_verbose,
    parse_user_constraints,
    update_state_from_parsed,
)


@pytest.fixture()
def temp_db(tmp_path: Path) -> RestaurantDatabase:
    orders_file = tmp_path / "orders.json"
    return RestaurantDatabase(orders_path=orders_file)


def test_normalize_slots() -> None:
    assert normalize_slot("centre", AREA_MAP) == "centre"
    assert normalize_slot("downtown", AREA_MAP) == "centre"
    assert normalize_slot("riverside area", AREA_MAP) == "riverside"
    assert normalize_slot("veggie", FOOD_MAP) == "vegetarian"
    assert normalize_slot("inexpensive", PRICE_MAP) == "cheap"
    assert normalize_slot("luxury", PRICE_MAP) is None


def test_database_search_and_menus(temp_db: RestaurantDatabase) -> None:
    all_res = temp_db.search_restaurants()
    assert all_res["count"] == 6

    res = temp_db.search_restaurants(area="centre", food="chinese", pricerange="cheap")
    assert res["count"] == 1
    assert res["results"][0]["id"] == "r001"
    assert res["results"][0]["name"] == "Centre Noodle House"

    res_empty = temp_db.search_restaurants(area="riverside", food="chinese")
    assert res_empty["count"] == 0

    menu = temp_db.get_menu("r001")
    assert menu["restaurant_id"] == "r001"
    assert any(item["item_id"] == "c01" for item in menu["menu"])

    # 1. Create booking
    booking = temp_db.book_restaurant(
        restaurant_id="r001",
        day="Friday",
        time="19:30",
        people=3,
        customer_name="Oscar",
        dishes={"c01": 2, "c02": 1},
    )
    assert booking["success"] is True
    order_id = booking["order_id"]
    assert order_id.startswith("ORD-")
    assert booking["details"]["total_price"] > 0

    # 2. Check order
    checked = temp_db.check_order(order_id)
    assert checked["success"] is True
    assert checked["order"]["status"] == "confirmed"
    assert checked["order"]["customer_name"] == "Oscar"

    # 3. Update order
    updated = temp_db.update_order(order_id, dishes={"c01": 1, "c03": 2})
    assert updated["success"] is True
    assert updated["details"]["total_price"] > 0

    # 4. Cancel order
    cancelled = temp_db.cancel_order(order_id)
    assert cancelled["success"] is True

    # 5. Non-destructive check
    checked_after = temp_db.check_order(order_id)
    assert checked_after["success"] is True
    assert checked_after["order"]["status"] in ("canceled", "cancelled")

    # Non-existent order ID
    not_found = temp_db.check_order("ORD-999999")
    assert not_found == {"error": "Order not found."}


def test_tool_executor_and_json_parser(temp_db: RestaurantDatabase) -> None:
    executor = ToolExecutor(temp_db)
    res = executor.execute("search_restaurants", {"area": "east", "food": "japanese"})
    assert res["count"] == 1
    assert res["results"][0]["id"] == "r003"

    unknown = executor.execute("nonexistent_tool", {})
    assert unknown["ok"] is False

    parsed = extract_json_action('```json\n{"action": "final", "response": "Hello!"}\n```')
    assert parsed["action"] == "final"
    assert parsed["response"] == "Hello!"


def test_agent_multi_hop_and_error_recovery(temp_db: RestaurantDatabase) -> None:
    responses = [
        "Not valid JSON",
        json.dumps({"action": "search_restaurants", "args": {"food": "chinese"}}),
        json.dumps({"action": "final", "response": "We have Centre Noodle House (r001)."}),
    ]
    idx = {"i": 0}

    def custom_llm(_messages):
        out = responses[idx["i"]]
        idx["i"] += 1
        return out, 10

    agent = RestaurantDialogueAgent(db=temp_db, llm_fn=custom_llm)
    reply, trace, ok = agent.step("I want Chinese food")
    assert ok is True
    assert "Centre Noodle House" in reply
    assert len(trace) == 3
    assert trace[0]["error"] is not None
    assert trace[1]["tool"] == "search_restaurants"
    assert trace[2]["action"] == "final"


def test_clarification_dm_slot_normalisation() -> None:
    # 1. Exact mapping
    m_exact = normalize_slot_verbose("centre", AREA_MAP)
    assert m_exact["status"] == "mapped"
    assert m_exact["value"] == "centre"
    assert m_exact["confidence"] == 1.0

    # 2. Substring matching
    m_sub = normalize_slot_verbose("I would like to sit in the city center area please", AREA_MAP)
    assert m_sub["status"] == "substring"
    assert m_sub["value"] == "centre"
    assert m_sub["confidence"] >= 0.95

    # 3. Fuzzy matching on noisy speech transcript (e.g. "lo priced", "japnese")
    m_fuzzy_food = normalize_slot_verbose("some japnese dining", FOOD_MAP)
    assert m_fuzzy_food["status"] == "fuzzy"
    assert m_fuzzy_food["value"] == "japanese"
    assert m_fuzzy_food["confidence"] >= 0.72

    m_fuzzy_price = normalize_slot_verbose("I want some lo priced food", PRICE_MAP)
    assert m_fuzzy_price["status"] == "fuzzy"
    assert m_fuzzy_price["value"] == "cheap"

    # 4. Unknown / Out-of-vocabulary
    m_unknown = normalize_slot_verbose("barbecue ribs and steaks", FOOD_MAP)
    assert m_unknown["status"] == "unknown"
    assert m_unknown["value"] is None

    # 5. Missing
    m_missing = normalize_slot_verbose("", AREA_MAP)
    assert m_missing["status"] == "missing"
    assert m_missing["confidence"] == 0.0


def test_clarification_dm_policies_and_preprocessing() -> None:
    state = {
        "constraints": {"area": None, "food": None, "pricerange": None},
        "pending_clarification": None,
    }

    # Case 1: Out-of-vocabulary cuisine mentioned -> explicit clarification with options
    resp1, state, parsed1 = dm_preprocess("I want some mexican food please", state)
    assert resp1 is not None
    assert resp1["action"] == "explicit_clarification"
    assert "didn't quite catch the food" in resp1["response"]
    assert "chinese, italian, indian" in resp1["response"]

    # Case 2: Clear food specified -> updates state and asks follow-up for missing slot
    state["pending_clarification"] = None
    resp2, state, parsed2 = dm_preprocess("I want delicious Italian cuisine", state)
    assert state["constraints"]["food"] == "italian"
    assert resp2 is not None
    assert resp2["action"] == "follow_up"

    # Case 3: Booking fields extraction
    booking_slots = extract_booking_fields("table for 4 people on Friday at 19:30")
    assert booking_slots["people"] == 4
    assert booking_slots["day"] == "friday"
    assert booking_slots["time"] == "19:30"


def test_agent_with_clarification_dm(temp_db: RestaurantDatabase) -> None:
    agent = RestaurantDialogueAgent(db=temp_db)
    # The agent intercepts an OOV food request directly via DM without needing LLM invocation
    reply, trace, ok = agent.step_with_clarification("I am looking for some greek food in town")
    assert ok is True
    assert trace[0]["dm_handled"] is True
    assert "didn't quite catch the food" in reply


def test_scenario_evaluator_all_scenarios_pass(temp_db: RestaurantDatabase) -> None:
    evaluator = ScenarioEvaluator(db=temp_db)
    summary = evaluator.evaluate_all()
    assert summary["total_scenarios"] == 4
    assert summary["passed_scenarios"] == 4
    assert summary["pass_rate"] == 100.0
