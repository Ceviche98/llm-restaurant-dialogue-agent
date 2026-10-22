"""Comprehensive unit tests for the LLM Restaurant Dialogue Agent, RAG Retriever, and Fine-Tuning Suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from restaurant_agent import (
    AREA_MAP,
    BENCHMARK_RESULTS,
    FOOD_MAP,
    PRICE_MAP,
    LoRATrainingSpec,
    RestaurantDatabase,
    RestaurantDialogueAgent,
    RestaurantKnowledgeRetriever,
    SFTTrainingSpec,
    ScenarioEvaluator,
    ToolExecutor,
    build_retrieval_query,
    compute_f1,
    evaluate_predictions,
    extract_json_action,
    format_chat_messages_manual,
    format_rag_dialogue_prompt,
    normalize_answer,
    normalize_slot,
    passage_to_text,
)


@pytest.fixture()
def temp_db(tmp_path: Path) -> RestaurantDatabase:
    orders_file = tmp_path / "orders.json"
    return RestaurantDatabase(orders_path=orders_file)


def test_normalize_slots() -> None:
    assert normalize_slot("in the city center", AREA_MAP) == "centre"
    assert normalize_slot("downtown", AREA_MAP) == "centre"
    assert normalize_slot("veggie", FOOD_MAP) == "vegetarian"
    assert normalize_slot("moderately priced", PRICE_MAP) == "moderate"
    assert normalize_slot("unknown_val", AREA_MAP) is None
    assert normalize_slot(None, AREA_MAP) is None


def test_database_search_and_menus(temp_db: RestaurantDatabase) -> None:
    res_chinese = temp_db.search_restaurants(area="centre", food="chinese", pricerange="cheap")
    assert res_chinese["count"] == 1
    assert res_chinese["results"][0]["id"] == "r001"
    assert res_chinese["results"][0]["name"] == "Centre Noodle House"

    res_veg = temp_db.search_restaurants(dietary="vegetarian")
    assert res_veg["count"] == 1
    assert res_veg["results"][0]["id"] == "r005"

    res_empty = temp_db.search_restaurants(area="south", food="japanese")
    assert res_empty["count"] == 0

    menu_r001 = temp_db.get_menu("r001")
    assert menu_r001["restaurant_id"] == "r001"
    assert len(menu_r001["menu"]) == 5

    menu_missing = temp_db.get_menu("r999")
    assert "error" in menu_missing


def test_booking_lifecycle_and_non_destructive_cancel(temp_db: RestaurantDatabase) -> None:
    # Invalid people count
    err_ppl = temp_db.book_restaurant("r001", "Monday", "18:00", 0, "Oscar")
    assert "error" in err_ppl

    # Missing customer name
    err_name = temp_db.book_restaurant("r001", "Monday", "18:00", 4, "   ")
    assert "error" in err_name

    # Valid booking with 3x Kung Pao Chicken (c02 @ £10.00 = £30.00)
    booked = temp_db.book_restaurant(
        restaurant_id="r001",
        day="Monday",
        time="18:00",
        people=4,
        customer_name="Oscar Moreno",
        dishes={"c02": 3},
        order_id="ORD-187648",
    )
    assert booked["success"] is True
    assert booked["order_id"] == "ORD-187648"
    assert booked["details"]["total_price"] == 30.0
    assert booked["details"]["status"] == "confirmed"

    # Check active order
    checked = temp_db.check_order("ORD-187648")
    assert checked["success"] is True
    assert checked["order"]["customer_name"] == "Oscar Moreno"

    # Update order dishes: 2x Fried Rice (c04 @ 7.5) + 1x Spring Rolls (c05 @ 5.0) = £20.00
    updated = temp_db.update_order("ORD-187648", {"c04": 2, "c05": 1})
    assert updated["success"] is True
    assert updated["details"]["total_price"] == 20.0

    # Cancel order non-destructively
    canceled = temp_db.cancel_order("ORD-187648")
    assert canceled["success"] is True
    assert canceled["details"]["status"] == "canceled"

    # Verify canceled order cannot be updated
    update_after_cancel = temp_db.update_order("ORD-187648", {"c01": 1})
    assert update_after_cancel["error"] == "Cannot update a canceled order."

    # Verify canceled order remains auditable via check_order (CW1 Task 5)
    checked_after_cancel = temp_db.check_order("ORD-187648")
    assert checked_after_cancel["success"] is True
    assert checked_after_cancel["order"]["status"] == "canceled"

    # Non-existent order ID (CW1 Task 3)
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
    # Simulate an LLM that:
    # 1. Outputs malformed text on Hop 1
    # 2. Recovers on Hop 2 by calling search_restaurants
    # 3. Summarizes with action=final on Hop 3
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


def test_rag_engine_query_formatting_and_retrieval(temp_db: RestaurantDatabase) -> None:
    history = [
        ("A", "Hey, do you happen to know what was the 2009 Santos FC season?"),
        ("B", "Yes, it was Santos Futebol Clube's ninety-seventh season."),
        ("A", "Which of these players retired in 2013?"),
        ("B", "Sérgio retired in 2013."),
    ]
    current_q = "What team did he move from?"
    rq = build_retrieval_query(history, current_q, max_turns=3)
    assert "Sérgio retired in 2013." in rq
    assert current_q in rq

    passage = {"title": "Gevora Hotel", "text": "The Gevora Hotel has 528 rooms spread over 75 floors."}
    p_text = passage_to_text(passage)
    assert "Title: Gevora Hotel" in p_text
    assert "528 rooms" in p_text

    msgs = format_rag_dialogue_prompt(history, current_q, docs_text=[p_text])
    assert msgs[0]["role"] == "system"
    assert "CONTEXT:" in msgs[-1]["content"]

    chatml = format_chat_messages_manual(msgs)
    assert "<|im_start|>system" in chatml
    assert chatml.endswith("<|im_start|>assistant\n")

    retriever = RestaurantKnowledgeRetriever.from_database(temp_db.restaurants, temp_db.menus)
    top_docs = retriever.search_document_demo("Japanese sushi and ramen in the east", n_docs=2)
    assert len(top_docs) == 2
    assert top_docs[0]["id"] == "r003"


def test_finetuning_metrics_and_specs() -> None:
    assert normalize_answer("The Gevora Hotel has 528 rooms!") == "gevora hotel has 528 rooms"
    f1_exact = compute_f1("You're welcome!", "You're welcome.")
    assert f1_exact == 1.0

    f1_partial = compute_f1(
        "The Gevora Hotel has 528 rooms spread over 75 floors.",
        "Yes, Gevora Hotel have 528 total rooms",
    )
    assert abs(f1_partial - 0.5) < 1e-4

    eval_out = evaluate_predictions(["You're welcome!"], ["You're welcome."])
    assert eval_out["average_f1"] == 100.0
    assert eval_out["total"] == 1

    lora_spec = LoRATrainingSpec(r=16, lora_alpha=32)
    assert lora_spec.to_peft_kwargs()["r"] == 16
    assert "q_proj" in lora_spec.to_peft_kwargs()["target_modules"]

    sft_spec = SFTTrainingSpec()
    assert sft_spec.to_dict()["optim"] == "adamw_bnb_8bit"
    assert BENCHMARK_RESULTS["smoltalk_finetuning"]["lora_r16_alpha32_8gb_vram"]["f1"] == 31.88


def test_scenario_evaluator_all_scenarios_pass(temp_db: RestaurantDatabase) -> None:
    evaluator = ScenarioEvaluator(db=temp_db)
    summary = evaluator.evaluate_all()
    assert summary["total_scenarios"] == 4
    assert summary["passed_scenarios"] == 4
    assert summary["pass_rate"] == 100.0
