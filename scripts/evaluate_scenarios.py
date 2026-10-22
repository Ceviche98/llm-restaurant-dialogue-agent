#!/usr/bin/env python3
"""Evaluate multi-turn restaurant dialogue scenarios over data/test_scenarios.json."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from restaurant_agent import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    RestaurantDatabase,
    ScenarioEvaluator,
    call_ollama,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run automated multi-turn scenario benchmarks on test_scenarios.json."
    )
    parser.add_argument(
        "--live-ollama",
        action="store_true",
        help="Use live local Ollama server instead of deterministic mock policy",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        orders_file = Path(tmpdir) / "orders_eval.json"
        db = RestaurantDatabase(orders_path=orders_file)
        evaluator = ScenarioEvaluator(db=db)

        llm_fn = None
        if args.live_ollama:
            llm_fn = lambda msgs: call_ollama(msgs, model=args.model, ollama_url=args.ollama_url)

        summary = evaluator.evaluate_all(llm_fn=llm_fn)

    print("=" * 72)
    print(
        f"SCENARIO EVALUATION SUMMARY: {summary['passed_scenarios']}/{summary['total_scenarios']} "
        f"Passed ({summary['pass_rate']:.1f}%)"
    )
    print("=" * 72)
    for sc in summary["scenarios"]:
        status = "PASS" if sc["passed"] else "FAIL"
        print(f"[{status}] {sc['scenario_id']} ({sc['num_turns']} turns): {sc['description']}")
        print(f"       Checks: {json.dumps(sc['checks'])}")


if __name__ == "__main__":
    main()
