#!/usr/bin/env python3
"""Interactive CLI for the Task-Oriented LLM Restaurant Booking & Ordering Agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from restaurant_agent import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    RestaurantDatabase,
    RestaurantDialogueAgent,
    RuleBasedMockLLM,
    call_ollama,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 6-Phase Task-Oriented LLM Restaurant Dialogue Agent."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model identifier (default: llama3.1:8b-instruct-q4_K_M)",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Ollama chat completion endpoint URL",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic offline RuleBasedMockLLM instead of live Ollama server",
    )
    args = parser.parse_args()

    db = RestaurantDatabase()
    if args.mock:
        llm_fn = RuleBasedMockLLM(db)
    else:
        llm_fn = lambda msgs: call_ollama(msgs, model=args.model, ollama_url=args.ollama_url)

    agent = RestaurantDialogueAgent(db=db, llm_fn=llm_fn)
    print("=" * 72)
    print(" LLM Restaurant Booking & Order Assistant (6-Phase State Machine)")
    print(" Type 'exit' or 'quit' to terminate, 'reset' to clear session state.")
    print("=" * 72)

    while True:
        try:
            user_input = input("\n[USER]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting session.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Session terminated.")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("[SYSTEM]: Session state reset.")
            continue

        reply, trace, _ = agent.step(user_input)
        for ev in trace:
            if ev.get("tool"):
                print(f"  -> [TOOL CALL] {ev['tool']}({ev['tool_args']})")
        print(f"[ASSISTANT]: {reply}")


if __name__ == "__main__":
    main()
