"""LLM Restaurant Dialogue Agent, Dense RAG Retriever & LoRA Fine-Tuning Suite."""

from .agent import DEFAULT_MODEL, DEFAULT_OLLAMA_URL, SYSTEM_PROMPT, RestaurantDialogueAgent, call_ollama
from .database import (
    AREA_MAP,
    FOOD_MAP,
    PRICE_MAP,
    VALID_AREAS,
    VALID_FOODS,
    VALID_PRICES,
    RestaurantDatabase,
    normalize_slot,
)
from .evaluator import RuleBasedMockLLM, ScenarioEvaluator
from .finetuning import (
    BENCHMARK_RESULTS,
    LoRATrainingSpec,
    SFTTrainingSpec,
    compute_f1,
    evaluate_predictions,
    normalize_answer,
)
from .rag_engine import (
    RestaurantKnowledgeRetriever,
    build_retrieval_query,
    format_chat_messages_manual,
    format_rag_dialogue_prompt,
    passage_to_text,
)
from .tools import TOOL_ACTIONS, TOOL_SCHEMAS, ToolExecutor, extract_json_action

__all__ = [
    "AREA_MAP",
    "BENCHMARK_RESULTS",
    "DEFAULT_MODEL",
    "DEFAULT_OLLAMA_URL",
    "FOOD_MAP",
    "LoRATrainingSpec",
    "PRICE_MAP",
    "RestaurantDatabase",
    "RestaurantDialogueAgent",
    "RestaurantKnowledgeRetriever",
    "RuleBasedMockLLM",
    "SFTTrainingSpec",
    "SYSTEM_PROMPT",
    "ScenarioEvaluator",
    "TOOL_ACTIONS",
    "TOOL_SCHEMAS",
    "ToolExecutor",
    "VALID_AREAS",
    "VALID_FOODS",
    "VALID_PRICES",
    "build_retrieval_query",
    "call_ollama",
    "compute_f1",
    "evaluate_predictions",
    "extract_json_action",
    "format_chat_messages_manual",
    "format_rag_dialogue_prompt",
    "normalize_answer",
    "normalize_slot",
    "passage_to_text",
]
