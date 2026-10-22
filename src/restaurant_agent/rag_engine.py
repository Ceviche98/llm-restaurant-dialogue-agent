"""Retrieval-Augmented Generation (RAG) pipeline for grounded dialogue generation (Lab 4)."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def passage_to_text(p: Dict[str, Any], max_chars: int = 1200) -> str:
    """Convert a retrieved passage dictionary into a normalized text string."""
    title = p.get("title", "")
    text = p.get("text", "") or p.get("contents", "") or ""
    if title and text:
        s = f"Title: {title}\nText: {text}"
    elif text:
        s = text
    else:
        s = json.dumps(p, ensure_ascii=False)
    return s.replace("\n", " ").strip()[:max_chars]


def build_retrieval_query(
    history: List[Tuple[str, str]],
    current_query: str,
    max_turns: int = 6,
    max_chars: int = 1200,
) -> str:
    """Build a multi-turn retrieval query from recent dialogue history and current user query."""
    query_parts = [current_query]
    for i in range(len(history) - 1, -1, -1):
        if len(query_parts) < max_turns:
            _, utterance = history[i]
            query_parts.insert(0, utterance)
        else:
            break
    retrieval_query = " ".join(query_parts)
    return retrieval_query[:max_chars]


def format_rag_dialogue_prompt(
    history: List[Tuple[str, str]],
    current_query: str,
    docs_text: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Format multi-turn dialogue messages with optional retrieved CONTEXT passages."""
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use the provided CONTEXT to answer when relevant. "
                "If the answer is not in the CONTEXT, say you don't know."
            ),
        }
    ]
    for speaker, utterance in history:
        messages.append(
            {"role": "user" if speaker == "A" else "assistant", "content": utterance}
        )

    if docs_text:
        context = "\nCONTEXT:\n" + "\n".join(docs_text)
        messages.append({"role": "user", "content": context + "\n\n" + current_query})
    else:
        messages.append({"role": "user", "content": current_query})
    return messages


def format_chat_messages_manual(messages_list: List[Dict[str, str]]) -> str:
    """Fallback ChatML manual formatter when tokenizer.apply_chat_template is unavailable."""
    formatted_str = ""
    for msg in messages_list:
        role = msg["role"]
        content = msg["content"]
        if role in ("system", "user", "assistant"):
            formatted_str += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return formatted_str + "<|im_start|>assistant\n"


class RestaurantKnowledgeRetriever:
    """Lightweight TF-IDF + BM25-style cosine similarity retriever over restaurant & menu documents."""

    def __init__(self, documents: Optional[List[Dict[str, Any]]] = None) -> None:
        self.documents: List[Dict[str, Any]] = documents or []
        self._doc_tokens: List[Counter[str]] = []
        self._idf: Dict[str, float] = {}
        if self.documents:
            self._build_index()

    @classmethod
    def from_database(
        cls,
        restaurants: List[Dict[str, Any]],
        menus: Dict[str, List[Dict[str, Any]]],
    ) -> "RestaurantKnowledgeRetriever":
        """Index restaurants and their menu catalogs into searchable passage documents."""
        docs: List[Dict[str, Any]] = []
        for r in restaurants:
            rid = r["id"]
            cuisine = r.get("cuisine") or r.get("food", "")
            menu_items = menus.get(rid, [])
            menu_str = ", ".join(
                f"{item['name']} ({item['item_id']}, £{item['price']:.2f})"
                for item in menu_items
            )
            text = (
                f"{r['name']} (ID: {rid}) is a {r['pricerange']} {cuisine} restaurant "
                f"located in the {r['area']} area at {r['address']} (Phone: {r.get('phone', 'N/A')}). "
                f"Menu items: {menu_str}."
            )
            docs.append({"id": rid, "title": r["name"], "text": text})
        return cls(docs)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_index(self) -> None:
        self._doc_tokens = []
        df: Counter[str] = Counter()
        for doc in self.documents:
            full_text = f"{doc.get('title', '')} {doc.get('text', '')}"
            tokens = Counter(self._tokenize(full_text))
            self._doc_tokens.append(tokens)
            for term in tokens:
                df[term] += 1
        n_docs = max(len(self.documents), 1)
        self._idf = {
            term: math.log((n_docs + 1) / (freq + 1)) + 1.0 for term, freq in df.items()
        }

    def search_document_demo(self, query: str, n_docs: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant knowledge documents for a given query."""
        if not self.documents:
            return []
        q_tokens = Counter(self._tokenize(query))
        q_vec = {t: count * self._idf.get(t, 1.0) for t, count in q_tokens.items()}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        scored: List[Tuple[float, int]] = []
        for idx, d_tokens in enumerate(self._doc_tokens):
            d_vec = {t: count * self._idf.get(t, 1.0) for t, count in d_tokens.items()}
            d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
            dot = sum(q_vec[t] * d_vec.get(t, 0.0) for t in q_vec)
            score = dot / (q_norm * d_norm)
            scored.append((score, idx))

        scored.sort(key=lambda x: (-x[0], x[1]))
        top_k = max(1, min(int(n_docs), len(self.documents)))
        results = []
        for score, idx in scored[:top_k]:
            doc_copy = dict(self.documents[idx])
            doc_copy["score"] = round(score, 4)
            results.append(doc_copy)
        return results
