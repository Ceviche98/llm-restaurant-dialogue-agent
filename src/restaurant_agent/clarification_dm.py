"""
Dialogue Manager for Clarification & Error Handling under Noisy / Uncertain User Input.

Implements Lab 9 specifications:
1. Slot Normalisation (`normalize_slot_verbose`):
   - Categorizes hypotheses into {"missing", "mapped", "substring", "fuzzy", "unknown"}.
   - Computes confidence scores via token-level n-gram Levenshtein / SequenceMatcher.
2. Clarification Policy (`decide_clarification`):
   - Missing required slot -> Follow-up query.
   - Vague / OOV slot -> Explicit clarification presenting valid ontology options.
   - Low confidence (< 0.84) -> Explicit confirmation ("I think you said X. Is that right?").
   - Medium confidence (0.84 - 0.94) -> Implicit confirmation (carry forward hypothesis into next turn).
   - High confidence (>= 0.94) -> Direct state commit without interruption.
3. Safe State Update (`update_state_from_parsed`):
   - Preserves confirmed state values against noisy or unknown speech hypotheses.
4. DM Preprocessing (`dm_preprocess`):
   - Intercepts user input, evaluates confidence, manages dialogue moves, and passes
     grounded state to the LLM agent.
"""

from __future__ import annotations

import difflib
import re
import string
from typing import Any, Dict, List, Optional, Tuple

AREA_MAP: Dict[str, str] = {
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
    "riverside area": "riverside",
}

FOOD_MAP: Dict[str, str] = {
    "chinese": "chinese",
    "italian": "italian",
    "indian": "indian",
    "japanese": "japanese",
    "vegetarian": "vegetarian",
    "veggie": "vegetarian",
    "british": "british",
    "english": "british",
}

PRICE_MAP: Dict[str, str] = {
    "cheap": "cheap",
    "inexpensive": "cheap",
    "budget": "cheap",
    "low": "cheap",
    "low priced": "cheap",
    "moderate": "moderate",
    "medium": "moderate",
    "moderately priced": "moderate",
    "expensive": "expensive",
    "pricey": "expensive",
    "high": "expensive",
    "high end": "expensive",
}

VALID_AREAS: List[str] = ["centre", "north", "south", "east", "west", "riverside"]
VALID_FOODS: List[str] = ["chinese", "italian", "indian", "japanese", "vegetarian", "british"]
VALID_PRICES: List[str] = ["cheap", "moderate", "expensive"]

AREA_HINTS: List[str] = [
    "area", "location", "part of town", "part", "side", "district",
    "centre", "center", "central", "north", "south", "east", "west",
    "riverside", "river side", "downtown",
]

FOOD_HINTS: List[str] = [
    "food", "cuisine", "eat", "meal",
    "chinese", "italian", "indian", "japanese", "vegetarian", "veggie",
    "british", "english", "asian", "oriental",
]

PRICE_HINTS: List[str] = [
    "price", "priced", "cost", "cheap", "budget", "expensive", "pricey",
    "moderate", "mid", "medium", "midscale", "premium", "high end",
]


def normalize_slot_verbose(
    value: Any,
    mapping: Dict[str, str],
    fuzzy_cutoff: float = 0.72,
) -> Dict[str, Any]:
    """
    Normalize raw user text against a canonical ontology mapping.

    Returns:
        dict:
            - raw: original input value
            - value: canonical mapped value or None
            - status: {"missing", "mapped", "substring", "fuzzy", "unknown"}
            - confidence: float score in [0.0, 1.0]
            - candidates: list of candidate values
    """
    if not value or str(value).strip() == "":
        return {
            "raw": value,
            "value": None,
            "status": "missing",
            "confidence": 0.0,
            "candidates": [],
        }

    val_str = str(value).strip().lower()

    # 1. Exact match
    if val_str in mapping:
        return {
            "raw": value,
            "value": mapping[val_str],
            "status": "mapped",
            "confidence": 1.0,
            "candidates": [mapping[val_str]],
        }

    # 2. Substring match
    candidates = []
    for key, canonical in mapping.items():
        # Match as whole word substring or phrase
        if re.search(rf"\b{re.escape(key)}\b", val_str):
            candidates.append(canonical)

    if candidates:
        unique_candidates = sorted(list(set(candidates)))
        return {
            "raw": value,
            "value": unique_candidates[0],
            "status": "substring",
            "confidence": 0.95,
            "candidates": unique_candidates,
        }

    # 3. Fuzzy matching on 1-grams and 2-grams
    words = [w.strip(string.punctuation) for w in val_str.split() if w.strip(string.punctuation)]
    ngrams = words.copy()
    for i in range(len(words) - 1):
        ngrams.append(f"{words[i]} {words[i+1]}")

    best_match = None
    best_score = 0.0

    for ngram in ngrams:
        for key in mapping:
            score = difflib.SequenceMatcher(None, ngram, key).ratio()
            if score > best_score:
                best_score = score
                best_match = key

    if best_score >= fuzzy_cutoff and best_match is not None:
        return {
            "raw": value,
            "value": mapping[best_match],
            "status": "fuzzy",
            "confidence": float(best_score),
            "candidates": [mapping[best_match]],
        }

    # 4. Unknown
    return {
        "raw": value,
        "value": None,
        "status": "unknown",
        "confidence": 0.0,
        "candidates": [],
    }


def detect_slot_mention(text: str, hints: List[str]) -> bool:
    """Check if any semantic indicator or hint is present in the utterance."""
    return any(h in text for h in hints)


def parse_slot_with_hints(
    text: str, mapping: Dict[str, str], hints: List[str]
) -> Dict[str, Any]:
    """Parse a single slot, recording whether the concept was mentioned even if unmapped."""
    info = normalize_slot_verbose(text, mapping)
    if info["status"] == "unknown":
        if detect_slot_mention(text, hints):
            info["mentioned"] = True
        else:
            info["status"] = "missing"
            info["mentioned"] = False
    else:
        info["mentioned"] = info["status"] != "missing"
    return info


def parse_user_constraints(user_text: str) -> Dict[str, Dict[str, Any]]:
    """Extract search constraint slot hypotheses from user utterance."""
    text = user_text.lower()
    return {
        "area": parse_slot_with_hints(text, AREA_MAP, AREA_HINTS),
        "food": parse_slot_with_hints(text, FOOD_MAP, FOOD_HINTS),
        "pricerange": parse_slot_with_hints(text, PRICE_MAP, PRICE_HINTS),
    }


def extract_booking_fields(user_text: str) -> Dict[str, Any]:
    """Extract transaction fields (day, time, party size) from user text."""
    text = user_text.lower()

    people = None
    m_people = re.search(r"\bfor (\d+)\b", text) or re.search(r"\b(\d+) (?:people|guests|persons)\b", text)
    if m_people:
        people = int(m_people.group(1))

    time_str = None
    m_time = re.search(r"\b(\d{1,2}:\d{2})\b", text)
    if m_time:
        time_str = m_time.group(1)
    else:
        m_time2 = re.search(r"\bat (\d{1,2})(?:am|pm)?\b", text)
        if m_time2:
            hh = int(m_time2.group(1))
            if "pm" in text and hh < 12:
                hh += 12
            time_str = f"{hh:02d}:00"

    day = None
    for d in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "tomorrow"]:
        if d in text:
            day = d
            break

    return {
        "day": day,
        "time": time_str,
        "people": people,
    }


def get_next_missing_slot(state: Dict[str, Any], exclude_slot: Optional[str] = None) -> Optional[str]:
    """Return the next unfilled search constraint slot."""
    constraints = state.get("constraints", {})
    for slot in ["area", "food", "pricerange"]:
        if slot == exclude_slot:
            continue
        if constraints.get(slot) is None:
            return slot
    return None


def slot_values_text(slot: str) -> str:
    """Format allowable ontology choices for explicit clarification."""
    if slot == "area":
        return ", ".join(VALID_AREAS[:-1]) + f", or {VALID_AREAS[-1]}"
    if slot == "food":
        return ", ".join(VALID_FOODS[:-1]) + f", or {VALID_FOODS[-1]}"
    if slot == "pricerange":
        return ", ".join(VALID_PRICES[:-1]) + f", or {VALID_PRICES[-1]}"
    return ""


def follow_up_question(slot: str) -> str:
    """Generate default follow-up question for slot filling."""
    if slot == "area":
        return "What area are you looking for?"
    if slot == "food":
        return "What kind of cuisine or food would you like?"
    if slot == "pricerange":
        return "What price range do you prefer (cheap, moderate, or expensive)?"
    return "Could you provide a bit more detail?"


def decide_clarification(
    state: Dict[str, Any],
    parsed_slots: Dict[str, Dict[str, Any]],
    low_conf: float = 0.84,
    medium_conf: float = 0.94,
) -> Optional[Dict[str, Any]]:
    """
    Decide clarification action according to uncertainty policy:
    1. Vague/OOV mentioned slot -> Explicit clarification with ontology options.
    2. Low confidence (< low_conf) -> Explicit clarification with yes/no verification.
    3. Medium confidence (low_conf <= conf < medium_conf) -> Implicit clarification.
    4. Unspecified slot -> Follow-up question.
    5. Fully specified & confident -> Return None.
    """
    for slot in ["area", "food", "pricerange"]:
        info = parsed_slots.get(slot, {})
        status = info.get("status", "missing")
        conf = info.get("confidence", 0.0)

        # 1. Mentioned but unmapped / out-of-vocabulary
        if status == "unknown" and info.get("mentioned", False):
            return {
                "type": "explicit_clarification",
                "slot": slot,
                "info": info,
                "requires_options": True,
            }

        # 2. Fuzzy or substring match confidence assessment
        if status in ["fuzzy", "substring", "mapped"]:
            if conf < low_conf:
                return {
                    "type": "explicit_clarification",
                    "slot": slot,
                    "info": info,
                    "requires_options": False,
                }
            elif conf < medium_conf:
                return {
                    "type": "implicit_clarification",
                    "slot": slot,
                    "info": info,
                }

    # 3. Missing slot follow-up
    missing = get_next_missing_slot(state)
    if missing:
        return {
            "type": "follow_up",
            "slot": missing,
            "info": None,
        }

    return None


def build_clarification_response(clf: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build structured clarification response object."""
    ctype = clf.get("type")
    slot = clf.get("slot", "")
    info = clf.get("info") or {}

    if ctype == "follow_up":
        return {
            "action": "follow_up",
            "slot": slot,
            "response": follow_up_question(slot),
        }

    if ctype == "explicit_clarification":
        requires_options = clf.get("requires_options", False)
        if requires_options:
            options = slot_values_text(slot)
            return {
                "action": "explicit_clarification",
                "slot": slot,
                "response": f"I didn't quite catch the {slot}. You can choose between: {options}.",
            }
        guess = info.get("value", "that")
        return {
            "action": "explicit_clarification",
            "slot": slot,
            "guess": guess,
            "response": f"I think you said {guess}. Is that right?",
        }

    if ctype == "implicit_clarification":
        guess = info.get("value", "")
        next_slot = None
        if state is not None:
            next_slot = get_next_missing_slot(state, exclude_slot=slot)
        if not next_slot:
            next_slot = "food" if slot != "food" else "area"

        next_q = follow_up_question(next_slot)
        return {
            "action": "implicit_clarification",
            "slot": slot,
            "guess": guess,
            "response": f"Looking for {guess} {slot}. {next_q}",
        }

    return {
        "action": "none",
        "response": "How can I help you with your restaurant booking?",
    }


def update_state_from_parsed(
    state: Dict[str, Any],
    parsed_slots: Dict[str, Dict[str, Any]],
    low_conf: float = 0.84,
) -> Dict[str, Any]:
    """Safely commit slot values to dialogue state if confidence threshold is met."""
    if "constraints" not in state:
        state["constraints"] = {"area": None, "food": None, "pricerange": None}

    for slot, info in parsed_slots.items():
        status = info.get("status", "missing")
        conf = info.get("confidence", 0.0)
        val = info.get("value")

        if status in ["mapped", "substring"]:
            if val:
                state["constraints"][slot] = val
        elif status == "fuzzy" and conf >= low_conf:
            if val:
                state["constraints"][slot] = val

    return state


def dm_preprocess(
    user_text: str,
    state: Dict[str, Any],
    low_conf: float = 0.84,
    medium_conf: float = 0.94,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Dialogue Manager Preprocessing pipeline:
    1. Parses constraints from raw utterance.
    2. Safely updates dialogue state.
    3. Evaluates clarification policy.
    4. If clarification or follow-up is needed, returns response object, updated state, parsed.
    5. Otherwise returns (None, updated state, parsed) for LLM / tool processing.
    """
    if "constraints" not in state:
        state["constraints"] = {"area": None, "food": None, "pricerange": None}
    if "pending_clarification" not in state:
        state["pending_clarification"] = None

    # Handle pending explicit clarification responses (e.g., user answering "yes" or "no")
    pending = state.get("pending_clarification")
    if pending:
        user_lower = user_text.strip().lower()
        slot = pending.get("slot")
        guess = (pending.get("info") or {}).get("value")

        if any(w in user_lower for w in ["yes", "yeah", "correct", "right", "sure"]):
            if slot and guess:
                state["constraints"][slot] = guess
            state["pending_clarification"] = None
        elif any(w in user_lower for w in ["no", "nope", "wrong", "incorrect"]):
            state["pending_clarification"] = None
            if slot:
                state["constraints"][slot] = None
            options = slot_values_text(slot) if slot else ""
            resp = {
                "action": "explicit_clarification",
                "slot": slot,
                "response": f"Apologies. What {slot} would you prefer? Options: {options}.",
            }
            return resp, state, {}

    parsed = parse_user_constraints(user_text)
    state = update_state_from_parsed(state, parsed, low_conf=low_conf)
    clf = decide_clarification(state, parsed, low_conf=low_conf, medium_conf=medium_conf)

    if clf:
        if clf.get("type") == "explicit_clarification":
            state["pending_clarification"] = clf
        else:
            state["pending_clarification"] = None

        response_obj = build_clarification_response(clf, state=state)
        return response_obj, state, parsed

    state["pending_clarification"] = None
    return None, state, parsed
