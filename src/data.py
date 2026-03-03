import re
import logging
from typing import Optional

import numpy as np
from datasets import load_dataset

logger = logging.getLogger(__name__)


def extract_answer_number(solution: str) -> Optional[int]:
    """Extract numeric answer from a GSM8K solution string."""
    # GSM8K format: "#### 624"
    match = re.search(r"####\s*(-?\d[\d,]*)", solution)
    if match:
        return int(match.group(1).replace(",", ""))
    # Fallback: last number in the string
    numbers = re.findall(r"-?\d[\d,]*", solution)
    if numbers:
        return int(numbers[-1].replace(",", ""))
    return None


def count_reasoning_steps(solution: str) -> int:
    """Estimate the number of reasoning steps from a GSM8K solution."""
    lines = [l.strip() for l in solution.strip().split("\n") if l.strip()]
    # Exclude the final "#### answer" line
    steps = [l for l in lines if not l.startswith("####")]
    return max(len(steps), 1)


def load_matched_problems(
    n_problems: int = 30,
    seed: int = 42,
    en_dataset: str = "openai/gsm8k",
    tr_dataset: str = "malhajar/gsm8k-tr",
    en_split: str = "test",
    tr_split: str = "test",
) -> list[dict]:
    """
    Load English and Turkish GSM8K problems matched by index.
    Returns a list of matched problem dicts with verified answer alignment.
    """
    logger.info("Loading English dataset: %s", en_dataset)
    ds_en = load_dataset(en_dataset, "main", split=en_split)

    logger.info("Loading Turkish dataset: %s", tr_dataset)
    ds_tr = load_dataset(tr_dataset, "main", split=tr_split)

    # Match by index and verify answer alignment
    matched = []
    mismatches = 0
    min_len = min(len(ds_en), len(ds_tr))

    for idx in range(min_len):
        en = ds_en[idx]
        tr = ds_tr[idx]

        answer_en = extract_answer_number(en["answer"])
        answer_tr = extract_answer_number(tr["answer"])

        if answer_en is None or answer_tr is None:
            continue
        if answer_en != answer_tr:
            mismatches += 1
            continue

        n_steps = count_reasoning_steps(en["answer"])
        matched.append({
            "idx": idx,
            "question_en": en["question"],
            "question_tr": tr["question"],
            "solution_en": en["answer"],
            "solution_tr": tr["answer"],
            "answer_number": answer_en,
            "n_steps": n_steps,
        })

    logger.info(
        "Matched %d problems (%d mismatches discarded)", len(matched), mismatches
    )

    # Stratified sampling by difficulty
    rng = np.random.RandomState(seed)
    bins = {"easy": [], "medium": [], "hard": [], "very_hard": []}
    for p in matched:
        if p["n_steps"] <= 2:
            bins["easy"].append(p)
        elif p["n_steps"] <= 4:
            bins["medium"].append(p)
        elif p["n_steps"] <= 6:
            bins["hard"].append(p)
        else:
            bins["very_hard"].append(p)

    per_bin = max(n_problems // len(bins), 1)
    selected = []
    for name, problems in bins.items():
        if not problems:
            continue
        k = min(per_bin, len(problems))
        indices = rng.choice(len(problems), k, replace=False)
        selected.extend([problems[i] for i in indices])
        logger.info("  %s: selected %d/%d", name, k, len(problems))

    # Fill remaining if needed
    remaining = n_problems - len(selected)
    if remaining > 0:
        selected_idxs = {p["idx"] for p in selected}
        pool = [p for p in matched if p["idx"] not in selected_idxs]
        extra_indices = rng.choice(len(pool), min(remaining, len(pool)), replace=False)
        selected.extend([pool[i] for i in extra_indices])

    selected = selected[:n_problems]
    logger.info("Final selection: %d problems", len(selected))
    return selected


def construct_prompt(question: str, style: str = "direct") -> str:
    """
    Construct a prompt for the model.

    Styles:
        direct: For logit lens analysis (model must produce answer immediately)
        cot_en: English chain-of-thought
        cot_tr: Turkish chain-of-thought
    """
    if style == "direct":
        return f"Q: {question}\nA: The answer is"
    elif style == "direct_tr":
        return f"Soru: {question}\nCevap:"
    elif style == "cot_en":
        return f"Q: {question}\nA: Let's solve this step by step.\n"
    elif style == "cot_tr":
        return f"Soru: {question}\nCevap: Bunu adım adım çözelim.\n"
    else:
        raise ValueError(f"Unknown prompt style: {style}")


def compute_token_fertility(tokenizer, text_en: str, text_tr: str) -> dict:
    """Compute tokenization fertility ratio between English and Turkish texts."""
    tokens_en = tokenizer.encode(text_en, add_special_tokens=False)
    tokens_tr = tokenizer.encode(text_tr, add_special_tokens=False)
    return {
        "n_tokens_en": len(tokens_en),
        "n_tokens_tr": len(tokens_tr),
        "fertility_ratio": len(tokens_tr) / max(len(tokens_en), 1),
        "n_words_en": len(text_en.split()),
        "n_words_tr": len(text_tr.split()),
        "tokens_per_word_en": len(tokens_en) / max(len(text_en.split()), 1),
        "tokens_per_word_tr": len(tokens_tr) / max(len(text_tr.split()), 1),
    }
