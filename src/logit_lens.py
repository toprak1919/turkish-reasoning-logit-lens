import logging

import torch
import torch.nn.functional as F
import numpy as np

from .model import LlamaWrapper

logger = logging.getLogger(__name__)


def compute_logit_lens(
    wrapper: LlamaWrapper,
    prompt: str,
    target_token_ids: list[int],
    top_k: int = 10,
) -> dict:
    """
    Run the logit lens on a prompt: for each layer, project the hidden state
    through the unembedding matrix and track P(target tokens).

    Note: For multi-token answers, only the first token is tracked.
    This is a standard simplification in logit lens studies.
    """
    inputs = wrapper.tokenize(prompt)
    input_ids = inputs["input_ids"].to(wrapper.device)

    all_logits = wrapper.get_logit_lens_all_layers(input_ids, position=-1)
    n_states = all_logits.shape[0]

    # Cast to float32 before softmax — critical for 128K vocabulary.
    # Float16 flushes most probabilities to zero, breaking entropy and JS divergence.
    all_probs = F.softmax(all_logits.float(), dim=-1)  # (n_states, vocab)

    # Track P(target) at each layer — vectorized
    if target_token_ids:
        target_probs = all_probs[:, target_token_ids].cpu().numpy()
        primary_probs = target_probs.max(axis=1)
    else:
        target_probs = np.zeros((n_states, 0))
        primary_probs = np.zeros(n_states)

    # Top-k tokens per layer
    top_tokens = []
    for layer_idx in range(n_states):
        probs = all_probs[layer_idx]
        topk = torch.topk(probs, top_k)
        layer_top = []
        for tid, p in zip(topk.indices.cpu().tolist(), topk.values.cpu().tolist()):
            token_str = wrapper.tokenizer.decode([tid])
            layer_top.append((tid, token_str, p))
        top_tokens.append(layer_top)

    # Entropy at each layer — vectorized (already float32 from softmax above)
    probs_clamped = torch.clamp(all_probs, min=1e-12)
    entropy = -(probs_clamped * torch.log2(probs_clamped)).sum(dim=-1).cpu().numpy()

    # Move to CPU and free GPU memory
    logits_cpu = all_logits.cpu()
    del all_logits, all_probs, probs_clamped
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "probs": primary_probs,
        "all_target_probs": target_probs,
        "top_tokens": top_tokens,
        "entropy": entropy,
        "logits": logits_cpu,
    }


def compute_js_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> np.ndarray:
    """
    Compute Jensen-Shannon divergence between two sets of per-layer logits.
    Uses the formula: JS(p,q) = 0.5 * KL(p||m) + 0.5 * KL(q||m) where m = 0.5*(p+q).
    """
    # Cast to float32 for numerical stability
    probs_a = F.softmax(logits_a.float(), dim=-1)
    probs_b = F.softmax(logits_b.float(), dim=-1)

    m = 0.5 * (probs_a + probs_b)
    # Clamp before log to avoid -inf
    log_m = torch.clamp(m, min=1e-12).log()
    # F.kl_div expects (log_input, target): computes sum(target * (log(target) - log_input))
    # So F.kl_div(log_m, p) = KL(p || m)
    kl_pm = F.kl_div(log_m, probs_a, reduction="none").sum(dim=-1)  # (n_states,)
    kl_qm = F.kl_div(log_m, probs_b, reduction="none").sum(dim=-1)
    # Clamp to [0, ln(2)] — JS divergence is bounded and non-negative
    js_div = torch.clamp(0.5 * (kl_pm + kl_qm), min=0.0).cpu().numpy()

    return js_div


def compute_random_baseline(
    wrapper: LlamaWrapper,
    prompt: str,
    correct_answer: int,
    n_distractors: int = 5,
    top_k: int = 10,
) -> dict:
    """
    Compute P(wrong answer) trajectories as a baseline.
    Tracks P for several wrong numeric answers to show that P(correct)
    is meaningfully higher than P(arbitrary number).
    """
    import random
    rng = random.Random(42)

    # Pick distractor numbers different from the correct answer
    distractors = []
    candidates = list(range(1, 100)) + [200, 300, 500, 1000]
    candidates = [c for c in candidates if c != correct_answer]
    rng.shuffle(candidates)
    distractors = candidates[:n_distractors]

    baseline_probs = []
    for distractor in distractors:
        distractor_ids = wrapper.get_answer_token_ids(distractor)
        result = compute_logit_lens(wrapper, prompt, distractor_ids, top_k=top_k)
        baseline_probs.append(result["probs"])

    return {
        "mean_baseline": np.mean(baseline_probs, axis=0),
        "distractors": distractors,
        "all_baseline_probs": baseline_probs,
    }


def detect_english_pivot(
    wrapper: LlamaWrapper, top_tokens: list[list[tuple]]
) -> list[dict]:
    """
    For each layer, check if the top-1 predicted token is English or Turkish.
    Filters out numeric tokens and punctuation for accuracy.

    Returns:
        List of dicts per layer with top_token, is_english, and decoded token string.
    """
    results = []
    for layer_idx, layer_top in enumerate(top_tokens):
        if not layer_top:
            results.append({"layer": layer_idx, "top_token": "", "is_english": None})
            continue
        tid, token_str, prob = layer_top[0]
        stripped = token_str.strip()

        # Skip numeric tokens and punctuation — they are language-neutral
        if not stripped or all(c.isdigit() or not c.isalpha() for c in stripped):
            is_english = None  # language-neutral
        else:
            # Heuristic: ASCII alphabetic characters suggest English/Latin script
            alpha_chars = [c for c in stripped if c.isalpha()]
            is_english = all(ord(c) < 128 for c in alpha_chars) if alpha_chars else None

        results.append({
            "layer": layer_idx,
            "top_token": token_str,
            "token_id": tid,
            "prob": prob,
            "is_english": is_english,
        })
    return results
