import numpy as np
from scipy import stats


def compute_divergence_onset(
    probs_en: np.ndarray, probs_tr: np.ndarray, threshold: float = 0.1
) -> int:
    """
    Find the first layer where |P_en(answer) - P_tr(answer)| > threshold.
    Returns -1 if no divergence exceeds threshold.
    """
    diff = np.abs(probs_en - probs_tr)
    indices = np.where(diff > threshold)[0]
    return int(indices[0]) if len(indices) > 0 else -1


def compute_convergence_layer(probs: np.ndarray, threshold: float = 0.5) -> int:
    """Find the first layer where P(answer) > threshold. Returns -1 if never."""
    indices = np.where(probs > threshold)[0]
    return int(indices[0]) if len(indices) > 0 else -1


def categorize_divergence_pattern(
    probs_en: np.ndarray, probs_tr: np.ndarray
) -> str:
    """
    Classify where the EN-TR divergence is concentrated.
    Region boundaries are derived from the number of layers (thirds).
    """
    diff = probs_en - probs_tr  # positive = EN ahead

    if np.max(np.abs(diff)) < 0.05:
        return "no_divergence"

    if np.mean(diff) < -0.05:
        return "tr_advantage"

    max_layer = np.argmax(np.abs(diff))
    n_layers = len(diff)
    early_end = n_layers // 3
    mid_end = 2 * n_layers // 3

    if max_layer <= early_end:
        return "early_divergence"
    elif max_layer <= mid_end:
        return "mid_divergence"
    else:
        return "late_divergence"


def _derive_regions(n_states: int) -> dict:
    """Derive early/mid/late layer region boundaries from the number of states."""
    third = n_states // 3
    return {
        "early": [0, third - 1],
        "mid": [third, 2 * third - 1],
        "late": [2 * third, n_states - 1],
    }


def compute_layer_region_stats(
    all_probs_en: list[np.ndarray],
    all_probs_tr: list[np.ndarray],
    regions: dict = None,
) -> dict:
    """
    Compute divergence statistics for early/mid/late layer regions.
    Applies Holm-Bonferroni correction for multiple comparisons.
    """
    if regions is None:
        n_states = len(all_probs_en[0])
        regions = _derive_regions(n_states)

    raw_results = {}
    for region_name, (start, end) in regions.items():
        en_means = []
        tr_means = []
        for probs_en, probs_tr in zip(all_probs_en, all_probs_tr):
            en_means.append(np.mean(probs_en[start:end + 1]))
            tr_means.append(np.mean(probs_tr[start:end + 1]))

        en_arr = np.array(en_means)
        tr_arr = np.array(tr_means)
        diff = en_arr - tr_arr

        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1) if len(diff) > 1 else 1e-10
        n = len(diff)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        # Hedges' correction for small-sample bias (canonical formula)
        hedges_g = cohens_d * (1 - 3 / (4 * (n - 1) - 1)) if n > 2 else cohens_d

        p_value = paired_permutation_test(en_arr, tr_arr)

        raw_results[region_name] = {
            "mean_en": float(np.mean(en_arr)),
            "mean_tr": float(np.mean(tr_arr)),
            "mean_divergence": float(mean_diff),
            "cohens_d": float(cohens_d),
            "hedges_g": float(hedges_g),
            "p_value": float(p_value),
        }

    # Holm-Bonferroni correction with monotonicity enforcement
    sorted_regions = sorted(raw_results.keys(), key=lambda r: raw_results[r]["p_value"])
    n_tests = len(sorted_regions)
    prev_adjusted = 0.0
    for rank, region_name in enumerate(sorted_regions):
        raw_p = raw_results[region_name]["p_value"]
        adjusted_p = min(raw_p * (n_tests - rank), 1.0)
        # Enforce monotonicity: each adjusted p must be >= the previous
        adjusted_p = max(adjusted_p, prev_adjusted)
        prev_adjusted = adjusted_p
        raw_results[region_name]["p_value_corrected"] = float(adjusted_p)

    return raw_results


def paired_permutation_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    seed: int = 42,
) -> float:
    """
    Exact paired permutation test (sign-flip).
    For n <= 20, enumerates all 2^n permutations. Otherwise uses Monte Carlo.
    """
    diffs = values_a - values_b
    observed_diff = np.abs(np.mean(diffs))
    n = len(diffs)

    if n <= 20:
        # Exact test: enumerate all 2^n sign flips
        n_perms = 2 ** n
        count = 0
        for i in range(n_perms):
            signs = np.array([(1 if (i >> bit) & 1 else -1) for bit in range(n)])
            if np.abs(np.mean(signs * diffs)) >= observed_diff:
                count += 1
        return count / n_perms
    else:
        # Vectorized Monte Carlo
        rng = np.random.RandomState(seed)
        n_permutations = 100000
        signs = rng.choice([-1, 1], size=(n_permutations, n))
        permuted_diffs = np.abs(np.mean(signs * diffs[None, :], axis=1))
        count = np.sum(permuted_diffs >= observed_diff)
        return float(count / n_permutations)


def aggregate_trajectories(
    all_probs: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean and 95% CI across problems using t-distribution.
    Returns: mean, ci_lower, ci_upper — each shape (n_layers+1,)
    """
    stacked = np.stack(all_probs)  # (n_problems, n_layers+1)
    n = stacked.shape[0]
    mean = stacked.mean(axis=0)
    se = stacked.std(axis=0, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)  # two-tailed 95% CI
    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se
    return mean, ci_lower, ci_upper


def compute_behavioral_cis(n_correct: int, n_total: int) -> dict:
    """Compute Wilson score confidence interval for a proportion."""
    if n_total == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    p = n_correct / n_total
    z = 1.96
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denom
    return {
        "mean": float(p),
        "ci_lower": float(max(0, center - margin)),
        "ci_upper": float(min(1, center + margin)),
    }


def fishers_exact_test(correct_en: list, correct_tr: list) -> float:
    """Fisher's exact test on 2x2 contingency (language x correct)."""
    a = sum(1 for c in correct_en if c)   # EN correct
    b = sum(1 for c in correct_en if not c)  # EN wrong
    c = sum(1 for c in correct_tr if c)   # TR correct
    d = sum(1 for c in correct_tr if not c)  # TR wrong
    _, p_value = stats.fisher_exact([[a, b], [c, d]])
    return float(p_value)
