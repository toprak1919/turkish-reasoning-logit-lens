import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# ── Style ──────────────────────────────────────────────────────────────
COLOR_EN = "#2563EB"       # Blue
COLOR_TR = "#D55E00"       # Vermillion (colorblind-safe, replaces red)
REGION_COLORS = {"early": "#DBEAFE", "mid": "#FEF3C7", "late": "#FCE7F3"}


def _sanitize_token(s):
    """Remove characters that matplotlib PDF backend can't render (e.g., BOM U+FEFF)."""
    cleaned = "".join(c for c in s.strip() if len(c) == 1 and 32 <= ord(c) < 65279 and c.isprintable())
    return cleaned or "<?>"


def _derive_region_labels(n_states):
    """Derive region labels and boundaries from the number of states."""
    third = n_states // 3
    return {
        "early": {"range": (0, third - 1), "label": f"Early\n(0\u2013{third - 1})"},
        "mid": {"range": (third, 2 * third - 1), "label": f"Mid\n({third}\u2013{2 * third - 1})"},
        "late": {"range": (2 * third, n_states - 1), "label": f"Late\n({2 * third}\u2013{n_states - 1})"},
    }


def _add_region_shading(ax, n_states, alpha=0.15):
    """Add early/mid/late region background shading to an axis."""
    regions = _derive_region_labels(n_states)
    for name, info in regions.items():
        start, end = info["range"]
        ax.axvspan(start - 0.5, end + 0.5, alpha=alpha, color=REGION_COLORS[name])


def set_publication_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save_and_close(fig, save_path):
    """Save figure and close to prevent memory leaks."""
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


# ── Figure 1: Behavioral accuracy ─────────────────────────────────────
def plot_accuracy_comparison(
    acc_en: float,
    acc_tr: float,
    acc_by_difficulty: dict = None,
    save_path: str = None,
    ci_en: dict = None,
    ci_tr: dict = None,
    fisher_p: float = None,
    n_total: int = None,
    difficulty_ns: dict = None,
):
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 2]})

    # (a) Overall with error bars
    ax = axes[0]
    vals = [acc_en, acc_tr]
    yerr = None
    if ci_en and ci_tr:
        yerr = [[acc_en - ci_en["ci_lower"] * 100, acc_tr - ci_tr["ci_lower"] * 100],
                [ci_en["ci_upper"] * 100 - acc_en, ci_tr["ci_upper"] * 100 - acc_tr]]
    bars = ax.bar(["English", "Turkish"], vals,
                  color=[COLOR_EN, COLOR_TR], width=0.5,
                  hatch=["", "///"], edgecolor="white",
                  yerr=yerr, capsize=5, error_kw={"lw": 1.5})
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) Overall Accuracy")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 6,
                f"{val:.1f}%", ha="center", fontsize=10)
    # Annotate Fisher's p and sample size
    annot_parts = []
    if fisher_p is not None:
        annot_parts.append(f"Fisher's p = {fisher_p:.3f}")
    if n_total is not None:
        annot_parts.append(f"n = {n_total}")
    if annot_parts:
        ax.text(0.5, 0.92, ", ".join(annot_parts), ha="center", va="top",
                transform=ax.transAxes, fontsize=9, color="gray")

    # (b) By difficulty
    if acc_by_difficulty:
        ax = axes[1]
        categories = list(acc_by_difficulty.keys())
        en_vals = [acc_by_difficulty[c]["en"] for c in categories]
        tr_vals = [acc_by_difficulty[c]["tr"] for c in categories]
        x = np.arange(len(categories))
        w = 0.35
        ax.bar(x - w / 2, en_vals, w, label="English", color=COLOR_EN, edgecolor="white")
        ax.bar(x + w / 2, tr_vals, w, label="Turkish", color=COLOR_TR,
               hatch="///", edgecolor="white")
        ax.set_xlabel("Difficulty (reasoning steps)")
        ax.set_title("(b) Accuracy by Difficulty")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        # Annotate sample sizes per bin
        if difficulty_ns:
            for i, cat in enumerate(categories):
                n = difficulty_ns.get(cat, "")
                if n:
                    ax.text(i, -8, f"n={n}", ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    _save_and_close(fig, save_path)
    return fig


# ── Figure 2: Aggregate reasoning trajectory (HERO FIGURE) ────────────
def plot_aggregate_trajectory(
    mean_en: np.ndarray,
    ci_lower_en: np.ndarray,
    ci_upper_en: np.ndarray,
    mean_tr: np.ndarray,
    ci_lower_tr: np.ndarray,
    ci_upper_tr: np.ndarray,
    save_path: str = None,
    divergence_onset: int = None,
):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    n_states = len(mean_en)
    layers = np.arange(n_states)

    # Layer region backgrounds
    _add_region_shading(ax, n_states)

    # CI bands
    ax.fill_between(layers, ci_lower_en, ci_upper_en, alpha=0.2, color=COLOR_EN)
    ax.fill_between(layers, ci_lower_tr, ci_upper_tr, alpha=0.2, color=COLOR_TR)

    # Mean lines
    ax.plot(layers, mean_en, color=COLOR_EN, linewidth=2.5, label="English", marker="o", markersize=3)
    ax.plot(layers, mean_tr, color=COLOR_TR, linewidth=2.5, label="Turkish", marker="s", markersize=3)

    # Divergence onset annotation
    if divergence_onset is not None:
        ax.axvline(divergence_onset, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.annotate(f"Divergence onset\n(layer {divergence_onset})",
                    xy=(divergence_onset, ax.get_ylim()[1] * 0.5 if ax.get_ylim()[1] > 0 else 0.5),
                    xytext=(divergence_onset - 6, ax.get_ylim()[1] * 0.7 if ax.get_ylim()[1] > 0 else 0.7),
                    fontsize=9, color="gray",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))

    ax.set_xlabel("Layer")
    ax.set_ylabel("P(correct answer token)")
    ax.set_title(f"Aggregate Reasoning Trajectory (n={len(mean_en)} layers)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(0, n_states - 1)
    ax.set_ylim(bottom=0)

    # Region labels at top (AFTER plotting so ylim is correct)
    regions = _derive_region_labels(n_states)
    ymax = ax.get_ylim()[1]
    for name, info in regions.items():
        start, end = info["range"]
        mid = (start + end) / 2
        ax.text(mid, ymax * 0.97, info["label"], ha="center", va="top",
                fontsize=8, color="gray", style="italic")

    plt.tight_layout()
    _save_and_close(fig, save_path)
    return fig


# ── Figure 3: Logit lens heatmap ──────────────────────────────────────
def plot_logit_lens_heatmap(
    top_tokens: list[list[tuple]],
    probs: np.ndarray,
    title: str,
    save_path: str = None,
    top_k: int = 5,
):
    set_publication_style()
    n_layers = len(top_tokens)
    fig, ax = plt.subplots(figsize=(10, max(8, n_layers * 0.3)))

    display_data = np.zeros((n_layers, top_k))
    annotations = []
    for layer_idx, layer_top in enumerate(top_tokens):
        row_annot = []
        for k in range(min(top_k, len(layer_top))):
            tid, token_str, prob = layer_top[k]
            display_data[layer_idx, k] = prob
            row_annot.append(f"{_sanitize_token(token_str)}\n{prob:.3f}")
        annotations.append(row_annot)

    sns.heatmap(
        display_data,
        ax=ax,
        cmap="viridis",
        annot=np.array([[a for a in row] + [""] * (top_k - len(row)) for row in annotations]),
        fmt="",
        annot_kws={"fontsize": 7},
        xticklabels=[f"Top-{k+1}" for k in range(top_k)],
        yticklabels=[f"L{i}" for i in range(n_layers)],
        cbar_kws={"label": "Probability"},
    )
    ax.set_title(title)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Layer")

    plt.tight_layout()
    _save_and_close(fig, save_path)
    return fig


# ── Figure 4: Divergence onset histogram ──────────────────────────────
def plot_divergence_onset_histogram(
    onset_layers: list[int],
    save_path: str = None,
    n_states: int = 33,
):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    valid = [l for l in onset_layers if l >= 0]
    if not valid:
        ax.text(0.5, 0.5, "No divergence detected", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        _save_and_close(fig, save_path)
        return fig

    _add_region_shading(ax, n_states)

    bins = np.arange(-0.5, n_states + 0.5, 1)
    ax.hist(valid, bins=bins, color=COLOR_TR, edgecolor="white", alpha=0.8)
    ax.axvline(np.median(valid), color="black", linestyle="--", linewidth=2,
               label=f"Median: layer {np.median(valid):.0f}")

    ax.set_xlabel("Layer of Divergence Onset")
    ax.set_ylabel("Number of Problems")
    ax.set_title("Where Does Turkish Reasoning First Diverge from English?")
    ax.legend()

    plt.tight_layout()
    _save_and_close(fig, save_path)
    return fig


# ── Figure 5: Token fertility vs P(answer) gap ───────────────────────
def plot_fertility_vs_drop(
    fertility_ratios: list[float],
    correct_en: list,
    correct_tr: list,
    save_path: str = None,
    prob_gaps: list[float] = None,
):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    if prob_gaps is not None:
        from scipy import stats as sp_stats
        # Continuous P(answer) gap — much more informative than binary
        prob_gaps = [float(x) for x in prob_gaps]  # ensure float64 for linalg
        fertility_ratios = [float(x) for x in fertility_ratios]
        ax.scatter(fertility_ratios, prob_gaps, alpha=0.7, s=80, c=COLOR_TR, edgecolors="white")
        if len(set(prob_gaps)) > 1:
            z = np.polyfit(fertility_ratios, prob_gaps, 1)
            p = np.poly1d(z)
            x_arr = np.array(fertility_ratios)
            y_arr = np.array(prob_gaps)
            x_line = np.linspace(min(fertility_ratios), max(fertility_ratios), 100)
            # Compute R-squared and p-value
            r_val, p_val = sp_stats.pearsonr(x_arr, y_arr)
            r_squared = r_val ** 2
            # Confidence band
            n = len(x_arr)
            y_pred = p(x_arr)
            se_resid = np.sqrt(np.sum((y_arr - y_pred) ** 2) / (n - 2))
            x_mean = np.mean(x_arr)
            ss_x = np.sum((x_arr - x_mean) ** 2)
            t_crit = sp_stats.t.ppf(0.975, df=n - 2)
            y_line = p(x_line)
            se_line = se_resid * np.sqrt(1 / n + (x_line - x_mean) ** 2 / ss_x)
            ax.fill_between(x_line, y_line - t_crit * se_line, y_line + t_crit * se_line,
                            alpha=0.15, color="gray")
            ax.plot(x_line, y_line, "--", color="black", alpha=0.7,
                    label=f"r={r_val:.2f}, R²={r_squared:.2f}, p={p_val:.3f}")
            ax.legend()
        ax.set_ylabel("P(answer) Gap (EN \u2212 TR, final layer)")
    else:
        # Fallback: binary accuracy drop with jitter
        drops = [int(en and not tr) for en, tr in zip(correct_en, correct_tr)]
        jitter = np.random.RandomState(42).uniform(-0.05, 0.05, len(drops))
        ax.scatter(fertility_ratios, np.array(drops) + jitter, alpha=0.6, s=80,
                   c=COLOR_TR, edgecolors="white")
        ax.set_ylabel("Accuracy Drop (1 = EN correct, TR wrong)")
        ax.set_yticks([0, 1])

    ax.set_xlabel("Token Fertility Ratio (TR/EN)")
    ax.set_title("Token Fertility vs. P(answer) Gap")

    plt.tight_layout()
    _save_and_close(fig, save_path)
    return fig


# ── Figure 6: Entropy trajectories ────────────────────────────────────
def plot_entropy_trajectories(
    mean_entropy_en: np.ndarray,
    mean_entropy_tr: np.ndarray,
    save_path: str = None,
    all_entropy_en: list[np.ndarray] = None,
    all_entropy_tr: list[np.ndarray] = None,
):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    n_states = len(mean_entropy_en)
    layers = np.arange(n_states)

    _add_region_shading(ax, n_states, alpha=0.1)

    # CI bands if raw data provided
    if all_entropy_en is not None and all_entropy_tr is not None:
        stacked_en = np.stack(all_entropy_en)
        stacked_tr = np.stack(all_entropy_tr)
        se_en = stacked_en.std(axis=0, ddof=1) / np.sqrt(len(all_entropy_en))
        se_tr = stacked_tr.std(axis=0, ddof=1) / np.sqrt(len(all_entropy_tr))
        ax.fill_between(layers, mean_entropy_en - 1.96 * se_en,
                        mean_entropy_en + 1.96 * se_en, alpha=0.15, color=COLOR_EN)
        ax.fill_between(layers, mean_entropy_tr - 1.96 * se_tr,
                        mean_entropy_tr + 1.96 * se_tr, alpha=0.15, color=COLOR_TR)

    ax.plot(layers, mean_entropy_en, color=COLOR_EN, linewidth=2, label="English", marker="o", markersize=3)
    ax.plot(layers, mean_entropy_tr, color=COLOR_TR, linewidth=2, label="Turkish", marker="s", markersize=3)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Model Uncertainty Across Layers")
    ax.legend()

    plt.tight_layout()
    _save_and_close(fig, save_path)
    return fig


# ── Figure 7: Case study two-panel ────────────────────────────────────
def plot_case_study(
    problem: dict,
    probs_en: np.ndarray,
    probs_tr: np.ndarray,
    top_tokens_en: list[list[tuple]],
    top_tokens_tr: list[list[tuple]],
    save_path: str = None,
):
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    n_states = len(probs_en)
    layers = np.arange(n_states)

    for panel_idx, (ax, probs, top_tokens, color, lang) in enumerate([
        (axes[0], probs_en, top_tokens_en, COLOR_EN, "English"),
        (axes[1], probs_tr, top_tokens_tr, COLOR_TR, "Turkish"),
    ]):
        _add_region_shading(ax, n_states, alpha=0.1)
        ax.plot(layers, probs, color=color, linewidth=2.5, marker="o", markersize=4)
        ax.set_xlabel("Layer")
        panel_label = "(a)" if panel_idx == 0 else "(b)"
        ax.set_title(f"{panel_label} {lang}: P(correct answer = {problem['answer_number']})")

        # Annotate top-1 tokens at selected layers
        for l in range(0, len(layers), 4):
            if top_tokens[l]:
                token_str = _sanitize_token(top_tokens[l][0][1])
                if len(token_str) > 8:
                    token_str = token_str[:8] + "..."
                ax.annotate(
                    token_str, (l, probs[l]),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=7, ha="center", color="gray",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                )

        ax.set_ylim(bottom=0)

    axes[0].set_ylabel("P(correct answer token)")

    fig.suptitle(
        f"Case Study \u2014 Problem #{problem['idx']} "
        f"(Answer: {problem['answer_number']}, Steps: {problem['n_steps']})",
        fontsize=14,
    )
    plt.tight_layout()
    _save_and_close(fig, save_path)
    return fig


# ── Figure 8: Small multiples grid ────────────────────────────────────
def plot_small_multiples(
    all_probs_en: list[np.ndarray],
    all_probs_tr: list[np.ndarray],
    problems: list[dict],
    save_path: str = None,
    ncols: int = 5,
):
    set_publication_style()
    n = len(problems)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2), sharex=True, sharey=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (probs_en, probs_tr, problem) in enumerate(zip(all_probs_en, all_probs_tr, problems)):
        ax = axes_flat[i]
        layers = np.arange(len(probs_en))
        ax.plot(layers, probs_en, color=COLOR_EN, linewidth=1, alpha=0.8)
        ax.plot(layers, probs_tr, color=COLOR_TR, linewidth=1, alpha=0.8)
        ax.set_title(f"#{problem['idx']} ({problem['n_steps']}s)", fontsize=7)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=6)

    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    en_patch = mpatches.Patch(color=COLOR_EN, label="English")
    tr_patch = mpatches.Patch(color=COLOR_TR, label="Turkish")
    fig.legend(handles=[en_patch, tr_patch], loc="upper right", fontsize=9)

    fig.suptitle("Per-Problem Reasoning Trajectories", fontsize=13)
    fig.supxlabel("Layer", fontsize=11)
    fig.supylabel("P(correct answer)", fontsize=11)
    plt.tight_layout()
    _save_and_close(fig, save_path)
    return fig


# ── Prompt Frame Control Figure ──────────────────────────────────────
def plot_prompt_frame_control(
    mean_en: np.ndarray,
    ci_lo_en: np.ndarray,
    ci_hi_en: np.ndarray,
    mean_tr_en_frame: np.ndarray,
    ci_lo_tr_en: np.ndarray,
    ci_hi_tr_en: np.ndarray,
    mean_tr_tr_frame: np.ndarray,
    ci_lo_tr_tr: np.ndarray,
    ci_hi_tr_tr: np.ndarray,
    save_path: str = None,
):
    """Three-condition comparison: EN, TR (English frame), TR (Turkish frame)."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    n_states = len(mean_en)
    layers = np.arange(n_states)

    COLOR_TR_FRAME = "#009E73"  # Teal (colorblind-safe, Okabe-Ito)

    # CI bands
    ax.fill_between(layers, ci_lo_en, ci_hi_en, alpha=0.15, color=COLOR_EN)
    ax.fill_between(layers, ci_lo_tr_en, ci_hi_tr_en, alpha=0.15, color=COLOR_TR)
    ax.fill_between(layers, ci_lo_tr_tr, ci_hi_tr_tr, alpha=0.15, color=COLOR_TR_FRAME)

    # Lines
    ax.plot(layers, mean_en, color=COLOR_EN, linewidth=2.5,
            label=f"EN (English frame) \u2014 final P={mean_en[-1]:.2f}")
    ax.plot(layers, mean_tr_en_frame, color=COLOR_TR, linewidth=2.5,
            label=f"TR (English frame) \u2014 final P={mean_tr_en_frame[-1]:.2f}")
    ax.plot(layers, mean_tr_tr_frame, color=COLOR_TR_FRAME, linewidth=2.5, linestyle="--",
            label=f"TR (Turkish frame) \u2014 final P={mean_tr_tr_frame[-1]:.2f}")

    ax.set_xlabel("Layer")
    ax.set_ylabel("P(correct answer token)")
    ax.set_title("Effect of Prompt Frame on P(answer)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.set_xlim(0, n_states - 1)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    _save_and_close(fig, save_path)
    return fig


# ── Figure 9 (NEW): JS Divergence Heatmap ─────────────────────────────
def plot_js_divergence_heatmap(
    all_js_div: list[np.ndarray],
    problems: list[dict],
    save_path: str = None,
):
    """Heatmap showing JS divergence across layers and problems."""
    set_publication_style()
    data = np.stack(all_js_div)  # (n_problems, n_states)
    n_problems, n_states = data.shape

    fig, ax = plt.subplots(figsize=(12, max(4, n_problems * 0.35)))
    sns.heatmap(
        data,
        ax=ax,
        cmap="magma",
        xticklabels=[str(i) if i % 5 == 0 else "" for i in range(n_states)],
        yticklabels=[f"#{p['idx']}" for p in problems],
        cbar_kws={"label": "JS Divergence"},
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Problem")
    ax.set_title("Jensen-Shannon Divergence Between English and Turkish (per layer, per problem)")

    # Mark region boundaries
    regions = _derive_region_labels(n_states)
    for name, info in regions.items():
        _, end = info["range"]
        if end < n_states - 1:
            ax.axvline(end + 0.5, color="white", linewidth=1, linestyle="--", alpha=0.5)

    plt.tight_layout()
    _save_and_close(fig, save_path)
    return fig
