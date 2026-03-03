"""
Runner script for Google Colab execution via colab-cli.
Executes the full pipeline: install deps → behavioral baseline → logit lens → visualizations → bonus.
"""
import subprocess
import sys
import os

# Set non-interactive backend before any matplotlib import (including transitive)
import matplotlib
matplotlib.use("Agg")

# Install dependencies
print("=== Installing dependencies ===")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "torch", "transformers", "accelerate", "datasets", "huggingface-hub",
    "numpy", "scipy", "pandas", "matplotlib", "seaborn", "pyyaml",
    "python-dotenv", "tqdm", "einops"])

# HF auth
from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=False)

# Verify GPU
import torch
print(f"\nGPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    print(f"VRAM: {mem / 1e9:.1f} GB")

# Set HF token from environment
hf_token = os.environ.get("HF_TOKEN", "")
if not hf_token:
    print("\n⚠️  HF_TOKEN not set! Set it with: export HF_TOKEN=your_token")
    print("You need access to NousResearch/Meta-Llama-3-8B on HuggingFace.")
    sys.exit(1)

# Add src to path
sys.path.insert(0, ".")

from src.utils import set_seed, setup_logging, ensure_dir
set_seed(42)
logger = setup_logging()
ensure_dir("results/figures")
ensure_dir("results/tables")
ensure_dir("results/cached_activations")

# ═══════════════════════════════════════════════════════════════════
# PHASE 1: Load data
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 1: Loading matched problems")
print("="*60)

from src.data import load_matched_problems, compute_token_fertility, construct_prompt, extract_answer_number
import pandas as pd
import numpy as np

problems = load_matched_problems(n_problems=30, seed=42)
print(f"Loaded {len(problems)} matched EN-TR problems")

# ═══════════════════════════════════════════════════════════════════
# PHASE 2: Behavioral baseline (Instruct model)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 2: Behavioral baseline")
print("="*60)

from src.model import LlamaWrapper
from src.metrics import compute_behavioral_cis, fishers_exact_test
from tqdm import tqdm

wrapper_instruct = LlamaWrapper(
    model_name="NousResearch/Meta-Llama-3-8B-Instruct",
    dtype="float16",
)

# Tokenization analysis
fertility_data = []
for p in problems:
    f = compute_token_fertility(wrapper_instruct.tokenizer, p["question_en"], p["question_tr"])
    f["idx"] = p["idx"]
    fertility_data.append(f)

df_fertility = pd.DataFrame(fertility_data)
df_fertility.to_csv("results/tables/fertility_analysis.csv", index=False)
print(f"Mean fertility ratio: {df_fertility['fertility_ratio'].mean():.2f}x")

# Run behavioral eval
results = []
for p in tqdm(problems, desc="Behavioral eval"):
    prompt_en = construct_prompt(p["question_en"], style="cot_en")
    output_en = wrapper_instruct.generate(prompt_en, max_new_tokens=512)
    pred_en = extract_answer_number(output_en)

    prompt_tr = construct_prompt(p["question_tr"], style="cot_tr")
    output_tr = wrapper_instruct.generate(prompt_tr, max_new_tokens=512)
    pred_tr = extract_answer_number(output_tr)

    results.append({
        "idx": p["idx"], "answer": p["answer_number"], "n_steps": p["n_steps"],
        "pred_en": pred_en, "pred_tr": pred_tr,
        "correct_en": pred_en == p["answer_number"],
        "correct_tr": pred_tr == p["answer_number"],
    })

df_results = pd.DataFrame(results)
df_results.to_csv("results/tables/behavioral_results.csv", index=False)

acc_en = df_results["correct_en"].mean() * 100
acc_tr = df_results["correct_tr"].mean() * 100
n_en = int(df_results["correct_en"].sum())
n_tr = int(df_results["correct_tr"].sum())
n_total = len(df_results)

# Confidence intervals and Fisher's exact test
ci_en = compute_behavioral_cis(n_en, n_total)
ci_tr = compute_behavioral_cis(n_tr, n_total)
fisher_p = fishers_exact_test(df_results["correct_en"].tolist(), df_results["correct_tr"].tolist())

print(f"\nEnglish accuracy: {acc_en:.1f}% [95% CI: {ci_en['ci_lower']*100:.1f}–{ci_en['ci_upper']*100:.1f}]")
print(f"Turkish accuracy: {acc_tr:.1f}% [95% CI: {ci_tr['ci_lower']*100:.1f}–{ci_tr['ci_upper']*100:.1f}]")
print(f"Gap: {acc_en - acc_tr:.1f}pp")
print(f"Fisher's exact test: p = {fisher_p:.4f}")

# Free Instruct model memory
del wrapper_instruct
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════
# PHASE 3: Logit lens analysis (base model)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 3: Logit lens analysis")
print("="*60)

from src.logit_lens import compute_logit_lens, compute_js_divergence, compute_random_baseline
from src.metrics import (
    compute_divergence_onset, categorize_divergence_pattern,
    compute_layer_region_stats, aggregate_trajectories,
)
from src.utils import cache_results

wrapper = LlamaWrapper(
    model_name="NousResearch/Meta-Llama-3-8B",
    dtype="float16",
)

# Sanity check
print("\nSanity check — logit lens on 'The capital of France is':")
test_result = compute_logit_lens(wrapper, "The capital of France is", [], top_k=3)
for i in [0, 10, 20, 30, 32]:
    if i < len(test_result["top_tokens"]) and test_result["top_tokens"][i]:
        t = test_result["top_tokens"][i][0]
        print(f"  Layer {i:2d}: '{t[1].strip()}' (p={t[2]:.3f})")

# Run logit lens on all problems
all_results_en = []
all_results_tr = []

for i, p in enumerate(tqdm(problems, desc="Logit lens")):
    target_ids = wrapper.get_answer_token_ids(p["answer_number"])

    prompt_en = construct_prompt(p["question_en"], style="direct")
    result_en = compute_logit_lens(wrapper, prompt_en, target_ids, top_k=10)
    result_en["prompt"] = prompt_en
    all_results_en.append(result_en)

    prompt_tr = construct_prompt(p["question_tr"], style="direct")
    result_tr = compute_logit_lens(wrapper, prompt_tr, target_ids, top_k=10)
    result_tr["prompt"] = prompt_tr
    all_results_tr.append(result_tr)

    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{len(problems)}] EN={result_en['probs'][-1]:.4f} TR={result_tr['probs'][-1]:.4f}")

# Random baseline (credibility check) — run on ALL problems
print("\nComputing random baseline (P(wrong answer))...")
baseline_results = []
for i, p in enumerate(tqdm(problems, desc="Random baseline")):
    prompt_en = construct_prompt(p["question_en"], style="direct")
    bl = compute_random_baseline(wrapper, prompt_en, p["answer_number"], n_distractors=5)
    baseline_results.append(bl)

if baseline_results:
    mean_baseline = np.mean([bl["mean_baseline"][-1] for bl in baseline_results])
    print(f"  Mean P(wrong answer) at final layer: {mean_baseline:.4f}")
    print(f"  vs Mean P(correct answer) at final layer: {np.mean([r['probs'][-1] for r in all_results_en]):.4f}")

# Turkish-framed prompt control
print("\nTurkish-framed prompt control (Soru: ... Cevap:)...")
all_results_tr_framed = []
for i, p in enumerate(tqdm(problems, desc="TR-framed logit lens")):
    target_ids = wrapper.get_answer_token_ids(p["answer_number"])
    prompt_tr_framed = construct_prompt(p["question_tr"], style="direct_tr")
    result_tr_framed = compute_logit_lens(wrapper, prompt_tr_framed, target_ids, top_k=10)
    result_tr_framed["prompt"] = prompt_tr_framed
    all_results_tr_framed.append(result_tr_framed)

all_probs_tr_framed = [r["probs"] for r in all_results_tr_framed]
mean_tr_framed = np.mean(all_probs_tr_framed, axis=0)
print(f"  TR (English frame) final P(answer): {np.mean([r['probs'][-1] for r in all_results_tr]):.4f}")
print(f"  TR (Turkish frame) final P(answer): {mean_tr_framed[-1]:.4f}")

# Analysis
all_probs_en = [r["probs"] for r in all_results_en]
all_probs_tr = [r["probs"] for r in all_results_tr]

mean_en, ci_lo_en, ci_hi_en = aggregate_trajectories(all_probs_en)
mean_tr, ci_lo_tr, ci_hi_tr = aggregate_trajectories(all_probs_tr)

print(f"\nFinal layer P(answer): EN={mean_en[-1]:.4f} TR={mean_tr[-1]:.4f}")

# Divergence analysis
onset_layers = []
patterns = []
for pe, pt, p in zip(all_probs_en, all_probs_tr, problems):
    onset_layers.append(compute_divergence_onset(pe, pt, threshold=0.1))
    patterns.append(categorize_divergence_pattern(pe, pt))

valid_onsets = [l for l in onset_layers if l >= 0]
median_onset = int(np.median(valid_onsets)) if valid_onsets else None
if valid_onsets:
    print(f"Median divergence onset: layer {np.median(valid_onsets):.0f}")

from collections import Counter
print("\nDivergence patterns:")
for pattern, count in Counter(patterns).most_common():
    print(f"  {pattern}: {count}")

region_stats = compute_layer_region_stats(all_probs_en, all_probs_tr)
print("\nLayer region stats (Holm-Bonferroni corrected):")
for region, stats in region_stats.items():
    print(f"  {region}: divergence={stats['mean_divergence']:.4f} "
          f"g={stats['hedges_g']:.2f} p={stats['p_value_corrected']:.4f}")

# Entropy
all_entropy_en = [r["entropy"] for r in all_results_en]
all_entropy_tr = [r["entropy"] for r in all_results_tr]
mean_entropy_en = np.mean(all_entropy_en, axis=0)
mean_entropy_tr = np.mean(all_entropy_tr, axis=0)

# JS divergence
all_js_div = []
for r_en, r_tr in zip(all_results_en, all_results_tr):
    all_js_div.append(compute_js_divergence(r_en["logits"], r_tr["logits"]))

# P(answer) gaps for fertility scatter
prob_gaps = [float(pe[-1] - pt[-1]) for pe, pt in zip(all_probs_en, all_probs_tr)]

# Cache everything
cache_results({
    "problems": problems,
    "all_probs_en": all_probs_en, "all_probs_tr": all_probs_tr,
    "all_probs_tr_framed": all_probs_tr_framed,
    "mean_en": mean_en, "ci_lo_en": ci_lo_en, "ci_hi_en": ci_hi_en,
    "mean_tr": mean_tr, "ci_lo_tr": ci_lo_tr, "ci_hi_tr": ci_hi_tr,
    "onset_layers": onset_layers, "patterns": patterns,
    "region_stats": region_stats,
    "all_js_div": all_js_div,
    "mean_entropy_en": mean_entropy_en, "mean_entropy_tr": mean_entropy_tr,
    "all_entropy_en": all_entropy_en, "all_entropy_tr": all_entropy_tr,
    "all_results_en": all_results_en, "all_results_tr": all_results_tr,
    "all_results_tr_framed": all_results_tr_framed,
    "prob_gaps": prob_gaps,
    "median_onset": median_onset,
    "baseline_results": baseline_results,
}, "results/cached_activations/analysis_results.pt")
print("\nResults cached.")

# ═══════════════════════════════════════════════════════════════════
# PHASE 4: Generate figures
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 4: Generating figures")
print("="*60)

from src.visualization import (
    plot_accuracy_comparison, plot_aggregate_trajectory,
    plot_logit_lens_heatmap, plot_divergence_onset_histogram,
    plot_fertility_vs_drop, plot_entropy_trajectories,
    plot_case_study, plot_small_multiples, plot_js_divergence_heatmap,
    plot_prompt_frame_control,
)

# Fig 1: Accuracy
difficulty_bins = {"2-step": (1, 2), "3-4 step": (3, 4), "5-6 step": (5, 6), "7+ step": (7, 100)}
acc_by_diff = {}
for label, (lo, hi) in difficulty_bins.items():
    mask = (df_results["n_steps"] >= lo) & (df_results["n_steps"] <= hi)
    subset = df_results[mask]
    if len(subset) > 0:
        acc_by_diff[label] = {"en": subset["correct_en"].mean() * 100, "tr": subset["correct_tr"].mean() * 100}

# Compute per-bin sample sizes for difficulty panel
difficulty_ns = {}
for label, (lo, hi) in difficulty_bins.items():
    mask = (df_results["n_steps"] >= lo) & (df_results["n_steps"] <= hi)
    if mask.sum() > 0:
        difficulty_ns[label] = int(mask.sum())

fig1_kwargs = dict(ci_en=ci_en, ci_tr=ci_tr, fisher_p=fisher_p,
                   n_total=n_total, difficulty_ns=difficulty_ns)
plot_accuracy_comparison(acc_en, acc_tr, acc_by_diff, "results/figures/fig1_accuracy_comparison.pdf", **fig1_kwargs)
plot_accuracy_comparison(acc_en, acc_tr, acc_by_diff, "results/figures/fig1_accuracy_comparison.png", **fig1_kwargs)
print("  fig1_accuracy_comparison ✓")

# Fig 2: Hero figure — with divergence onset annotation
plot_aggregate_trajectory(mean_en, ci_lo_en, ci_hi_en, mean_tr, ci_lo_tr, ci_hi_tr,
                          "results/figures/fig2_aggregate_trajectory.pdf",
                          divergence_onset=median_onset)
plot_aggregate_trajectory(mean_en, ci_lo_en, ci_hi_en, mean_tr, ci_lo_tr, ci_hi_tr,
                          "results/figures/fig2_aggregate_trajectory.png",
                          divergence_onset=median_onset)
print("  fig2_aggregate_trajectory ✓")

# Fig 3: Heatmaps for most dramatic divergence
divergences = [np.max(np.abs(pe - pt)) for pe, pt in zip(all_probs_en, all_probs_tr)]
best_idx = np.argmax(divergences)
bp = problems[best_idx]

plot_logit_lens_heatmap(all_results_en[best_idx]["top_tokens"], all_probs_en[best_idx],
                        f"English — Problem #{bp['idx']}", "results/figures/fig3a_heatmap_en.pdf")
plot_logit_lens_heatmap(all_results_tr[best_idx]["top_tokens"], all_probs_tr[best_idx],
                        f"Turkish — Problem #{bp['idx']}", "results/figures/fig3b_heatmap_tr.pdf")
print("  fig3_heatmaps ✓")

# Fig 4: Divergence onset
plot_divergence_onset_histogram(onset_layers, "results/figures/fig4_divergence_onset.pdf")
plot_divergence_onset_histogram(onset_layers, "results/figures/fig4_divergence_onset.png")
print("  fig4_divergence_onset ✓")

# Fig 5: Fertility vs P(answer) gap (continuous, not binary)
plot_fertility_vs_drop(
    df_fertility["fertility_ratio"].tolist(),
    df_results["correct_en"].tolist(),
    df_results["correct_tr"].tolist(),
    "results/figures/fig5_fertility_vs_drop.pdf",
    prob_gaps=prob_gaps,
)
plot_fertility_vs_drop(
    df_fertility["fertility_ratio"].tolist(),
    df_results["correct_en"].tolist(),
    df_results["correct_tr"].tolist(),
    "results/figures/fig5_fertility_vs_drop.png",
    prob_gaps=prob_gaps,
)
print("  fig5_fertility_vs_drop ✓")

# Fig 6: Entropy with CI bands
plot_entropy_trajectories(mean_entropy_en, mean_entropy_tr,
                          "results/figures/fig6_entropy_trajectories.pdf",
                          all_entropy_en=all_entropy_en, all_entropy_tr=all_entropy_tr)
plot_entropy_trajectories(mean_entropy_en, mean_entropy_tr,
                          "results/figures/fig6_entropy_trajectories.png",
                          all_entropy_en=all_entropy_en, all_entropy_tr=all_entropy_tr)
print("  fig6_entropy_trajectories ✓")

# Fig 7: Case study
plot_case_study(bp, all_probs_en[best_idx], all_probs_tr[best_idx],
                all_results_en[best_idx]["top_tokens"], all_results_tr[best_idx]["top_tokens"],
                "results/figures/fig7_case_study.pdf")
print("  fig7_case_study ✓")

# Fig 8: Small multiples
plot_small_multiples(all_probs_en, all_probs_tr, problems, "results/figures/fig8_small_multiples.pdf")
plot_small_multiples(all_probs_en, all_probs_tr, problems, "results/figures/fig8_small_multiples.png")
print("  fig8_small_multiples ✓")

# Fig 9: JS divergence heatmap
plot_js_divergence_heatmap(all_js_div, problems, "results/figures/fig9_js_divergence.pdf")
plot_js_divergence_heatmap(all_js_div, problems, "results/figures/fig9_js_divergence.png")
print("  fig9_js_divergence ✓")

# Fig 10: Prompt frame control (3 conditions)
mean_tr_framed_agg, ci_lo_tr_framed, ci_hi_tr_framed = aggregate_trajectories(all_probs_tr_framed)
plot_prompt_frame_control(
    mean_en, ci_lo_en, ci_hi_en,
    mean_tr, ci_lo_tr, ci_hi_tr,
    mean_tr_framed_agg, ci_lo_tr_framed, ci_hi_tr_framed,
    "results/figures/fig10_prompt_frame_control.pdf",
)
plot_prompt_frame_control(
    mean_en, ci_lo_en, ci_hi_en,
    mean_tr, ci_lo_tr, ci_hi_tr,
    mean_tr_framed_agg, ci_lo_tr_framed, ci_hi_tr_framed,
    "results/figures/fig10_prompt_frame_control.png",
)
print("  fig10_prompt_frame_control ✓")

# ═══════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("COMPLETE")
print("="*60)
print(f"\nKey results:")
print(f"  English accuracy: {acc_en:.1f}% [95% CI: {ci_en['ci_lower']*100:.1f}–{ci_en['ci_upper']*100:.1f}]")
print(f"  Turkish accuracy: {acc_tr:.1f}% [95% CI: {ci_tr['ci_lower']*100:.1f}–{ci_tr['ci_upper']*100:.1f}]")
print(f"  Fisher's exact test: p = {fisher_p:.4f}")
print(f"  Final layer P(answer): EN={mean_en[-1]:.4f} TR={mean_tr[-1]:.4f}")
print(f"  TR (Turkish frame) P(answer): {mean_tr_framed[-1]:.4f}")
if valid_onsets:
    print(f"  Median divergence onset: layer {np.median(valid_onsets):.0f}")
print(f"  Late-region Hedges' g: {region_stats['late']['hedges_g']:.2f} "
      f"(p_corrected={region_stats['late']['p_value_corrected']:.4f})")
print(f"\nFigures saved to results/figures/")
print(f"Tables saved to results/tables/")
