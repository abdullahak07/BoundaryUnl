"""
plots.py — Generate all paper figures.

Figure 1 : Leakage Score bar chart across 4 conditions
Figure 2 : FA vs RA tradeoff scatter
Figure 3 : Cross-modal leakage heatmap (retrieval vs generation)
Figure 4 : Reactivation attack success rate
Figure 5 : Unlearning loss curves (GA + KL)
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional

from config import RESULTS_DIR, CONDITIONS
from conditions import ConditionResult
from evaluate import compute_metrics, is_correct, build_results_table

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font_scale=1.2)
PALETTE = {
    "baseline":     "#e74c3c",
    "index_only":   "#f39c12",
    "vlm_only":     "#3498db",
    "dual_channel": "#27ae60",
}
LABEL_MAP = {
    "baseline":     "Baseline",
    "index_only":   "Index-only",
    "vlm_only":     "VLM-only",
    "dual_channel": "Dual-Channel (DCL-FP)",
}


def _save(fig, name: str):
    import plots as _self
    out_dir = getattr(_self, "RESULTS_DIR", RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Figure 1: Main leakage bar chart
# ─────────────────────────────────────────────────────────────────

def plot_leakage_bars(
    all_results: Dict[str, ConditionResult],
    filename: str = "fig1_leakage.pdf", filename_prefix: str = "",
):
    metrics_per_cond = {c: compute_metrics(r) for c, r in all_results.items()}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metric_labels = [
        ("leakage_score",    "Leakage Score (LS↓)"),
        ("forget_accuracy",  "Forget Accuracy (FA↓)"),
        ("retrieval_hit",    "Retrieval Hit (RH↓)"),
    ]

    for ax, (metric_key, title) in zip(axes, metric_labels):
        values = [metrics_per_cond.get(c, {}).get(metric_key, 0.0) for c in CONDITIONS]
        colors = [PALETTE[c] for c in CONDITIONS]
        bars   = ax.bar(
            [LABEL_MAP[c] for c in CONDITIONS],
            values,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.tick_params(axis='x', rotation=30)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle("Dual-Channel Leakage Across Unlearning Conditions", fontsize=13)
    plt.tight_layout()
    _save(fig, filename_prefix + filename)


# ─────────────────────────────────────────────────────────────────
# Figure 2: FA vs RA trade-off
# ─────────────────────────────────────────────────────────────────

def plot_fa_ra_tradeoff(
    all_results: Dict[str, ConditionResult],
    filename: str = "fig2_fa_ra.pdf", filename_prefix: str = "",
):
    fig, ax = plt.subplots(figsize=(6, 5))

    for cond, res in all_results.items():
        m = compute_metrics(res)
        fa = m["forget_accuracy"]
        ra = m["retain_accuracy"]
        ax.scatter(fa, ra, s=180, color=PALETTE[cond],
                   label=LABEL_MAP[cond], zorder=3, edgecolors="black", linewidths=0.8)
        ax.annotate(LABEL_MAP[cond], (fa, ra),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    # Target region (bottom-right = low FA, high RA)
    ax.axvline(0.1, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.fill_between([0, 0.1], [0.8, 0.8], [1.05, 1.05],
                    color="green", alpha=0.07, label="GDPR-compliant region")

    ax.set_xlabel("Forget Accuracy (FA↓)", fontsize=11)
    ax.set_ylabel("Retain Accuracy (RA↑)", fontsize=11)
    ax.set_title("Privacy-Utility Trade-off", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, filename_prefix + filename)


# ─────────────────────────────────────────────────────────────────
# Figure 3: Cross-modal leakage heatmap
# ─────────────────────────────────────────────────────────────────

def plot_crossmodal_heatmap(
    all_results: Dict[str, ConditionResult],
    filename: str = "fig3_crossmodal.pdf", filename_prefix: str = "",
):
    """
    Heatmap: rows = conditions, cols = leakage channels.
    Cell value = fraction of entities leaking via that channel.
    """
    channels = ["Retrieval\n(Index)", "Generation\n(VLM)", "Combined\n(Either)"]
    data = []

    for cond in CONDITIONS:
        res = all_results.get(cond)
        if res is None:
            data.append([0, 0, 0])
            continue
        rh  = np.mean([qr.forget_in_topk for qr in res.forget_queries]) if res.forget_queries else 0
        fa  = np.mean([is_correct(qr.predicted_answer, qr.ground_truth)
                       for qr in res.forget_queries]) if res.forget_queries else 0
        ls  = np.mean([
            float(qr.forget_in_topk or is_correct(qr.predicted_answer, qr.ground_truth))
            for qr in res.forget_queries
        ]) if res.forget_queries else 0
        data.append([rh, fa, ls])

    df = pd.DataFrame(
        data,
        index=[LABEL_MAP[c] for c in CONDITIONS],
        columns=channels,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        df, ax=ax, annot=True, fmt=".2f",
        cmap="YlOrRd", vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Leakage Rate"},
    )
    ax.set_title("Cross-Modal Leakage Rates by Channel and Condition",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Leakage Channel")
    ax.set_ylabel("Unlearning Condition")
    plt.tight_layout()
    _save(fig, filename_prefix + filename)


# ─────────────────────────────────────────────────────────────────
# Figure 4: Reactivation attack
# ─────────────────────────────────────────────────────────────────

def plot_reactivation(
    all_results: Dict[str, ConditionResult],
    filename: str = "fig4_reactivation.pdf", filename_prefix: str = "",
):
    dual = all_results.get("dual_channel")
    if dual is None or not dual.reactivation_queries:
        logger.warning("No dual_channel reactivation data to plot.")
        return

    correct = sum(
        is_correct(qr.predicted_answer, qr.ground_truth)
        for qr in dual.reactivation_queries
    )
    total  = len(dual.reactivation_queries)
    rr     = correct / total if total else 0.0
    ls_dc  = compute_metrics(dual)["leakage_score"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Direct\nLeakage (LS)", "Reactivation\nAttack (RR)"],
        [ls_dc, rr],
        color=["#27ae60", "#8e44ad"],
        edgecolor="black", linewidth=0.8, width=0.4,
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Leakage Rate")
    ax.set_title("Residual Leakage: Direct vs Reactivation Attack\n(Dual-Channel Condition)",
                 fontsize=11)
    for bar, val in zip(bars, [ls_dc, rr]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=11)
    plt.tight_layout()
    _save(fig, filename_prefix + filename)


# ─────────────────────────────────────────────────────────────────
# Figure 5: Unlearning loss curves
# ─────────────────────────────────────────────────────────────────

def plot_loss_curves(
    history: Dict[str, List[float]],
    filename: str = "fig5_loss_curves.pdf", filename_prefix: str = "",
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history["ga_loss"], color="#e74c3c", label="GA loss (forget)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Gradient Ascent Loss (Forget Set)")
    axes[0].legend()

    axes[1].plot(history["kl_loss"], color="#3498db", label="KL loss (retain)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("KL Regularisation Loss (Retain Set)")
    axes[1].legend()

    plt.suptitle("VLM Unlearning Training Curves", fontsize=13)
    plt.tight_layout()
    _save(fig, filename_prefix + filename)


# ─────────────────────────────────────────────────────────────────
# Convenience: generate all figures at once
# ─────────────────────────────────────────────────────────────────

def generate_all_figures(
    all_results: Dict[str, ConditionResult],
    loss_history: Optional[Dict] = None,
    filename_prefix: str = "",
):
    plot_leakage_bars(all_results, filename_prefix=filename_prefix)
    plot_fa_ra_tradeoff(all_results, filename_prefix=filename_prefix)
    plot_crossmodal_heatmap(all_results, filename_prefix=filename_prefix)
    plot_reactivation(all_results, filename_prefix=filename_prefix)
    if loss_history:
        plot_loss_curves(loss_history, filename_prefix=filename_prefix)
    logger.info("All figures generated.")
