"""
Publication-Quality Visualization Suite for ThermoCLIP-DNA.
Generates Figures 1 to 6 with scientific typography and styling.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Publication styling defaults
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#2c3e50"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = "--"


def plot_architecture_diagram(save_path: str = "figures/fig1_architecture.png", show: bool = False) -> None:
    """Renders the comprehensive ThermoCLIP-DNA dual-chemistry system architecture diagram."""
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle="", bg="#ffffff", border="#2c3e50", title_color="#1a252f"):
        box = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.8, edgecolor=border, facecolor=bg, zorder=2
        )
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=11, fontweight="bold", color=title_color, zorder=3)
            ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center", fontsize=8.5, color="#555555", zorder=3)
        else:
            ax.text(x + w / 2, y + h * 0.50, title, ha="center", va="center", fontsize=11, fontweight="bold", color=title_color, zorder=3)

    def draw_arrow(x1, y1, x2, y2, label="", color="#34495e", style="->"):
        ax.annotate(
            label, xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle=style, color=color, lw=2.0, mutation_scale=15),
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=color, zorder=4
        )

    # Title Banner
    ax.text(0.5, 0.96, "ThermoCLIP-DNA: Dual-Chemistry Molecular Semantic Search Engine",
            ha="center", va="center", fontsize=17, fontweight="bold", color="#1b263b")
    ax.text(0.5, 0.925, "Active Lab Calibration • CRISPR-Cas9 Coarse Filter • SantaLucia Hybridization Reranking • Molecular Abstention",
            ha="center", va="center", fontsize=10.5, color="#415a77", style="italic")

    # Column 1: Query Input & Neural Trunk
    draw_box(0.04, 0.72, 0.24, 0.14, "Natural Language Query", "Unstructured Text (STS-B / PubMed)", bg="#e8f4f8", border="#0077b6")
    draw_box(0.04, 0.50, 0.24, 0.15, "Dense Semantic Backbone", "Frozen all-MiniLM-L6-v2 (384-d)", bg="#edf2f4", border="#4a4e69")
    draw_arrow(0.16, 0.72, 0.16, 0.65)

    # Epistemic Uncertainty Branch
    draw_box(0.04, 0.28, 0.24, 0.15, "MC-Dropout Uncertainty", "Epistemic Variance & Confidence c(x)", bg="#fefae0", border="#dda15e")
    draw_arrow(0.16, 0.50, 0.16, 0.43)

    # Column 2: Dual-Channel Molecular Encoder
    draw_box(0.35, 0.68, 0.28, 0.18, "Channel A: Semantic Code", "20-bp Cas9 Protospacer + 80-bp Duplex Probe\nStraight-Through Gumbel-Softmax Discretization", bg="#e8f8f5", border="#2ec4b6")
    draw_box(0.35, 0.38, 0.28, 0.18, "Channel B: Confidence Beacon", "24-bp Orthogonal Molecular Reporter\nor Self-Complementary Clamp Hairpin", bg="#fff0f3", border="#ff4d6d")
    draw_arrow(0.28, 0.58, 0.35, 0.76, label="Latent\nTrunk")
    draw_arrow(0.28, 0.36, 0.35, 0.46, label="Epistemic\nGate")

    # Molecular Abstention Clamp Action
    draw_box(0.35, 0.12, 0.28, 0.14, "Molecular Abstention Gate", "c(x) < tau -> Fold Clamped Hairpin\nPhysically silences cleavage & binding", bg="#ffe5d9", border="#d90429", title_color="#9d0208")
    draw_arrow(0.49, 0.38, 0.49, 0.26, label="Low Conf.")

    # Column 3: Dual-Chemistry Cascade
    draw_box(0.69, 0.72, 0.27, 0.17, "Stage 1: Cas9 Coarse Filter", "CRISPR-SpCas9 Off-Target Cleavage\nFilters 10^5-10^6 Pool -> Top Candidates (K=50)", bg="#d8f3dc", border="#2d6a4f")
    draw_box(0.69, 0.46, 0.27, 0.17, "Stage 2: Hybridization Reranker", "SantaLucia Duplex Kinetics (dH, dS)\nMulti-Temperature Stringency Spectrum theta(T)", bg="#d7e3fc", border="#1d3557")
    draw_arrow(0.63, 0.76, 0.69, 0.79, label="Guide RNA")
    draw_arrow(0.82, 0.72, 0.82, 0.63, label="Enriched\nCandidates")
    draw_arrow(0.63, 0.70, 0.69, 0.55, label="Duplex Probe")

    # Column 4 / Bottom: Closed-Loop Active Learning Calibration
    draw_box(0.69, 0.14, 0.27, 0.22, "Active Biophysical Calibration Loop", "BALD Active Acquisition\nGrounding via In-Silico Assay Simulator\nMinimizes Expected Calibration Error (ECE)", bg="#f3e8ee", border="#7209b7")
    draw_arrow(0.82, 0.46, 0.82, 0.36, label="Residual\nFeatures")
    draw_arrow(0.69, 0.25, 0.63, 0.25, style="<->", label="Closed Loop\nCalibration")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def plot_cascade_performance(
    k_vals: List[int],
    recalls: Dict[str, List[float]],
    ndcgs: Dict[str, List[float]],
    pool_sizes: List[int],
    times_ms: Dict[str, List[float]],
    save_path: str = "figures/fig2_baselines.png",
    show: bool = False,
) -> None:
    """Plots Fig 2: Dual-Chemistry Cascade vs Single Chemistry Baselines."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

    colors = {"Cascade (Ours)": "#2b9348", "Cas9-Only": "#e63946", "Hybridization-Only": "#4361ee"}
    markers = {"Cascade (Ours)": "o", "Cas9-Only": "s", "Hybridization-Only": "^"}

    # Panel A: Recall@k
    ax = axes[0]
    for name, vals in recalls.items():
        ax.plot(k_vals, vals, marker=markers.get(name, "o"), label=name, color=colors.get(name, "#333"), lw=2.2, ms=7)
    ax.set_title("(a) Retrieval Recall@k", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Top-k Retrieved Candidates", fontsize=11)
    ax.set_ylabel("Recall@k", fontsize=11)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ccc")

    # Panel B: NDCG@k
    ax = axes[1]
    for name, vals in ndcgs.items():
        ax.plot(k_vals, vals, marker=markers.get(name, "o"), label=name, color=colors.get(name, "#333"), lw=2.2, ms=7)
    ax.set_title("(b) Ranking Quality (NDCG@k)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Top-k Candidates", fontsize=11)
    ax.set_ylabel("NDCG@k", fontsize=11)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ccc")

    # Panel C: Latency vs Library Pool Size
    ax = axes[2]
    for name, t_list in times_ms.items():
        ax.plot(pool_sizes, t_list, marker=markers.get(name, "o"), label=name, color=colors.get(name, "#333"), lw=2.2, ms=7)
    ax.set_title("(c) Search Latency vs Pool Size", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Molecular Library Size (N)", fontsize=11)
    ax.set_ylabel("Search Latency (ms)", fontsize=11)
    ax.set_xscale("log")
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ccc")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def plot_active_learning_calibration(
    budgets: List[int],
    ece_bald: List[float],
    ece_random: List[float],
    recall_bald: List[float],
    recall_random: List[float],
    rel_bins: np.ndarray,
    rel_uncal: np.ndarray,
    rel_cal: np.ndarray,
    save_path: str = "figures/fig3_zero_shot.png",
    show: bool = False,
) -> None:
    """Plots Fig 3: Active-Learning Calibration & Reliability Diagram."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

    # Panel A: ECE vs Budget
    ax = axes[0]
    ax.plot(budgets, ece_bald, "o-", color="#2a9d8f", lw=2.4, ms=8, label="Active Selection (BALD)")
    ax.plot(budgets, ece_random, "s--", color="#e76f51", lw=2.0, ms=7, label="Passive Random Selection")
    ax.set_title("(a) Expected Calibration Error (ECE)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Simulated Assay Budget (N Pairs)", fontsize=11)
    ax.set_ylabel("Assay Yield ECE (Lower is Better)", fontsize=11)
    ax.grid(True)
    ax.legend(frameon=True)

    # Panel B: Top-5 Recall Gain vs Budget
    ax = axes[1]
    ax.plot(budgets, recall_bald, "o-", color="#2a9d8f", lw=2.4, ms=8, label="Active Selection (BALD)")
    ax.plot(budgets, recall_random, "s--", color="#e76f51", lw=2.0, ms=7, label="Passive Random Selection")
    ax.set_title("(b) Top-Quartile Yield Overlap vs Assay Budget", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Simulated Assay Budget (N Pairs)", fontsize=11)
    ax.set_ylabel("Top-25% Yield Overlap", fontsize=11)
    ax.grid(True)
    ax.legend(frameon=True)

    # Panel C: Reliability Diagram (Calibration Curve)
    ax = axes[2]
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")
    # Filter out NaN values from empty bins
    valid_uncal = ~np.isnan(rel_uncal)
    valid_cal = ~np.isnan(rel_cal)
    ax.plot(rel_bins[valid_uncal], rel_uncal[valid_uncal], "x:", color="#e63946", lw=1.8, ms=7, label="Uncalibrated Biophysical Model")
    ax.plot(rel_bins[valid_cal], rel_cal[valid_cal], "o-", color="#2a9d8f", lw=2.2, ms=7, label="Actively Calibrated (Ours)")
    ax.set_title("(c) Empirical Reliability Diagram", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Biophysical Yield", fontsize=11)
    ax.set_ylabel("Ground-Truth Simulated Yield", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True)
    ax.legend(frameon=True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def plot_abstention_and_ood(
    conf_in_domain: np.ndarray,
    conf_biosses: np.ndarray,
    conf_cross_lingual: np.ndarray,
    conf_noise: np.ndarray,
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    coverage: np.ndarray,
    risk: np.ndarray,
    save_path: str = "figures/fig4_bio_validity.png",
    show: bool = False,
) -> None:
    """Plots Fig 4: Molecular Uncertainty, OOD Detection & Selective Abstention."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

    # Panel A: Confidence Score Distributions across Domains
    ax = axes[0]
    sns.kdeplot(conf_in_domain, ax=ax, label="In-Domain (STS-B)", color="#2a9d8f", fill=True, alpha=0.35, lw=2.0, warn_singular=False)
    sns.kdeplot(conf_biosses, ax=ax, label="Biomedical (OOD)", color="#e76f51", fill=True, alpha=0.25, lw=1.8, warn_singular=False)
    sns.kdeplot(conf_cross_lingual, ax=ax, label="Cross-Lingual (OOD)", color="#f4a261", fill=True, alpha=0.25, lw=1.8, warn_singular=False)
    sns.kdeplot(conf_noise, ax=ax, label="Adversarial / Random", color="#d62828", fill=True, alpha=0.25, lw=1.8, warn_singular=False)
    ax.axvline(0.55, color="#333", linestyle="--", lw=1.5, label="Abstention Cutoff (tau=0.55)")
    ax.set_title("(a) Molecular Confidence by Domain", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epistemic Confidence Score c(x)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.grid(True)
    ax.legend(frameon=True, fontsize=8.5)

    # Panel B: ROC Curve for Abstaining on OOD / Garbage
    ax = axes[1]
    ax.plot(fpr, tpr, color="#2b2d42", lw=2.4, label=f"OOD AUROC = {auroc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2)
    ax.set_title("(b) Selective Abstention ROC", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (Unjustified Retrieval)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Valid In-Domain)", fontsize=11)
    ax.grid(True)
    ax.legend(frameon=True, loc="lower right")

    # Panel C: Risk-Coverage Curve
    ax = axes[2]
    ax.plot(coverage * 100.0, risk, "o-", color="#9d0208", lw=2.2, ms=6)
    ax.set_title("(c) Risk vs Coverage Trade-Off", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Library Coverage (%) [100% - Abstain%]", fontsize=11)
    ax.set_ylabel("Retrieval Error Rate (Risk)", fontsize=11)
    ax.grid(True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def plot_melting_spectra(
    temps_c: np.ndarray,
    theta_exact: np.ndarray,
    theta_near: np.ndarray,
    theta_topical: np.ndarray,
    theta_unrelated: np.ndarray,
    theta_clamped: np.ndarray,
    save_path: str = "figures/fig5_ablation.png",
    show: bool = False,
) -> None:
    """Plots Fig 5: Thermodynamic Semantic Melting Spectrum theta(T)."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ax.plot(temps_c, theta_exact, "o-", color="#1b4965", lw=2.6, ms=5, label="Exact Semantic Match (Cosine > 0.95)")
    ax.plot(temps_c, theta_near, "s-", color="#2a9d8f", lw=2.2, ms=5, label="Near Semantic Match (Cosine ~ 0.75)")
    ax.plot(temps_c, theta_topical, "^-", color="#e76f51", lw=2.0, ms=5, label="Topical / Broadly Related (Cosine ~ 0.45)")
    ax.plot(temps_c, theta_unrelated, "x--", color="#6c757d", lw=1.8, ms=5, label="Semantic Mismatch (Cosine < 0.15)")
    ax.plot(temps_c, theta_clamped, "d:", color="#d90429", lw=2.2, ms=5, label="Abstained (Hairpin Clamped)")

    # Highlight hierarchical temperature regimes
    ax.axvspan(45, 60, color="#bde0fe", alpha=0.3, label="Coarse Regime (Broad Recall)")
    ax.axvspan(65, 78, color="#ffcbf2", alpha=0.3, label="Fine Regime (High Precision)")

    ax.set_title("Thermodynamic Semantic Spectrum: Temperature-Controlled Melting theta(T)", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Fraction Bound theta(T)", fontsize=12)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def plot_biophysical_viability(
    gc_fractions: np.ndarray,
    hp_counts: np.ndarray,
    hairpin_mfes: np.ndarray,
    save_path: str = "figures/fig6_comprehensive_comparison.png",
    show: bool = False,
) -> None:
    """Plots Fig 6: Biological Viability & Synthesis Feasibility."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

    # Panel A: GC Content Distribution
    ax = axes[0]
    sns.histplot(gc_fractions * 100.0, ax=ax, kde=True, color="#2b9348", bins=25)
    ax.axvspan(40, 60, color="#55a630", alpha=0.2, label="Optimal Synthesis Window (40-60%)")
    ax.set_title("(a) Sequence GC% Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("GC Content (%)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True)

    # Panel B: Homopolymer Runs
    ax = axes[1]
    bins = np.arange(0, max(hp_counts) + 2) - 0.5
    ax.hist(hp_counts, bins=bins, color="#4361ee", edgecolor="#1d3557", rwidth=0.7)
    ax.set_title("(b) Consecutive Homopolymer Runs (>=3 bp)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Homopolymer Runs per Strand", fontsize=11)
    ax.set_ylabel("Sequence Count", fontsize=11)
    ax.grid(True)

    # Panel C: Hairpin Stability MFE
    ax = axes[2]
    sns.histplot(hairpin_mfes, ax=ax, kde=True, color="#7209b7", bins=20, label="Library Strands (Unfolded)")
    ax.axvline(0.0, color="#333333", linestyle=":", lw=1.4, label="Linear Unstructured (0.0 kcal/mol)")
    ax.set_title("(c) Intramolecular Hairpin Folding Free Energy", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Folding Free Energy DeltaG_hairpin (kcal/mol)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(frameon=True, fontsize=8.5)
    ax.grid(True)


    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
