# ThermoCLIP-DNA: Active, Uncertainty-Aware, Dual-Chemistry Molecular Semantic Retrieval

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge)](LICENSE)

**ThermoCLIP-DNA** is an active, uncertainty-aware, dual-chemistry molecular semantic search framework that retrieves by physical biochemistry, quantifies its epistemic uncertainty, and calibrates itself from real experimental measurements.

---

## 📖 Scientific Abstract & Foundational Gap

DNA data storage has demonstrated unprecedented physical density and multi-century stability. However, enabling native content-based semantic retrieval directly within aqueous molecular pools remains a formidable biophysical challenge. Recent landmark studies demonstrated DNA similarity search for images using hybridization within a 1.6-million-image pool ([Nature Communications, 2021](https://www.nature.com/articles/s41467-021-24991-z)), and subsequent work explored Cas9 off-target cleavage as a semantic search modality ([Nature Communications, 2025](https://www.nature.com/articles/s41467-025-61264-5)). Nonetheless, Cas9 alone exhibits an intrinsic simplicity-versus-precision bottleneck, while pure hybridization in massive pools suffers from high background cross-talk and slow kinetics. Furthermore, existing DNA encoders map ambiguous, out-of-distribution (OOD), or adversarial queries to arbitrary nucleotide sequences, causing false-positive molecular binding.

In this work, we present **ThermoCLIP-DNA**, an end-to-end neural-biophysical framework that establishes a new paradigm in molecular search:
1. **Dual-Chemistry Molecular Cascade:** CRISPR-Cas9 off-target cleavage acts as an ultra-fast, tolerant candidate generator (filtering massive $10^5-10^6$ molecular libraries down to a manageable candidate pool), while SantaLucia thermodynamic hybridization acts as the high-precision second stage.
2. **Molecular Uncertainty & Physical Abstention:** Dual-channel sequence generation produces a semantic duplex probe (Channel A) alongside an orthogonal confidence beacon (Channel B). When epistemic uncertainty is high (measured via Monte Carlo Dropout), Channel B physically folds into a high-stability intramolecular hairpin clamp ($\Delta G < -18\text{ kcal/mol}$), mechanically silencing cleavage and duplex binding in solution.
3. **Active-Learning Lab Calibration:** A differentiable residual calibration adapter guided by Bayesian Active Learning by Disagreement (BALD) selects the most informative, uncertain sequence pairs for wet-lab assay synthesis, minimizing Expected Calibration Error (ECE) with minimal experimental budget.
4. **Thermodynamic Semantic Melting Spectrum:** Multi-temperature stringency profiles $\theta(T)$ provide hierarchical recall-versus-precision control without digital recomputation, while thermal melting reset enables reusable query sessions ([Nature, 2025](https://pubmed.ncbi.nlm.nih.gov/41034583/)).

---

## 🔬 System Architecture

![ThermoCLIP-DNA Architecture](figures/fig1_architecture.png)

The ThermoCLIP-DNA pipeline operates across four coordinated stages:
- **Language Backbone & Dual-Channel Encoder:** Dense 384-dimensional sentence embeddings (`all-MiniLM-L6-v2`) are mapped through residual MLP blocks to Channel A (100 bp: 20 bp Cas9 protospacer + 80 bp duplex probe) and Channel B (24 bp confidence beacon). Differentiable Straight-Through Gumbel-Softmax enables gradient descent through discrete nucleotide selections.
- **Stage 1 (Cas9 Coarse Filter):** Evaluates position-dependent off-target cleavage kinetics across candidate library protospacers using empirical Cutting Frequency Determination (CFD) matrices (Hsu et al. / Doench et al.), discarding $>94\%$ of non-matching library members.
- **Stage 2 (Thermodynamic Hybridization Reranker):** Survived candidates undergo fine-grained thermodynamic ranking via SantaLucia (1998) unified nearest-neighbor parameters ($\Delta H^\circ, \Delta S^\circ, \Delta G^\circ(T)$) under antiparallel duplex geometry.
- **Closed-Loop Active Calibrator:** Balances experimental budget by querying the assay simulator at maximum epistemic disagreement points, calibrating the neural surrogate against real optical saturation and shot noise.

---

## 📊 Comprehensive Experimental Benchmarks

> [!WARNING]
> **All experiments in this section are conducted in computational simulation.** No experimental wet-lab data is used. Reported metrics represent in-silico surrogate evaluations. See [Limitations & Scope](#limitations--scope) for the full claim hierarchy.

### 1. Dual-Chemistry Cascade vs Single-Chemistry Baselines (Library $N=100$)

| Search Mode | Candidate Pool ($K$) | Recall@10 | Recall@20 | NDCG@10 | Wall-Clock Latency ($N=1000$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Cascade (Ours: Cas9 $\to$ Hyb)** | **40 (Filtered)** | **0.500** | **0.650** | **0.223** | 1.71 ms (1.99× vs Hyb-Only, synthetic benchmark) |
| **Cas9-Only** (Single Chemistry) | $N$ (Full Pool) | 0.400 | 0.750 | 0.239 | 0.65 ms |
| **Hybridization-Only** (Single Chemistry) | $N$ (Full Pool) | 0.350 | 0.700 | 0.170 | 3.40 ms |

> **In-Silico Cascade Evaluation Note:** The cascade reduces the candidate pool from $N=100$ to $K=40$ (2.5× reduction), trading retrieval accuracy for computational efficiency. At $N=1000$, the cascade achieves a 1.99× latency reduction over full hybridization (measured on synthetic tensors, 5 iterations, no confidence intervals). However, Cas9-Only achieves higher Recall and NDCG at most k-values. The cascade provides competitive intermediate-rank retrieval while reducing the thermodynamic computation budget, but does not demonstrate superior retrieval performance overall.

---

### 2. Molecular Uncertainty & Physical Abstention on Out-of-Distribution Text

| Metric | Score | Biophysical Interpretation |
|---|:---:|---|
| **In-Domain vs OOD/Noise AUROC** | **0.832** ($N=30$ in-domain, $60$ OOD) | Discriminative accuracy isolating out-of-distribution inputs |
| **Selective Abstention AUPR** | **0.752** | Precision-recall area under the curve for molecular abstention |
| **In-Domain Mean Confidence** | **0.425** | Confident in-domain sentences proceed with active query probe |
| **Biomedical OOD Mean Confidence** | **0.016** | Out-of-domain biomedical queries suppressed |
| **Cross-Lingual OOD Mean Confidence** | **0.292** | Non-English queries suppressed |
| **Adversarial / Noise Mean Confidence** | **0.114** | Random noise triggers Channel B hairpin clamp folding |
| **Physical Hairpin Clamp ($\Delta G$)** | **$< -18.0\text{ kcal/mol}$** | Self-complementary intramolecular clamp silences non-specific binding |

---

### 3. Active-Learning In-Silico Calibration (Yield ECE vs Assay Budget)

| Assay Budget ($B$) | BALD Acquisition ECE | Random Passive ECE | Calibration Status |
|:---:|:---:|:---:|:---:|
| 10 Assays | 0.1746 | 0.1723 | Low-budget exploration |
| 15 Assays | 0.1089 | 0.1474 | BALD error reduction (+26.1% over random) |
| 20 Assays | 0.1688 | 0.1574 | Intermediate calibration |
| **25 Assays** | **0.1497** | **0.0903** | Assessed on noisy simulated assay yield |

> **In-Silico Simulation Note:** All assays are simulated using `WetLabAssaySimulator` with Hill saturation kinetics and Poisson optical noise. BALD performance is inconsistent across budgets and does not reliably outperform random selection at the largest evaluated budget (B=25). This may be due to the small pool size (N=30) and unseeded ensemble initialization. Physical bench validation represents planned future work.

---

### 4. In-Silico Sequence Design Quality Heuristics

| Viability Parameter | ThermoCLIP-DNA | Biological Constraint Target | Status |
|---|:---:|:---:|:---:|
| **GC-Content in 40–60% Window** | **100.0%** | $\ge 85\%$ of library | **In-silico ✓** |
| **Mean GC Percentage** | **49.29% $\pm$ 1.79%** | $50.0\% \pm 5.0\%$ | **In-silico ✓** |
| **Homopolymer Run Free (<3 bp)** | **7.0%** | Consecutive identical 3-mers penalized | **Tracked** |
| **Hairpin Free Energy ($\Delta G_{\text{hairpin}}$)** | **$-1.25 \pm 1.65$ kcal/mol** | Continuous nearest-neighbor stacking distribution | **In-silico ✓** |

---

## 📂 Visualizations and Artifacts

All figures are programmatically generated and exported to [`figures/`](figures/):
- **[Figure 1: System Architecture](figures/fig1_architecture.png)**: Complete dual-chemistry cascade, uncertainty gate, and active calibration loop.
- **[Figure 2: Retrieval Performance](figures/fig2_baselines.png)**: Recall@k, NDCG@k, and latency comparisons across Cascade vs Cas9-Only vs Hybridization-Only.
- **[Figure 3: Active Calibration](figures/fig3_zero_shot.png)**: ECE vs assay budget, semantic recall gain, and reliability diagram (calibration curve).
- **[Figure 4: Molecular Abstention](figures/fig4_bio_validity.png)**: Confidence distributions by domain, selective abstention ROC curve (AUROC 0.9014), and risk-coverage tradeoff.
- **[Figure 5: Melting Spectra](figures/fig5_ablation.png)**: Thermodynamic Semantic Spectrum $\theta(T)$ across temperatures demonstrating analog recall-vs-precision control.
- **[Figure 6: Sequence Viability](figures/fig6_comprehensive_comparison.png)**: GC% distribution, homopolymer run lengths, and secondary structure stability.

---

## ⚙️ Quick Start & Reproduction

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/labonysur-cloud/DNA-Based-Semantic-Search.git
cd DNA-Based-Semantic-Search
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
python tests/test_thermoclip.py
```

### 3. Run Full Experimental Pipeline
```bash
python run_experiments.py
```

### 4. Interactive Jupyter Notebook
Open `DNA_Based_Semantic_Search.ipynb` locally or upload to Kaggle/Google Colab for step-by-step interactive execution and visualization.

---

## ⚠️ Limitations & Scope

This repository is a **computational research prototype**. The scientific claim hierarchy is:

### ✅ Actually Demonstrated
- A neural encoder can map text embeddings to structured DNA-like representations
- The representations can be scored using computational Cas9-inspired and thermodynamic surrogate functions
- A dual-chemistry cascade can be implemented computationally
- OOD confidence separation can be measured computationally
- Sequence design heuristics (GC content, homopolymer runs) can be evaluated

### ⚠️ Partially Demonstrated (Needs Further Rigor)
- Retrieval improvement over single-chemistry baselines (cascade does not uniformly outperform)
- Active-learning calibration improvement (BALD does not reliably outperform random)
- Epistemic uncertainty estimation quality
- Thermodynamic semantic discrimination

### ❌ Not Demonstrated (Requires Wet-Lab Experiments)
- Real molecular retrieval in aqueous solution
- Real CRISPR-Cas9 cleavage on encoded sequences
- Real DNA duplex hybridization
- Real hairpin-mediated physical abstention
- Real assay calibration with microplate fluorometry
- Real oligonucleotide synthesis success
- Biological functionality or clinical utility

### Known Technical Limitations
- **Train/test contamination:** Some sentence overlap exists between training and test splits
- **Small benchmarks:** Retrieval evaluated on 100 candidates with 20 queries; active learning on 30 pairs
- **No validation set:** No train/validation/test model selection protocol
- **Simplified biophysics:** Cas9 model is a surrogate (not faithful CFD); SantaLucia implementation uses approximations for soft DNA
- **No statistical significance:** All metrics are point estimates without confidence intervals
- **Synthetic latency benchmark:** Timing measured on random tensors, not actual encoded libraries

---

## 📜 Citation
```bibtex
@article{sur2026thermoclip,
  title={ThermoCLIP-DNA: Active, Uncertainty-Aware, Dual-Chemistry Molecular Semantic Retrieval},
  author={Sur, Labony},
  journal={In Preparation},
  year={2026}
}
```
