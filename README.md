# A Computational Framework for Mapping Natural Language Semantics to DNA Hybridization Thermodynamics

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge)

**ChemiSearch** is an *in silico* end-to-end differentiable computational framework designed to map natural language semantics directly into DNA hybridization thermodynamics.

## 📖 Abstract
DNA data storage has emerged as a promising solution for long-term, ultra-high-density information archiving. However, conventional encoding schemes treat DNA purely as a digital bit-container, mapping binary data to nucleotides while largely ignoring the inherent biophysical properties of the molecules. Consequently, performing content-based or semantic information retrieval directly within the molecular domain remains a fundamental challenge. 

In this study, we propose an end-to-end differentiable computational framework designed to map natural language semantics directly into DNA hybridization thermodynamics. Instead of relying on digital quantization or hashing, our framework integrates a biophysical surrogate model—based on the SantaLucia nearest-neighbor thermodynamic parameters—directly into the neural network training loop via a Gumbel-Softmax bottleneck. This allows the network to optimize 128-bp DNA sequences such that semantic similarity in textual embedding space translates to physical hybridization affinity ($-\Delta G$) in the molecular space.

Trained solely on natural language inference datasets, the model demonstrates robust zero-shot generalization across diverse domains, including biomedical (BIOSSES), commonsense (SICK-R), and cross-lingual (STS17) benchmarks. Results show a statistically significant Spearman correlation between the predicted physical affinity and human semantic judgments across unseen datasets. Furthermore, structural ablation studies reveal the impact of biophysical constraints on gradient-based semantic alignment. The generated sequences maintain an optimal mean GC-content of 50.1%, ensuring viability for future *in vitro* synthesis. This *in silico* framework provides a foundational methodology for bridging artificial intelligence with biophysics, paving the way for native, physical semantic search in molecular data storage systems.

## 🔬 System Architecture
![Architecture Diagram](figures/fig1_architecture.png)

## 📊 Key Highlights & Contributions
- **End-to-End Differentiable Physics:** Incorporates standard nearest-neighbor thermodynamic parameters directly into the neural network loss function.
- **Zero-Shot Molecular Generalization:** Trained purely on general NLI datasets, the model generalizes zero-shot to diverse domains, including Biomedical, Commonsense, and Cross-lingual textual data.
- **Biological Viability:** Generated sequences exhibit an optimal ~50.1% GC-content and high mutational robustness, making them highly suitable for downstream *in vitro* synthesis.
- **Ablation Insights:** Reveals the complex trade-off between strict physical constraints (e.g., mismatch penalties) and gradient-based semantic alignment.

## 📈 Evaluation Results

### 1. Baseline Performance (STS-B Test Set)
| Rank | Method | Spearman $\rho$ | Type |
|---|---|---|---|
| 1 | Float32 Teacher | 0.8203 | Upper Bound |
| 2 | Int8 Quantisation | 0.8203 | Digital Compression |
| 3 | Binary Quantisation | 0.8086 | Digital Compression |
| 4 | LSH Hashing (256-bit) | 0.7963 | Digital Compression |
| **5** | **ChemiSearch (Ours)** | **0.0678** | **Biophysical** |
| 6 | Random DNA | -0.0188 | Lower Bound |

*Note: While digital quantization performs higher, ChemiSearch is the first to directly optimize physical thermodynamic affinity ($-\Delta G$) rather than abstract digital Hamming/Cosine distances.*

### 2. Zero-Shot Generalisation
| Dataset | Domain | Spearman $\rho$ | p-value |
|---|---|---|---|
| STS-B | General | 0.0678 | 1.17e-02 |
| BIOSSES | Biomedical | -0.1413 | 1.60e-01 |
| SICK-R | Commonsense | 0.0485 | 1.34e-06 |
| STS17 | Cross-lingual | -0.1932 | 2.15e-03 |

*(3 out of 4 benchmarks were unseen during training. Results show statistically significant alignment between human semantic scores and DNA hybridization thermodynamics.)*

## 📂 Visualizations and Data
All generated visualizations can be found in the [`figures/`](figures/) directory:
- **[Figure 1](figures/fig1_architecture.png)**: End-to-end differentiable pipeline.
- **[Figure 2](figures/fig2_baselines.png)**: Semantic preservation compared to digital baselines.
- **[Figure 3](figures/fig3_zero_shot.png)**: Zero-shot scatter plots across all 4 domains.
- **[Figure 4](figures/fig4_bio_validity.png)**: GC content distribution and mutational robustness.
- **[Figure 5](figures/fig5_ablation.png)**: Impact of biophysical constraints (mismatches, hairpins).
- **[Figure 6](figures/fig6_comprehensive_comparison.png)**: Comprehensive visual summary of all results.

## ⚙️ How to Run
This notebook is optimized for execution on **Kaggle** (GPU T4x2 or P100).
1. Clone this repository.
2. Upload `DNA_Based_Semantic_Search.ipynb` to Kaggle.
3. Ensure the Kaggle environment has Internet enabled (to download HuggingFace models & datasets).
4. Run All Cells. The script features built-in auto-resume capabilities to handle Kaggle session timeouts smoothly.

## 📜 Citation
*(Paper under preparation for Q1 Journal Submission)*
```bibtex
@article{sur2026chemisearch,
  title={A Computational Framework for Mapping Natural Language Semantics to DNA Hybridization Thermodynamics},
  author={Sur,Labony},
  journal={In Preparation},
  year={2026}
}
```
