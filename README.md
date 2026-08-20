# DNA-Based-Semantic-Search

![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)
![Status](https://img.shields.io/badge/Status-Publication_Ready-success.svg?style=flat-square)

> **The Biophysical Information Bottleneck:** Thermodynamically Constrained Representation Learning for Semantic DNA Similarity Search

This repository contains the official, mathematically rigorous, and 100% physically constrained pipeline for translating high-dimensional Natural Language Processing (NLP) semantics directly into physical DNA sequences. 

Designed for **DNA-based Data Storage** and **In-Memory Molecular Search**, this framework allows querying text databases using highly parallelized thermodynamic hybridization instead of traditional computational hashing.

## 🧬 Overview

Current DNA storage solutions act as "cold archives." Retrieving specific information requires sequencing everything. We propose a paradigm shift: **Content-addressable molecular search**.

By mapping continuous semantic geometry into discrete thermodynamic constraints, information retrieval is executed passively via molecular annealing at O(1) time complexity.

### Features
* **Differentiable Thermodynamics:** An end-to-end differentiable thermodynamic surrogate evaluating DNA hybridization affinity.
* **Q1-Grade Biophysical Rigor:** 
  * Enforces exact SantaLucia nearest-neighbor parameters.
  * Models high-stringency hybridization (75°C / 348.15K) to eliminate non-specific binding.
  * Implements a 4x4 sequence-dependent wobble/mismatch penalty matrix.
  * Penalizes intra-molecular secondary structures (hairpins).
* **Zero-Shot Generalization:** Retains >80% semantic information (Spearman ρ) on unseen medical datasets (BIOSSES) despite extreme physical limitations.

## 🚀 Quick Start

The entire pipeline is wrapped in a single, highly modular Jupyter Notebook ready for Kaggle, Colab, or local execution.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/labonysur-cloud/DNA-Based-Semantic-Search.git
   cd DNA-Based-Semantic-Search
   ```

2. **Install dependencies:**
   ```bash
   pip install torch sentence-transformers datasets matplotlib seaborn scipy scikit-learn
   ```

3. **Run the pipeline:**
   Open `DNA_Based_Semantic_Search.ipynb` in your preferred IDE (Jupyter, VSCode, Kaggle) and execute the cells. The notebook will automatically download standard NLP datasets, encode them, train the surrogate, and output all publication-ready visualizations into a `./figures` directory.

## 📊 Benchmarks & Visualization

The notebook systematically reproduces all critical plots required for scientific validation:
- **Baseline Comparisons:** Benchmarks against Binary Quantization and Locality Sensitive Hashing (LSH).
- **Biological Viability:** Verifies GC content and melting temperatures ($T_m$).
- **Mutation Robustness:** Evaluates semantic preservation under simulated sequencing/synthesis errors.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation
*(Citation details will be updated upon publication)*
