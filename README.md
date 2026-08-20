# DNA-Based Semantic Search

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Publication_Ready-success?style=flat-square)

**ChemiSearch: Differentiable Biophysical Representation Learning for DNA-Based Semantic Search**

A fully differentiable framework that maps natural language semantics directly into physically constrained DNA oligonucleotide sequences, enabling content-addressable molecular information retrieval via thermodynamic hybridisation.

---

## Overview

Current DNA storage systems operate as cold archives. Retrieving specific information requires sequencing the entire pool. This work proposes a paradigm shift: **content-addressable molecular search**.

By learning to encode the continuous geometry of sentence embeddings into discrete DNA sequences whose thermodynamic affinity mirrors semantic similarity, information retrieval can be executed via molecular annealing at O(1) time complexity.

## Scientific Contributions

- End-to-end differentiable text-to-DNA encoder (Gumbel-Softmax relaxation)
- Fully vectorised SantaLucia nearest-neighbour thermodynamic surrogate (75 degrees C, 348.15 K)
- Sequence-dependent 4x4 wobble/mismatch penalty matrix
- Hairpin / secondary-structure penalty via self-complementarity dot product
- Rigorous ablation study, bootstrapped confidence intervals, and multi-benchmark zero-shot evaluation

## Results

| Method | STS-B Spearman rho |
|---|---|
| Float32 Teacher (upper bound) | 0.820 |
| LSH Hashing 256-bit | 0.796 |
| Binary Quantisation | 0.809 |
| Random DNA (lower bound) | ~0.02 |
| **ChemiSearch (Ours)** | **see notebook** |

Zero-shot evaluation: BIOSSES biomedical benchmark (no biomedical training data used).

## Quickstart

```bash
git clone https://github.com/labonysur-cloud/DNA-Based-Semantic-Search.git
cd DNA-Based-Semantic-Search
pip install -r requirements.txt
jupyter notebook DNA_Based_Semantic_Search.ipynb
```

Run all cells in order. Embeddings are cached after the first run; the pipeline auto-resumes after any kernel restart.

## Notebook Structure

| Cell | Description |
|---|---|
| 1 | Environment setup and seed locking |
| 2 | Data acquisition (STS-B, AllNLI, BIOSSES) |
| 3 | Teacher embedding extraction and persistent caching |
| 4 | Model architecture and thermodynamic surrogate |
| 5 | Training loop (AdamW, cosine LR, early stopping) |
| 6 | Figure 3: Zero-shot generalisation scatter plots |
| 7 | Figure 2: Baseline comparison bar chart |
| 8 | Figure 4: GC content distribution and mutation robustness |
| 9 | Figure 5: Ablation study |
| 10 | Molecular case study |
| 11 | Summary results tables and export |

## Requirements

See `requirements.txt`. Tested on Python 3.10, PyTorch 2.0+, CUDA 11.8+.

## Physical Model

The thermodynamic surrogate computes:

```
delta_G(i, i+1) = delta_H(i, i+1) - T * delta_S(i, i+1) / 1000
```

using SantaLucia (1998) nearest-neighbour parameters at T = 348.15 K (75 degrees C).
All physical constants are registered as non-trainable buffers. Only the
ResidualMLPEncoder (~790K parameters) has learnable weights.

## Limitations

- Sequences are evaluated in silico only; wet-lab validation is left as future work.
- Fixed sequence length of 128 bp; variable-length encoding is not yet supported.
- The Gumbel-Softmax relaxation introduces a train/eval distribution gap that grows at low temperatures.

## License

MIT License. See LICENSE file.

## Citation

Citation information will be added upon publication.
