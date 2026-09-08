"""
Thermodynamic Hybridization Engine based on SantaLucia (1998) Unified Nearest-Neighbor Parameters.
Implements:
  - Antiparallel duplex binding free energy DeltaG(T)
  - Salt-corrected enthalpy, entropy, and Gibbs free energy
  - Temperature melting curves theta(T) (Semantic Melting Spectrum)
  - Intramolecular hairpin stability estimator DeltaG_hairpin
  - Biological viability constraints (GC-content balance, homopolymer runs)
"""

from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GAS_CONSTANT_R = 1.9872e-3  # kcal / (mol * K)


class SantaLuciaDuplexModel(nn.Module):
    """
    Differentiable biophysical model for DNA duplex hybridization thermodynamics.
    """

    # SantaLucia 1998 unified nearest-neighbour enthalpy (dH, kcal/mol) for dinucleotide steps (5'->3')
    # Row: 5' base [A, C, G, T]; Col: 3' base [A, C, G, T]
    _DH = [
        [-7.9,  -8.4,  -7.8,  -7.2],  # AA, AC, AG, AT
        [-8.5,  -8.0, -10.6,  -7.8],  # CA, CC, CG, CT
        [-8.2,  -9.8,  -8.0,  -8.4],  # GA, GC, GG, GT
        [-7.2,  -8.2,  -8.5,  -7.9],  # TA, TC, TG, TT
    ]

    # SantaLucia 1998 unified nearest-neighbour entropy (dS, cal/(mol*K))
    _DS = [
        [-22.2, -22.4, -21.0, -20.4],  # AA, AC, AG, AT
        [-22.7, -19.9, -27.2, -21.0],  # CA, CC, CG, CT
        [-22.2, -24.4, -19.9, -22.4],  # GA, GC, GG, GT
        [-21.3, -22.2, -22.7, -22.2],  # TA, TC, TG, TT
    ]

    # Watson-Crick pairing matrix: A(0)-T(3), C(1)-G(2), G(2)-C(1), T(3)-A(0)
    _WC = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]

    # Mismatch penalty matrix (kcal/mol at 37 C) for antiparallel opposing bases
    # Rows: Strand 1 base (A=0, C=1, G=2, T=3)
    # Cols: Strand 2 base (A=0, C=1, G=2, T=3)
    _MM = [
        [ 2.2, 1.6, 2.0, 0.0 ],  # A opposes: A, C, G, T (A-T is WC match)
        [ 1.6, 1.8, 0.0, 1.5 ],  # C opposes: A, C, G, T (C-G is WC match)
        [ 2.0, 0.0, 2.2, 0.6 ],  # G opposes: A, C, G, T (G-C is WC match, G-T is wobble)
        [ 0.0, 1.5, 0.6, 1.8 ],  # T opposes: A, C, G, T (T-A is WC match, T-G is wobble)
    ]

    def __init__(
        self,
        seq_len: int = 80,
        default_temp_c: float = 65.0,
        na_conc_m: float = 0.150,  # 150 mM physiological / assay saline
    ):
        super().__init__()
        self.seq_len = seq_len
        self.default_temp_k = default_temp_c + 273.15
        self.na_conc = na_conc_m

        self.register_buffer("dH_matrix", torch.tensor(self._DH, dtype=torch.float32))
        self.register_buffer("dS_matrix", torch.tensor(self._DS, dtype=torch.float32))
        self.register_buffer("wc_matrix", torch.tensor(self._WC, dtype=torch.float32))
        self.register_buffer("mismatch_matrix", torch.tensor(self._MM, dtype=torch.float32))

        # Standard initiation parameters (SantaLucia 1998 for duplexes with at least one G/C)
        self.dH_init = 0.2    # kcal/mol
        self.dS_init = -5.7   # cal/(mol*K)

    def salt_corrected_dS(self, raw_dS: torch.Tensor, num_steps: int) -> torch.Tensor:
        """Apply SantaLucia salt correction for entropy based on Na+ concentration."""
        # delta_S(salt) = delta_S(1M Na+) + 0.368 * (N - 1) * ln([Na+])
        salt_term = 0.368 * num_steps * np.log(self.na_conc)
        return raw_dS + salt_term

    def compute_duplex_thermodynamics(
        self,
        dna1: torch.Tensor,
        dna2: torch.Tensor,
        temp_k: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Differentiably computes antiparallel hybridization thermodynamics between dna1 and dna2.
        Args:
            dna1: (B, L, 4) Query probe 5' -> 3'
            dna2: (B, L, 4) Target probe 5' -> 3'
            temp_k: Operating temperature in Kelvin (defaults to self.default_temp_k)
        Returns:
            Dict containing dH, dS, dG, mismatch_penalty, affinity (-dG), and fraction_bound
        """
        if temp_k is None:
            temp_k = self.default_temp_k

        B, L, _ = dna1.shape
        # Antiparallel duplex alignment: strand 2 runs 3' -> 5', so reverse its length dimension
        dna2_rev = torch.flip(dna2, dims=[1])

        # 1. Base-pair Watson-Crick match probability per position
        p_match = torch.einsum("bli,blj,ij->bl", dna1, dna2_rev, self.wc_matrix)  # (B, L)

        # 2. Doublet match probability: both position l and l+1 must be Watson-Crick pairs
        p_doublet = p_match[:, :-1] * p_match[:, 1:]  # (B, L-1)

        # 3. Dinucleotide step probabilities on strand 1 (5' -> 3')
        step1 = torch.einsum("bli,blj->blij", dna1[:, :-1], dna1[:, 1:])
        step_dH = torch.einsum("blij,ij->bl", step1, self.dH_matrix)
        step_dS = torch.einsum("blij,ij->bl", step1, self.dS_matrix)

        # 4. Accumulated doublet stacking enthalpy and entropy from matched base pairs
        # NOTE: For soft (probabilistic) DNA from Gumbel-Softmax, the product
        # step_dH * p_doublet introduces a mathematical approximation: strand-1
        # nucleotide probabilities appear in both terms, effectively weighting
        # contributions by p^2 rather than p. This is exact for hard one-hot
        # DNA (p \u2208 {0,1}) and is an upper-bound approximation for soft DNA.
        # A rigorous treatment would compute E[dH * I(WC)] jointly.
        raw_dH = (step_dH * p_doublet).sum(dim=-1)
        raw_dS = (step_dS * p_doublet).sum(dim=-1)
        corr_dS = self.salt_corrected_dS(raw_dS, L - 1)

        # 5. Base pair mismatch penalties across opposing aligned bases
        mismatches = torch.einsum("bli,blj,ij->b", dna1, dna2_rev, self.mismatch_matrix)

        # 6. Total Gibbs free energy with initiation: DeltaG = DeltaH - T * DeltaS / 1000 + mismatches
        total_dH = raw_dH + self.dH_init
        total_dS = corr_dS + self.dS_init
        dG = total_dH - temp_k * (total_dS / 1000.0) + mismatches
        affinity = -dG  # Higher affinity means stronger, more stable binding

        # 7. Equilibrium fraction bound theta(T) at standard strand concentration (100 nM)
        c0 = 1e-7  # 100 nM total strand concentration
        exponent = torch.clamp((dG / (GAS_CONSTANT_R * temp_k)) - np.log(c0 / 4.0), -40.0, 40.0)
        fraction_bound = 1.0 / (1.0 + torch.exp(exponent))

        return {
            "dH": total_dH,
            "dS": total_dS,
            "dG": dG,
            "mismatch_penalty": mismatches,
            "affinity": affinity,
            "fraction_bound": fraction_bound,
        }

    def forward(
        self,
        dna1: torch.Tensor,
        dna2: torch.Tensor,
        temp_k: Optional[float] = None,
    ) -> torch.Tensor:
        """Returns binding affinity (-DeltaG) for training rank alignment."""
        res = self.compute_duplex_thermodynamics(dna1, dna2, temp_k)
        return res["affinity"]

    def compute_melting_curve(
        self,
        dna1: torch.Tensor,
        dna2: torch.Tensor,
        temp_c_range: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates the multi-temperature Thermodynamic Semantic Spectrum theta(T).
        Args:
            dna1: (1, L, 4) query
            dna2: (1, L, 4) target
            temp_c_range: temperatures in Celsius (defaults to 40..85 C in 2.5 C steps)
        Returns:
            temps_c: numpy array of temperatures
            theta_curve: numpy array of fraction bound
        """
        if temp_c_range is None:
            temp_c_range = list(np.arange(40.0, 86.0, 2.5))

        thetas = []
        with torch.no_grad():
            for tc in temp_c_range:
                tk = tc + 273.15
                res = self.compute_duplex_thermodynamics(dna1, dna2, temp_k=tk)
                thetas.append(res["fraction_bound"].item())

        return np.array(temp_c_range), np.array(thetas)

    def estimate_hairpin_stability(self, dna: torch.Tensor, temp_k: float = 310.15) -> torch.Tensor:
        """
        Differentiable sliding-window intramolecular hairpin stability estimator.
        Evaluates SantaLucia (1998) doublet nearest-neighbor stacking thermodynamics along
        inverted repeat stems (stem length 4..8, loop length 3..5).
        Returns:
            hairpin_dG: (B,) negative value indicating stable self-hairpin folding (DeltaG_hairpin in kcal/mol),
                        or 0.0 for unstructured linear sequences.
        """
        B, L, _ = dna.shape
        best_dG = torch.zeros(B, device=dna.device)

        for stem_len in [4, 5, 6, 7, 8]:
            for loop_len in [3, 4, 5]:
                span = 2 * stem_len + loop_len
                if span > L:
                    continue
                for i in range(0, L - span + 1):
                    stem1 = dna[:, i : i + stem_len]
                    stem2 = dna[:, i + stem_len + loop_len : i + span]
                    stem2_rev = torch.flip(stem2, dims=[1])
                    matches = torch.einsum("bki,bkj,ij->bk", stem1, stem2_rev, self.wc_matrix)
                    p_doublet = matches[:, :-1] * matches[:, 1:]

                    # SantaLucia dinucleotide doublet nearest-neighbor stacking along stem
                    step_stem = torch.einsum("bki,bkj->bkij", stem1[:, :-1], stem1[:, 1:])
                    dH_stem = (torch.einsum("bkij,ij->bk", step_stem, self.dH_matrix) * p_doublet).sum(dim=-1)
                    dS_stem = (torch.einsum("bkij,ij->bk", step_stem, self.dS_matrix) * p_doublet).sum(dim=-1)
                    dG_stem = dH_stem - temp_k * (dS_stem / 1000.0)

                    # Jacobson-Stockmayer entropic loop penalty (SantaLucia 1998)
                    loop_penalty = 3.5 + 0.8 * float(np.log(loop_len))

                    # Full stem stability requirement: stable fold only if stem base pairs match
                    is_full_stem = (matches.sum(dim=-1) >= (stem_len - 1)).float()
                    dG_hp = (dG_stem + loop_penalty) * is_full_stem
                    best_dG = torch.minimum(best_dG, dG_hp)

        return best_dG

    def biological_constraints(self, dna: torch.Tensor) -> Dict[str, torch.Tensor]:

        """
        Computes biological synthesis viability metrics and penalties:
          1. GC-content deviation from ideal 40-60% window.
          2. Homopolymer runs >= 3 consecutive identical bases.
        """
        B, L, _ = dna.shape
        # GC ratio (target ideal 50% GC content)
        gc_fraction = (dna[:, :, 1] + dna[:, :, 2]).sum(dim=1) / float(L)
        gc_penalty = ((gc_fraction - 0.5) ** 2) * 50.0

        # Homopolymer runs (3 consecutive identical bases)
        hp = (dna[:, :-2] * dna[:, 1:-1] * dna[:, 2:]).sum(dim=(-1, -2))
        hp_penalty = hp * 3.0

        return {
            "gc_fraction": gc_fraction,
            "gc_penalty": gc_penalty,
            "homopolymer_count": hp,
            "homopolymer_penalty": hp_penalty,
            "total_bio_penalty": gc_penalty + hp_penalty,
        }
