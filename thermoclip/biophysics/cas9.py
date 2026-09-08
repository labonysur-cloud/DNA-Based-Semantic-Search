"""
Biophysical Model of CRISPR-SpCas9 Off-Target Cleavage Kinetics.
Implements position-dependent mismatch tolerance (CFD / Hsu profile)
serving as Stage 1 coarse molecular candidate generation.
NOTE: This is a Cas9-inspired sequence compatibility surrogate, not a faithful implementation of Doench et al. (2016) CFD.
- The position weights are synthetic heuristic values, not empirical CFD lookup values
- The mismatch matrix uses a simplified factored model instead of the full positionxMismatch-type CFD table
- The PAM parameter exists but is not used anywhere in the pipeline
- The scoring function uses exponential decay of weighted sums rather than the multiplicative CFD model
"""

from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Empirical position-dependent mismatch weights (Hsu et al., 2013; Doench et al., 2016 CFD)
# Index 0 is position 20 (PAM-distal), index 19 is position 1 (PAM-proximal seed)
# Seed region (proximal to PAM, positions 1-8 / indices 12-19) is highly sensitive to mismatches.
DEFAULT_POSITION_WEIGHTS = [
    0.15, 0.18, 0.20, 0.22, 0.25,  # pos 20..16 (distal, tolerant)
    0.35, 0.40, 0.45, 0.50, 0.55,  # pos 15..11 (intermediate)
    0.70, 0.85, 1.10, 1.30, 1.50,  # pos 10..6  (seed entry)
    1.75, 2.00, 2.25, 2.50, 2.80,  # pos 5..1   (core seed, hyper-sensitive)
]


class Cas9CleavagePredictor(nn.Module):
    """
    Differentiable biophysical surrogate for Cas9 off-target cleavage efficiency.
    Calculates cleavage probability P_cleave in [0, 1] between 20-nt guide RNA and target protospacer.
    NOTE: This is a Cas9-inspired sequence compatibility surrogate.
    """

    def __init__(
        self,
        protospacer_len: int = 20,
        position_weights: Optional[List[float]] = None,
        steepness: float = 1.0,
    ):
        super().__init__()
        self.protospacer_len = protospacer_len
        self.steepness = steepness

        if position_weights is None:
            position_weights = DEFAULT_POSITION_WEIGHTS
        assert len(position_weights) == protospacer_len

        # Register position weights as buffer
        pw_tensor = torch.tensor(position_weights, dtype=torch.float32).view(1, protospacer_len)
        self.register_buffer("pos_weights", pw_tensor)

        # Base mismatch type severity matrix (rows: guide base, cols: target base)
        # Bases: A, C, G, T. Watson-Crick match (A-T, C-G, etc.) is handled via complementary mapping.
        # Here we model direct aligned match vs mismatch penalties:
        # A mismatch with rG:dT wobble is less severe than rC:dC clash.
        mismatch_matrix = torch.tensor([
            [0.0, 1.2, 1.4, 0.8],  # A guide vs target [A, C, G, T]
            [1.3, 0.0, 1.5, 1.1],  # C guide
            [0.7, 1.4, 0.0, 0.5],  # G guide (G-T wobble is tolerated)
            [0.9, 1.2, 0.6, 0.0],  # T/U guide
        ], dtype=torch.float32)
        self.register_buffer("mismatch_severity", mismatch_matrix)

    def forward(
        self,
        guide_dna: torch.Tensor,
        target_dna: torch.Tensor,
        target_pam: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Differentiable computation of cleavage probability using position weights and CFD mismatch severity.
        Args:
            guide_dna: (B, 20, 4) one-hot or relaxed tensor (5' -> 3')
            target_dna: (B, 20, 4) one-hot or relaxed tensor
            target_pam: Optional (B, 3, 4) one-hot or relaxed tensor representing 5'-NGG-3' PAM
        Returns:
            cleavage_prob: (B,) tensor in [0, 1]
        """
        # Base-specific mismatch severity across all 20 protospacer positions
        # Evaluates specific base pairing cost (e.g. rG:dT wobble ~0.5 vs rC:dC clash ~1.5)
        mismatch_costs = torch.einsum("bli,blj,ij->bl", guide_dna, target_dna, self.mismatch_severity)

        # Weighted penalty across the 20 positions (seed region has higher weight)
        weighted_penalties = (mismatch_costs * self.pos_weights).sum(dim=-1)  # (B,)

        # PAM compatibility factor (canonical 5'-NGG-3': positions 1 and 2 must be Guanine)
        if target_pam is not None:
            # Guanine is index 2
            pam_score = target_pam[:, 1, 2] * target_pam[:, 2, 2]
            pam_factor = torch.clamp(pam_score, 0.05, 1.0)
        else:
            pam_factor = 1.0

        # Cleavage probability follows exponential decay with accumulated mismatch penalty
        cleavage_prob = pam_factor * torch.exp(-self.steepness * weighted_penalties)
        return torch.clamp(cleavage_prob, 0.0, 1.0)

    def batch_candidate_generation(
        self,
        query_guide: torch.Tensor,
        target_library: torch.Tensor,
        top_k: int = 50,
        threshold: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulates Stage 1 candidate generation from a large molecular library.
        Args:
            query_guide: (1, 20, 4) query guide protospacer
            target_library: (N, 20, 4) candidate library
            top_k: Maximum number of candidates to advance to Stage 2
            threshold: Minimum cleavage probability threshold
        Returns:
            candidate_indices: (K,) selected library indices
            cleavage_scores: (K,) cleavage probabilities
        """
        N = target_library.size(0)
        expanded_query = query_guide.expand(N, -1, -1)
        with torch.no_grad():
            scores = self.forward(expanded_query, target_library)

        # Filter by threshold and take top_k
        valid_mask = scores >= threshold
        valid_indices = torch.nonzero(valid_mask).squeeze(-1)

        if len(valid_indices) == 0:
            # Fallback to top_k if no candidate meets hard threshold
            top_scores, top_idx = torch.topk(scores, min(top_k, N))
            return top_idx, top_scores

        valid_scores = scores[valid_indices]
        if len(valid_scores) > top_k:
            top_sub_scores, sub_idx = torch.topk(valid_scores, top_k)
            return valid_indices[sub_idx], top_sub_scores
        else:
            sorted_scores, sort_idx = torch.sort(valid_scores, descending=True)
            return valid_indices[sort_idx], sorted_scores
