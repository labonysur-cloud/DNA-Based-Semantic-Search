"""
Molecular Uncertainty and Physical Abstention Gate.
Encodes confidence into Channel B (orthogonal molecular beacon / hairpin clamp)
and implements selective retrieval (Retrieve vs Abstain) to eliminate confident
false positives on out-of-distribution, ambiguous, or adversarial queries.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Predefined stable hairpin clamp sequence for molecular abstention:
# Self-complementary GC-rich stem that folds spontaneously (DeltaG < -20 kcal/mol)
# sterically silencing enzymatic cleavage and duplex hybridization in solution.
MOLECULAR_CLAMP_HAIRPIN = "CGCGCGCTAATTCGCGCGCG"
PERMISSIVE_BEACON_LINKER = "TAATACTATATAATACTATA"


class MolecularAbstentionGate(nn.Module):
    """
    Evaluates epistemic confidence and controls the molecular abstention switch.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.55,
        clamp_hairpin_seq: str = MOLECULAR_CLAMP_HAIRPIN,
        permissive_seq: str = PERMISSIVE_BEACON_LINKER,
    ):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.clamp_hairpin_seq = clamp_hairpin_seq
        self.permissive_seq = permissive_seq

    def evaluate_abstention(
        self,
        confidence_scores: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Determines whether each query should proceed with retrieval or abstain.
        Args:
            confidence_scores: (B,) tensor in [0, 1]
            threshold: Optional custom threshold (defaults to self.confidence_threshold)
        Returns:
            Dict containing:
              - 'abstain_mask': (B,) bool tensor (True if model abstains)
              - 'retrieve_mask': (B,) bool tensor (True if model retrieves)
              - 'coverage': scalar fraction of retained queries
        """
        tau = threshold if threshold is not None else self.confidence_threshold
        abstain_mask = confidence_scores < tau
        retrieve_mask = ~abstain_mask
        coverage = retrieve_mask.float().mean()

        return {
            "abstain_mask": abstain_mask,
            "retrieve_mask": retrieve_mask,
            "coverage": coverage,
            "threshold": torch.tensor(tau),
        }

    def generate_molecular_channel_b(
        self,
        confidence_scores: torch.Tensor,
        dna_conf_predicted: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Physically programs Channel B:
        - If confident (retrieve): preserves predicted beacon sequence.
        - If unconfident (abstain): overrides with high-stability molecular clamp hairpin.
        """
        B = confidence_scores.size(0)
        abstention_info = self.evaluate_abstention(confidence_scores)
        abstain_mask = abstention_info["abstain_mask"]

        # Clamp hairpin one-hot
        from thermoclip.models.encoder import dna_to_onehot, onehot_to_dna
        clamp_tensor = dna_to_onehot(self.clamp_hairpin_seq, device=confidence_scores.device)
        # Pad to conf_len if needed
        target_len = dna_conf_predicted.size(1)
        if clamp_tensor.size(0) < target_len:
            pad = torch.zeros((target_len - clamp_tensor.size(0), 4), device=confidence_scores.device)
            pad[:, 0] = 1.0  # pad with A
            clamp_tensor = torch.cat([clamp_tensor, pad], dim=0)
        elif clamp_tensor.size(0) > target_len:
            clamp_tensor = clamp_tensor[:target_len]

        clamp_batch = clamp_tensor.unsqueeze(0).expand(B, -1, -1)
        final_channel_b = torch.where(abstain_mask.unsqueeze(-1).unsqueeze(-1), clamp_batch, dna_conf_predicted)

        sequences = [onehot_to_dna(final_channel_b[i]) for i in range(B)]
        return final_channel_b, sequences

    @staticmethod
    def compute_selective_retrieval_curve(
        confidences: np.ndarray,
        retrieval_errors: np.ndarray,
        num_thresholds: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        Computes Risk-Coverage trade-off curve:
        Risk = average error among non-abstained queries.
        Coverage = fraction of non-abstained queries.
        """
        thresholds = np.linspace(0.0, 1.0, num_thresholds)
        used_thresholds = []
        coverages = []
        risks = []

        for tau in thresholds:
            active = confidences >= tau
            if active.sum() == 0:
                continue
            used_thresholds.append(tau)
            cov = float(active.mean())
            risk = float(retrieval_errors[active].mean())
            coverages.append(cov)
            risks.append(risk)

        return {
            "thresholds": np.array(used_thresholds),
            "coverage": np.array(coverages),
            "risk": np.array(risks),
        }
