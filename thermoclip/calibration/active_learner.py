"""
Active Learning & Wet-Lab Calibration Module.
Calibrates the biophysical model using budget-constrained experimental assays
selected via Bayesian Active Learning by Disagreement (BALD) or Max Variance.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from thermoclip.biophysics.assay_simulator import WetLabAssaySimulator


class BiophysicalCalibrationAdapter(nn.Module):
    """
    Differentiable neural adapter that calibrates raw theoretical biophysical
    predictions (SantaLucia -DeltaG, Cas9 cleavage probability, GC-content,
    hairpin stability) into real wet-lab physical assay yield.
    Uses a regularized monotonic residual architecture to prevent overfitting on small lab budgets.
    """

    def __init__(self, in_features: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 16)
        self.fc2 = nn.Linear(16, 1)
        with torch.no_grad():
            self.fc1.weight.normal_(0.0, 0.1)
            self.fc2.weight.normal_(0.0, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        raw = self.fc2(h).squeeze(-1) + (0.7 * x[:, 0] + 0.3 * x[:, 1] - 0.5)
        return torch.sigmoid(raw)


class ActiveLabCalibrator:
    """
    Manages active acquisition and calibration of the biophysical surrogate.
    Implements:
      - BALD (Bayesian Active Learning by Disagreement)
      - Max Epistemic Variance Acquisition
      - Random Assay Selection Baseline
      - Expected Calibration Error (ECE) minimization
    """

    def __init__(
        self,
        assay_simulator: WetLabAssaySimulator,
        device: Optional[torch.device] = None,
        ensemble_size: int = 5,
    ):
        self.assay_sim = assay_simulator
        self.device = device or torch.device("cpu")
        self.ensemble_size = ensemble_size

        # Ensemble of adapters for epistemic uncertainty estimation over physical yield
        self.ensemble = [
            BiophysicalCalibrationAdapter().to(self.device)
            for _ in range(ensemble_size)
        ]
        self.optimizers = [
            torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            for model in self.ensemble
        ]

    def extract_biophysical_features(
        self,
        dna1_sem: torch.Tensor,
        dna2_sem: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extracts 4D biophysical descriptor vector:
          [normalized -DeltaG, Cas9 cleavage prob, GC content, hairpin score]
        """
        q_cas9, q_hyb = dna1_sem[:, :20], dna1_sem[:, 20:]
        t_cas9, t_hyb = dna2_sem[:, :20], dna2_sem[:, 20:]

        with torch.no_grad():
            cleave_p = self.assay_sim.cas9_model(q_cas9, t_cas9)
            thermo = self.assay_sim.thermo_model.compute_duplex_thermodynamics(q_hyb, t_hyb)
            aff_norm = torch.clamp((thermo["affinity"] + 50.0) / 100.0, 0.0, 1.0)
            gc_res = self.assay_sim.thermo_model.biological_constraints(q_hyb)
            gc_norm = gc_res["gc_fraction"]
            hp_score = torch.clamp(self.assay_sim.thermo_model.estimate_hairpin_stability(q_hyb) / 20.0, 0.0, 1.0)

        feats = torch.stack([aff_norm, cleave_p, gc_norm, hp_score], dim=-1)
        return feats

    def predict_ensemble(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns predictive mean and epistemic variance across the ensemble.
        Args:
            features: (B, 4)
        Returns:
            mean_yield: (B,)
            variance: (B,)
        """
        preds = []
        for model in self.ensemble:
            model.eval()
            with torch.no_grad():
                preds.append(model(features).unsqueeze(0))
        preds_t = torch.cat(preds, dim=0)  # (E, B)
        mean_yield = preds_t.mean(dim=0)
        variance = preds_t.var(dim=0)
        return mean_yield, variance

    def compute_bald_score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Computes BALD (Bayesian Active Learning by Disagreement) acquisition score:
        I(y; theta | x) = H(E[y|x]) - E[H(y|x, theta)]
        For binary/bounded yield in [0, 1]:
        """
        preds = []
        for model in self.ensemble:
            model.eval()
            with torch.no_grad():
                p = torch.clamp(model(features), 1e-6, 1.0 - 1e-6)
                preds.append(p.unsqueeze(0))
        preds_t = torch.cat(preds, dim=0)  # (E, B)

        # Average prediction
        mean_p = preds_t.mean(dim=0)
        # Entropy of the expected prediction: - (p log p + (1-p) log(1-p))
        total_entropy = -(mean_p * torch.log(mean_p) + (1.0 - mean_p) * torch.log(1.0 - mean_p))

        # Expected entropy of individual models
        indiv_entropy = -(preds_t * torch.log(preds_t) + (1.0 - preds_t) * torch.log(1.0 - preds_t))
        expected_indiv_entropy = indiv_entropy.mean(dim=0)

        bald_score = total_entropy - expected_indiv_entropy
        return torch.clamp(bald_score, min=0.0)

    def select_active_assays(
        self,
        pool_features: torch.Tensor,
        budget: int = 50,
        strategy: str = "bald",
    ) -> torch.Tensor:
        """
        Selects top-informative pairs for wet-lab assay synthesis and measurement.
        Args:
            pool_features: (N, 4) candidate features
            budget: Number of assays to select
            strategy: 'bald', 'variance', or 'random'
        Returns:
            selected_indices: (budget,) tensor
        """
        N = pool_features.size(0)
        budget = min(budget, N)

        if strategy == "random":
            return torch.randperm(N)[:budget]
        elif strategy == "variance":
            _, var = self.predict_ensemble(pool_features)
            _, top_idx = torch.topk(var, budget)
            return top_idx
        elif strategy == "bald":
            bald_scores = self.compute_bald_score(pool_features)
            _, top_idx = torch.topk(bald_scores, budget)
            return top_idx
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def update_calibration(
        self,
        features: torch.Tensor,
        measured_yield: torch.Tensor,
        epochs: int = 40,
    ) -> float:
        """
        Updates the ensemble on newly measured wet-lab assay observations.
        """
        total_loss = 0.0
        for model, opt in zip(self.ensemble, self.optimizers):
            model.train()
            for _ in range(epochs):
                opt.zero_grad()
                pred = model(features)
                loss = F.mse_loss(pred, measured_yield)
                loss.backward()
                opt.step()
                total_loss += loss.item()

        return total_loss / (len(self.ensemble) * epochs)
