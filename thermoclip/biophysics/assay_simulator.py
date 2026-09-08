"""
Wet-Lab Grounding & Assay Simulator.
Simulates real in vitro biochemical measurements:
  - Hill equation optical fluorophore saturation
  - Secondary structure / steric hindrance attenuation
  - Poisson shot noise and baseline autofluorescence
  - Sequence-dependent background cross-talk
  - Synthesis error rate (in vitro oligo pool fidelity)
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from thermoclip.biophysics.thermodynamics import SantaLuciaDuplexModel
from thermoclip.biophysics.cas9 import Cas9CleavagePredictor


class WetLabAssaySimulator:
    """
    Simulates physical wet-lab assay measurements with realistic non-linearities,
    optical noise, and biophysical kinetics. Used as the simulated wet-lab measurement environment (in-silico surrogate)
    for Active Learning calibration.
    """

    def __init__(
        self,
        f_max: float = 1000.0,       # Maximum optical fluorophore intensity (RFU)
        f_bg: float = 25.0,          # Background autofluorescence baseline
        hill_coeff: float = 1.4,     # Cooperative binding / detection Hill coefficient
        noise_level: float = 0.05,   # Poisson/Gaussian relative measurement noise
        synthesis_error_rate: float = 0.008, # 0.8% synthesis deletion/mismatch rate
        device: Optional[torch.device] = None,
    ):
        self.f_max = f_max
        self.f_bg = f_bg
        self.hill_coeff = hill_coeff
        self.noise_level = noise_level
        self.synthesis_error_rate = synthesis_error_rate
        self.device = device or torch.device("cpu")

        self.thermo_model = SantaLuciaDuplexModel().to(self.device)
        self.cas9_model = Cas9CleavagePredictor().to(self.device)

    def introduce_synthesis_errors(self, dna: torch.Tensor) -> torch.Tensor:
        """Stochastically simulate sequence synthesis infidelity in oligo pools."""
        if self.synthesis_error_rate <= 0.0:
            return dna

        B, L, C = dna.shape
        error_mask = torch.rand((B, L), device=dna.device) < self.synthesis_error_rate
        if not error_mask.any():
            return dna

        # For erroneous positions, substitute with random nucleotide
        random_sub = F.one_hot(torch.randint(0, 4, (B, L), device=dna.device), num_classes=4).float()
        dna_corrupted = torch.where(error_mask.unsqueeze(-1), random_sub, dna)
        return dna_corrupted

    def measure_assay_yield(
        self,
        query_sem: torch.Tensor,
        target_sem: torch.Tensor,
        add_noise: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate experimental wet-lab assay measurement for (query, target) pairs.
        Args:
            query_sem: (B, 100, 4) query DNA (20 Cas9 + 80 Hyb)
            target_sem: (B, 100, 4) target DNA
            add_noise: Whether to add realistic experimental measurement noise
        Returns:
            Dict of wet-lab measured physical variables (Cas9 cleavage yield, fluorescence RFU, calibrated physical yield in [0, 1])
        """
        B = query_sem.size(0)

        # Separate Cas9 guide/protospacer (0..20) and hybridization duplex probe (20..100)
        q_cas9, q_hyb = query_sem[:, :20], query_sem[:, 20:]
        t_cas9, t_hyb = target_sem[:, :20], target_sem[:, 20:]

        # Apply synthesis error perturbation
        q_hyb_synth = self.introduce_synthesis_errors(q_hyb)
        t_hyb_synth = self.introduce_synthesis_errors(t_hyb)

        # 1. Cas9 cleavage efficiency
        cleave_prob = self.cas9_model(q_cas9, t_cas9)  # (B,)

        # 2. Hybridization thermodynamics
        thermo_res = self.thermo_model.compute_duplex_thermodynamics(q_hyb_synth, t_hyb_synth)
        ideal_fraction = thermo_res["fraction_bound"]  # (B,) in [0, 1]

        # 3. Steric / secondary structure attenuation
        hairpin_q = self.thermo_model.estimate_hairpin_stability(q_hyb)
        hairpin_t = self.thermo_model.estimate_hairpin_stability(t_hyb)
        steric_factor = torch.exp(0.03 * (hairpin_q + hairpin_t))
        effective_fraction = torch.clamp(ideal_fraction * steric_factor, 0.0, 1.0)

        # 4. Optical fluorophore non-linear response (Hill kinetics)
        hill_pow = effective_fraction ** self.hill_coeff
        rfu = self.f_bg + self.f_max * (hill_pow / (0.35 ** self.hill_coeff + hill_pow + 1e-8))

        # 5. Combined Dual-Chemistry Yield:
        # Weighted linear combination of Cas9 cleavage and hybridized fluorescence fraction
        combined_yield = torch.clamp(0.4 * cleave_prob + 0.6 * (rfu / self.f_max), 0.0, 1.0)

        # 6. Experimental noise injection (Poisson shot noise + Gaussian detector jitter)
        if add_noise:
            # Poisson-like standard deviation scaling with sqrt(yield)
            shot_std = self.noise_level * torch.sqrt(combined_yield + 0.05)
            gaussian_jitter = torch.randn_like(combined_yield) * shot_std
            measured_yield = torch.clamp(combined_yield + gaussian_jitter, 0.0, 1.0)
            measured_rfu = torch.clamp(rfu + torch.randn_like(rfu) * (self.noise_level * rfu), min=0.0)
        else:
            measured_yield = combined_yield
            measured_rfu = rfu

        return {
            "cleavage_probability": cleave_prob,
            "hybridization_affinity": thermo_res["affinity"],
            "fraction_bound": effective_fraction,
            "simulated_optical_rfu": measured_rfu,
            "simulated_physical_yield": measured_yield,
        }
