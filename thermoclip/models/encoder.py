"""
Dual-Channel DNA Encoder with Epistemic Uncertainty Estimation.
Generates:
  - Channel A (Semantic Protospacer + Thermodynamic Hybridization Probe, 100 bp)
  - Channel B (Confidence / Abstention Molecular Beacon, 24 bp)
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

BASES = ["A", "C", "G", "T"]
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}


def dna_to_onehot(seq: str, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert nucleotide string to (L, 4) one-hot tensor."""
    indices = [BASE_TO_IDX.get(b.upper(), 0) for b in seq]
    t = torch.tensor(indices, dtype=torch.long, device=device)
    return F.one_hot(t, num_classes=4).float()


def onehot_to_dna(tensor: torch.Tensor) -> str:
    """Convert (L, 4) one-hot or probability tensor to 5'->3' DNA string."""
    idx = tensor.argmax(dim=-1).cpu().tolist()
    return "".join(BASES[i] for i in idx)


class ResidualBlock(nn.Module):
    """Residual MLP block with LayerNorm, GELU, and Dropout."""

    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = self.ln2(self.fc2(x))
        x = self.dropout(x)
        return x + residual


class DualChannelDNAEncoder(nn.Module):
    """
    Maps dense continuous sentence embeddings (e.g. 384-d) into:
      1. Channel A: Semantic DNA sequence (L_cas9 + L_hyb = 20 + 80 = 100 bp)
      2. Channel B: Confidence / Molecular Abstention Beacon (L_conf = 24 bp)
    
    Includes Monte Carlo Dropout for epistemic uncertainty quantification.
    """

    def __init__(
        self,
        in_dim: int = 384,
        hidden_dim: int = 512,
        cas9_len: int = 20,
        hyb_len: int = 80,
        conf_len: int = 24,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.cas9_len = cas9_len
        self.hyb_len = hyb_len
        self.sem_len = cas9_len + hyb_len  # 100 bp
        self.conf_len = conf_len           # 24 bp
        self.dropout_rate = dropout

        # Shared representation trunk
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res1 = ResidualBlock(hidden_dim, dropout=dropout)
        self.res2 = ResidualBlock(hidden_dim, dropout=dropout)

        # Head A: Semantic Channel (Cas9 protospacer + Hybridization probe)
        self.head_semantic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.sem_len * 4),
        )

        # Head B: Molecular Confidence / Abstention Channel
        self.head_conf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.conf_len * 4),
        )

        # Scalar confidence predictor (calibrated epistemic head)
        self.conf_scalar_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        tau: float = 1.0,
        hard: bool = True,
        deterministic: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Input embeddings (B, in_dim)
            tau: Gumbel-Softmax temperature
            hard: Whether to use Straight-Through discretization
            deterministic: If True, uses argmax one-hot projection
        Returns:
            dna_sem: (B, sem_len, 4) one-hot/relaxed tensor
            dna_conf: (B, conf_len, 4) one-hot/relaxed tensor
            conf_score: (B, 1) continuous confidence score in [0, 1]
        """
        if deterministic is None:
            deterministic = not self.training

        h = self.in_proj(x)
        h = self.res1(h)
        h = self.res2(h)

        logits_sem = self.head_semantic(h).view(-1, self.sem_len, 4)
        logits_conf = self.head_conf(h).view(-1, self.conf_len, 4)
        conf_score = self.conf_scalar_head(h)

        if not deterministic and self.training:
            # Straight-Through Gumbel-Softmax relaxation
            dna_sem = F.gumbel_softmax(logits_sem, tau=tau, hard=hard, dim=-1)
            dna_conf = F.gumbel_softmax(logits_conf, tau=tau, hard=hard, dim=-1)
        else:
            # Exact deterministic projection (no stochastic noise)
            idx_sem = logits_sem.argmax(dim=-1)
            dna_sem = F.one_hot(idx_sem, num_classes=4).float()

            idx_conf = logits_conf.argmax(dim=-1)
            dna_conf = F.one_hot(idx_conf, num_classes=4).float()

        return dna_sem, dna_conf, conf_score

    def estimate_uncertainty(
        self,
        x: torch.Tensor,
        num_mc_samples: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate epistemic uncertainty via Monte Carlo Dropout.
        Enables dropout during evaluation and computes predictive variance across passes.
        """
        was_training = self.training
        # Force dropout active for MC sampling
        self.train()

        mc_embs = []
        mc_conf_scores = []

        with torch.no_grad():
            for _ in range(num_mc_samples):
                h = self.in_proj(x)
                h = self.res1(h)
                h = self.res2(h)
                mc_embs.append(h.unsqueeze(0))
                mc_conf_scores.append(self.conf_scalar_head(h).unsqueeze(0))

        # Restore previous mode
        if not was_training:
            self.eval()

        mc_embs = torch.cat(mc_embs, dim=0)          # (M, B, hidden_dim)
        mc_conf_scores = torch.cat(mc_conf_scores, dim=0)  # (M, B, 1)

        # Epistemic variance across latent dimensions
        latent_var = mc_embs.var(dim=0).mean(dim=-1)       # (B,) mean variance across hidden units
        head_conf = mc_conf_scores.mean(dim=0).squeeze(-1)  # (B,)

        # Natural epistemic calibration:
        # In-domain: head_conf ~ 0.95, latent_var ~ 0.1 -> conf ~ 0.88 (Proceeds with retrieval)
        # OOD / Ambiguous: head_conf ~ 0.55, latent_var ~ 0.7 -> conf ~ 0.31 (Abstains)
        # Adversarial / Noise: head_conf ~ 0.05, latent_var ~ 1.5 -> conf ~ 0.02 (Abstains)
        epistemic_factor = torch.exp(-0.8 * torch.clamp(latent_var, min=0.0))
        final_conf = torch.clamp(head_conf * epistemic_factor, 0.01, 0.99)

        return {
            "epistemic_variance": latent_var,
            "mean_conf": head_conf,
            "std_conf": mc_conf_scores.std(dim=0).squeeze(-1),
            "calibrated_confidence": final_conf,
        }

    @staticmethod
    def to_query_probe(dna_sem: torch.Tensor) -> torch.Tensor:
        """
        Transforms encoded target sequence into the complementary query probe:
          - Cas9 guide segment (first 20 bp): guide sequence targeting the protospacer
          - Hybridization duplex probe (last 80 bp): antiparallel Watson-Crick complement
            (5'->3' reverse-complement so position k opposes target position (L-1-k))
        """
        cas9_part = dna_sem[:, :20]
        hyb_part = dna_sem[:, 20:]
        # Watson-Crick complement: A(0)->T(3), C(1)->G(2), G(2)->C(1), T(3)->A(0)
        hyb_wc = hyb_part[:, :, [3, 2, 1, 0]]
        # Antiparallel alignment: reverse along length
        hyb_probe = torch.flip(hyb_wc, dims=[1])
        return torch.cat([cas9_part, hyb_probe], dim=1)

