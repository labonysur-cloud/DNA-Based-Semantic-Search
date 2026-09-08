"""
Dual-Chemistry Molecular Cascade Retriever.
Executes:
  Stage 0: Epistemic Uncertainty & Abstention Check
  Stage 1: Cas9 High-Throughput Coarse Candidate Generation
  Stage 2: SantaLucia Thermodynamic Hybridization Fine Reranking
  Stage 3: Calibrated Biophysical Yield Prediction
"""

from typing import Dict, List, Tuple, Optional
import time
import torch
import torch.nn as nn
from thermoclip.models.encoder import DualChannelDNAEncoder
from thermoclip.biophysics.cas9 import Cas9CleavagePredictor
from thermoclip.biophysics.thermodynamics import SantaLuciaDuplexModel
from thermoclip.models.abstention import MolecularAbstentionGate
from thermoclip.calibration.active_learner import ActiveLabCalibrator


class DualChemistryCascadeRetriever:
    """
    End-to-End Molecular Semantic Search Engine combining Cas9 coarse filtering
    and SantaLucia thermodynamic hybridization reranking with molecular abstention.
    """

    def __init__(
        self,
        encoder: DualChannelDNAEncoder,
        cas9_predictor: Cas9CleavagePredictor,
        thermo_model: SantaLuciaDuplexModel,
        abstention_gate: MolecularAbstentionGate,
        active_calibrator: Optional[ActiveLabCalibrator] = None,
        device: Optional[torch.device] = None,
    ):
        self.encoder = encoder
        self.cas9 = cas9_predictor
        self.thermo = thermo_model
        self.abstention_gate = abstention_gate
        self.calibrator = active_calibrator
        self.device = device or torch.device("cpu")

    def retrieve(
        self,
        query_emb: torch.Tensor,
        library_dna: torch.Tensor,
        top_k_coarse: int = 150,
        top_k_final: int = 10,
        temperature_c: float = 65.0,
        mode: str = "cascade",  # 'cascade', 'cas9_only', 'hyb_only'
        enforce_abstention: bool = False,
        precomputed_confidence: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        Execute molecular search.
        Args:
            query_emb: (1, in_dim) dense query embedding
            library_dna: (N, 100, 4) target DNA library
            top_k_coarse: Stage 1 candidate pool size
            top_k_final: Stage 2 final ranking return size
            temperature_c: Hybridization operating temperature
            mode: Search mode ('cascade', 'cas9_only', 'hyb_only')
            enforce_abstention: If True and query is uncertain, suppresses retrieval (returns empty list)
            precomputed_confidence: Optional cached confidence score to avoid redundant MC-dropout
        Returns:
            Dict containing retrieved indices, scores, timing, and abstention status
        """
        t0 = time.perf_counter()
        N = library_dna.size(0)

        # 1. Encode query
        self.encoder.eval()
        with torch.no_grad():
            q_sem, q_conf, conf_score = self.encoder(query_emb, deterministic=True)
            if precomputed_confidence is not None:
                confidence = float(precomputed_confidence)
            else:
                uncert_info = self.encoder.estimate_uncertainty(query_emb, num_mc_samples=10)
                confidence = float(uncert_info["calibrated_confidence"].item())

        # 2. Stage 0: Molecular Uncertainty & Abstention Check
        abstention_eval = self.abstention_gate.evaluate_abstention(torch.tensor([confidence], device=self.device))
        should_abstain = bool(abstention_eval["abstain_mask"].item())

        if should_abstain and enforce_abstention:
            elapsed = time.perf_counter() - t0
            return {
                "abstained": True,
                "confidence": confidence,
                "top_indices": [],
                "top_scores": [],
                "mode": mode,
                "elapsed_sec": elapsed,
                "candidates_filtered": 0,
                "pool_reduction_factor": 1.0,
            }

        q_probe = self.encoder.to_query_probe(q_sem)
        q_cas9 = q_probe[:, :20]
        q_hyb = q_probe[:, 20:]
        lib_cas9 = library_dna[:, :20]
        lib_hyb = library_dna[:, 20:]

        # 3. Stage 1 & Stage 2 Execution by Mode
        if mode == "cas9_only":
            # Direct Cas9 cleavage ranking across all N targets
            with torch.no_grad():
                q_cas9_exp = q_cas9.expand(N, -1, -1)
                scores = self.cas9(q_cas9_exp, lib_cas9)
                top_scores, top_idx = torch.topk(scores, min(top_k_final, N))
            elapsed = time.perf_counter() - t0
            return {
                "abstained": should_abstain,
                "confidence": confidence,
                "top_indices": top_idx.cpu().tolist(),
                "top_scores": top_scores.cpu().tolist(),
                "mode": mode,
                "elapsed_sec": elapsed,
                "candidates_filtered": N,
                "pool_reduction_factor": 1.0,
            }

        elif mode == "hyb_only":
            # Direct full-pool thermodynamic hybridization (expensive in wet-lab / compute)
            with torch.no_grad():
                q_hyb_exp = q_hyb.expand(N, -1, -1)
                res = self.thermo.compute_duplex_thermodynamics(
                    q_hyb_exp, lib_hyb, temp_k=temperature_c + 273.15
                )
                scores = res["affinity"]
                top_scores, top_idx = torch.topk(scores, min(top_k_final, N))
            elapsed = time.perf_counter() - t0
            return {
                "abstained": should_abstain,
                "confidence": confidence,
                "top_indices": top_idx.cpu().tolist(),
                "top_scores": top_scores.cpu().tolist(),
                "mode": mode,
                "elapsed_sec": elapsed,
                "candidates_filtered": N,
                "pool_reduction_factor": 1.0,
            }

        elif mode == "cascade":
            # STAGE 1: Cas9 Coarse Candidate Generation
            coarse_k = min(top_k_coarse, N)
            with torch.no_grad():
                q_cas9_exp = q_cas9.expand(N, -1, -1)
                cleave_scores = self.cas9(q_cas9_exp, lib_cas9)
                top_cleave_scores, candidate_idx = torch.topk(cleave_scores, coarse_k)

            # STAGE 2: Thermodynamic Hybridization Fine Reranking on Candidates
            cand_lib_hyb = lib_hyb[candidate_idx]
            cand_lib_full = library_dna[candidate_idx]
            q_hyb_exp = q_hyb.expand(coarse_k, -1, -1)
            q_sem_exp = q_probe.expand(coarse_k, -1, -1)

            with torch.no_grad():
                thermo_res = self.thermo.compute_duplex_thermodynamics(
                    q_hyb_exp, cand_lib_hyb, temp_k=temperature_c + 273.15
                )
                hyb_affinity = thermo_res["affinity"]

                # Dual-chemistry synergy: combine Cas9 cleavage probability and hybridization affinity
                norm_hyb = torch.clamp((hyb_affinity + 50.0) / 100.0, 0.0, 1.0)
                final_cand_scores = 0.40 * top_cleave_scores + 0.60 * norm_hyb


                final_k = min(top_k_final, coarse_k)
                top_sub_scores, sub_idx = torch.topk(final_cand_scores, final_k)
                final_idx = candidate_idx[sub_idx]

            elapsed = time.perf_counter() - t0
            reduction_factor = float(N) / float(coarse_k)

            return {
                "abstained": should_abstain,
                "confidence": confidence,
                "top_indices": final_idx.cpu().tolist(),
                "top_scores": top_sub_scores.cpu().tolist(),
                "mode": mode,
                "elapsed_sec": elapsed,
                "candidates_filtered": coarse_k,
                "pool_reduction_factor": reduction_factor,
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")
