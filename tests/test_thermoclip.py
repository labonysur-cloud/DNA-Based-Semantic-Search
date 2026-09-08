"""
Comprehensive Unit and Integration Test Suite for ThermoCLIP-DNA.
Tests:
  1. Dual-Channel DNA Encoder and MC-Dropout uncertainty estimation
  2. Cas9 off-target cleavage kinetics and coarse candidate filtering
  3. SantaLucia nearest-neighbor thermodynamic duplex model and melting curves
  4. Wet-lab assay simulator and non-linear optical fluorophore saturation
  5. Molecular abstention gate and hairpin clamp synthesis
  6. Active learning acquisition (BALD / variance) and biophysical calibration
  7. Dual-chemistry cascade retriever end-to-end
  8. Scientific evaluation metrics (Recall@k, NDCG@k, ECE, AUROC, viability)
"""

import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn.functional as F

from thermoclip.models.encoder import DualChannelDNAEncoder, dna_to_onehot, onehot_to_dna
from thermoclip.biophysics.cas9 import Cas9CleavagePredictor
from thermoclip.biophysics.thermodynamics import SantaLuciaDuplexModel
from thermoclip.biophysics.assay_simulator import WetLabAssaySimulator
from thermoclip.models.abstention import MolecularAbstentionGate
from thermoclip.calibration.active_learner import ActiveLabCalibrator
from thermoclip.search.cascade import DualChemistryCascadeRetriever
from thermoclip.evaluation.metrics import (
    recall_at_k, ndcg_at_k, mean_reciprocal_rank,
    expected_calibration_error, brier_score, compute_ood_auroc,
    evaluate_sequence_viability, compute_reliability_diagram_data,
    compute_risk_coverage_curve
)


class TestThermoCLIPDNA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.encoder = DualChannelDNAEncoder(in_dim=64, hidden_dim=64, cas9_len=20, hyb_len=80, conf_len=24).to(cls.device)
        cls.cas9 = Cas9CleavagePredictor().to(cls.device)
        cls.thermo = SantaLuciaDuplexModel(seq_len=80).to(cls.device)
        cls.assay_sim = WetLabAssaySimulator(device=cls.device)
        cls.abstention_gate = MolecularAbstentionGate(confidence_threshold=0.55)
        cls.calibrator = ActiveLabCalibrator(cls.assay_sim, device=cls.device, ensemble_size=3)

    def test_encoder_shapes_and_modes(self):
        """Test dual-channel output dimensions and deterministic vs training behaviors."""
        x = torch.randn(4, 64)
        self.encoder.train()
        sem, conf, conf_score = self.encoder(x, tau=1.0, hard=True, deterministic=False)
        self.assertEqual(sem.shape, (4, 100, 4))
        self.assertEqual(conf.shape, (4, 24, 4))
        self.assertEqual(conf_score.shape, (4, 1))

        # Check deterministic evaluation mode
        self.encoder.eval()
        with torch.no_grad():
            sem_det, conf_det, _ = self.encoder(x, deterministic=True)
        # Check exact one-hot property (each row sums to 1.0)
        self.assertTrue(torch.allclose(sem_det.sum(dim=-1), torch.ones(4, 100)))
        self.assertTrue(torch.allclose(conf_det.sum(dim=-1), torch.ones(4, 24)))

    def test_mc_dropout_uncertainty(self):
        """Test Monte Carlo Dropout uncertainty estimation."""
        x = torch.randn(5, 64)
        uncert = self.encoder.estimate_uncertainty(x, num_mc_samples=6)
        self.assertIn("epistemic_variance", uncert)
        self.assertIn("calibrated_confidence", uncert)
        self.assertEqual(uncert["calibrated_confidence"].shape, (5,))
        self.assertTrue((uncert["calibrated_confidence"] >= 0.0).all())
        self.assertTrue((uncert["calibrated_confidence"] <= 1.0).all())

    def test_dna_conversion_helpers(self):
        """Test nucleotide string <-> one-hot tensor roundtrip."""
        seq = "ACGTACGT"
        oh = dna_to_onehot(seq)
        self.assertEqual(oh.shape, (8, 4))
        recovered = onehot_to_dna(oh)
        self.assertEqual(seq, recovered)

    def test_cas9_cleavage_kinetics(self):
        """Test Cas9 cleavage probability and position sensitivity (seed vs distal)."""
        guide = dna_to_onehot("A" * 20).unsqueeze(0)
        exact_target = dna_to_onehot("A" * 20).unsqueeze(0)
        distal_mm_target = dna_to_onehot("C" + "A" * 19).unsqueeze(0)    # pos 20 mismatch (distal)
        seed_mm_target = dna_to_onehot("A" * 19 + "C").unsqueeze(0)      # pos 1 mismatch (seed)

        with torch.no_grad():
            p_exact = self.cas9(guide, exact_target).item()
            p_distal = self.cas9(guide, distal_mm_target).item()
            p_seed = self.cas9(guide, seed_mm_target).item()

        self.assertAlmostEqual(p_exact, 1.0, places=3)
        self.assertGreater(p_exact, p_distal)
        self.assertGreater(p_distal, p_seed)  # Seed mismatch penalizes cleavage far more than distal

    def test_thermodynamic_duplex_model(self):
        """Test SantaLucia nearest-neighbor thermodynamics and melting curves."""
        # 80-bp query and target
        q = dna_to_onehot("ACGT" * 20).unsqueeze(0)
        # Antiparallel complementary target
        wc_map = {"A": "T", "T": "A", "C": "G", "G": "C"}
        q_str = "ACGT" * 20
        # In antiparallel duplex, strand 2 (5'->3') reversed opposes strand 1
        t_str = "".join(wc_map[b] for b in q_str)[::-1]
        t = dna_to_onehot(t_str).unsqueeze(0)

        with torch.no_grad():
            res_match = self.thermo.compute_duplex_thermodynamics(q, t)
            # Create a heavily mismatched target
            bad_t = dna_to_onehot("AAAA" * 20).unsqueeze(0)
            res_mismatch = self.thermo.compute_duplex_thermodynamics(q, bad_t)

        self.assertGreater(res_match["affinity"].item(), res_mismatch["affinity"].item())
        self.assertGreater(res_match["fraction_bound"].item(), res_mismatch["fraction_bound"].item())

        # Test melting curve computation
        temps, thetas = self.thermo.compute_melting_curve(q, t, [50.0, 65.0, 80.0])
        self.assertEqual(len(temps), 3)
        self.assertGreaterEqual(thetas[0], thetas[2])  # Fraction bound decreases with temperature

    def test_wet_lab_assay_simulator(self):
        """Test physical assay simulation with optical noise and Hill saturation."""
        q = dna_to_onehot("ACGT" * 25).unsqueeze(0)  # 100 bp
        t = dna_to_onehot("ACGT" * 25).unsqueeze(0)

        res = self.assay_sim.measure_assay_yield(q, t, add_noise=True)
        self.assertIn("measured_physical_yield", res)
        self.assertIn("optical_rfu", res)
        self.assertTrue(0.0 <= res["measured_physical_yield"].item() <= 1.0)
        self.assertGreater(res["optical_rfu"].item(), 0.0)

    def test_molecular_abstention_gate(self):
        """Test selective abstention and clamp hairpin synthesis."""
        confs = torch.tensor([0.90, 0.40, 0.70, 0.20])
        eval_res = self.abstention_gate.evaluate_abstention(confs, threshold=0.55)
        self.assertEqual(eval_res["abstain_mask"].tolist(), [False, True, False, True])
        self.assertAlmostEqual(eval_res["coverage"].item(), 0.50)

        # Test Channel B clamp insertion
        pred_conf_dna = torch.zeros(4, 24, 4)
        pred_conf_dna[:, :, 0] = 1.0  # all A's
        final_b, seqs = self.abstention_gate.generate_molecular_channel_b(confs, pred_conf_dna)
        self.assertEqual(len(seqs), 4)
        # Abstained sequences should contain the clamp sequence prefix
        self.assertTrue(seqs[1].startswith("CGCGCGC"))
        self.assertTrue(seqs[3].startswith("CGCGCGC"))

    def test_active_learning_calibration(self):
        """Test BALD acquisition and calibration adapter ensemble updates."""
        pool_feats = torch.rand(30, 4)
        selected_bald = self.calibrator.select_active_assays(pool_feats, budget=5, strategy="bald")
        selected_var = self.calibrator.select_active_assays(pool_feats, budget=5, strategy="variance")
        self.assertEqual(len(selected_bald), 5)
        self.assertEqual(len(selected_var), 5)

        # Test update
        measured = torch.rand(5)
        loss = self.calibrator.update_calibration(pool_feats[selected_bald], measured, epochs=2)
        self.assertIsInstance(loss, float)

    def test_dual_chemistry_cascade_retriever(self):
        """Test end-to-end cascade retrieval vs single chemistries."""
        retriever = DualChemistryCascadeRetriever(
            encoder=self.encoder,
            cas9_predictor=self.cas9,
            thermo_model=self.thermo,
            abstention_gate=self.abstention_gate,
            active_calibrator=self.calibrator,
            device=self.device,
        )
        query = torch.randn(1, 64)
        lib = torch.zeros(20, 100, 4)
        lib[:, :, 0] = 1.0  # library of 20 targets

        # Cascade mode
        res_cascade = retriever.retrieve(query, lib, top_k_coarse=10, top_k_final=5, mode="cascade")
        self.assertIn("top_indices", res_cascade)
        self.assertIn("pool_reduction_factor", res_cascade)

        # Cas9-only mode
        res_cas9 = retriever.retrieve(query, lib, top_k_final=5, mode="cas9_only")
        self.assertIn("top_indices", res_cas9)

        # Hybridization-only mode
        res_hyb = retriever.retrieve(query, lib, top_k_final=5, mode="hyb_only")
        self.assertIn("top_indices", res_hyb)

    def test_scientific_metrics(self):
        """Test Recall@k, NDCG@k, ECE, and sequence viability metrics."""
        retrieved = [1, 2, 3, 4, 5]
        gt = [2, 5, 8]
        r3 = recall_at_k(retrieved, gt, k=3)
        self.assertAlmostEqual(r3, 1.0 / 3.0)
        r5 = recall_at_k(retrieved, gt, k=5)
        self.assertAlmostEqual(r5, 2.0 / 3.0)

        # NDCG@k
        rel_dict = {1: 1.0, 2: 3.0, 3: 0.0, 4: 2.0, 5: 1.0}
        ndcg = ndcg_at_k(retrieved, rel_dict, k=5)
        self.assertTrue(0.0 <= ndcg <= 1.0)

        # Expected Calibration Error (ECE)
        preds = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        ece = expected_calibration_error(preds, labels, n_bins=5)
        self.assertTrue(0.0 <= ece <= 1.0)

        # Sequence viability
        seqs = ["ACGTACGTACGTACGT", "AAAAAAAAAAAA", "CGCGCGCGCGCGCGCG"]
        viab = evaluate_sequence_viability(seqs)
        self.assertIn("gc_valid_pct", viab)
        self.assertIn("homopolymer_free_pct", viab)

        # Reliability diagram data
        rel_diag = compute_reliability_diagram_data(preds, labels, n_bins=5)
        self.assertEqual(len(rel_diag["bin_centers"]), 5)
        self.assertEqual(len(rel_diag["bin_accuracies"]), 5)
        self.assertEqual(len(rel_diag["bin_confidences"]), 5)

        # Risk-coverage curve
        conf = np.array([0.95, 0.85, 0.70, 0.40, 0.10])
        err = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        rc = compute_risk_coverage_curve(conf, err, num_points=5)
        self.assertEqual(len(rc["coverages"]), 5)
        self.assertEqual(len(rc["risks"]), 5)
        self.assertLessEqual(rc["risks"][0], rc["risks"][-1])  # Lowest risk at highest confidence

    def test_sliding_hairpin_mfe(self):
        """Test sliding-window hairpin MFE detection."""
        # Hairpin sequence: 5-bp stem 'GCGCG', 4-nt loop 'TTTT', 5-bp complement 'CGCGC'
        hp_seq = "AAAA" + "GCGCG" + "TTTT" + "CGCGC" + "AAAA"
        hp_oh = dna_to_onehot(hp_seq).unsqueeze(0)
        dG_hp = self.thermo.estimate_hairpin_stability(hp_oh).item()

        # Poly-A sequence should have zero or negligible secondary structure
        linear_seq = "A" * len(hp_seq)
        linear_oh = dna_to_onehot(linear_seq).unsqueeze(0)
        dG_linear = self.thermo.estimate_hairpin_stability(linear_oh).item()

        self.assertLess(dG_hp, 0.0, "Hairpin with complementary stem must have negative folding free energy DeltaG")
        self.assertEqual(dG_linear, 0.0, "Poly-A should have 0.0 secondary structure DeltaG")


    def test_cas9_mismatch_and_pam(self):
        """Test Cas9 mismatch matrix and PAM compatibility."""
        guide = dna_to_onehot("A" * 20).unsqueeze(0)
        target_exact = dna_to_onehot("A" * 20).unsqueeze(0)
        pam_ngg = dna_to_onehot("AGG").unsqueeze(0)
        pam_nha = dna_to_onehot("AGA").unsqueeze(0)

        p_ngg = self.cas9(guide, target_exact, target_pam=pam_ngg).item()
        p_nha = self.cas9(guide, target_exact, target_pam=pam_nha).item()

        self.assertGreater(p_ngg, p_nha, "NGG PAM should cleave more efficiently than non-canonical PAM")


    def test_retrieval_monotonicity(self):
        """Test cascade retriever monotonicity across k values."""
        retriever = DualChemistryCascadeRetriever(
            encoder=self.encoder,
            cas9_predictor=self.cas9,
            thermo_model=self.thermo,
            abstention_gate=self.abstention_gate,
            active_calibrator=self.calibrator,
            device=self.device,
        )
        query = torch.randn(1, 64)
        lib = torch.zeros(30, 100, 4)
        lib[:, :, 0] = 1.0

        res = retriever.retrieve(
            query, lib,
            top_k_coarse=20,
            top_k_final=20,
            enforce_abstention=False,
            precomputed_confidence=1.0
        )
        ranks = res["top_indices"]
        self.assertEqual(len(ranks), 20)

        # Evaluating monotonic Recall@k on this ranking
        gt = [ranks[2], ranks[8], ranks[15]]
        recalls = [recall_at_k(ranks, gt, k) for k in [1, 5, 10, 20]]
        for i in range(len(recalls) - 1):
            self.assertLessEqual(recalls[i], recalls[i + 1], "Recall@k must be mathematically monotonic")


if __name__ == "__main__":
    unittest.main()

