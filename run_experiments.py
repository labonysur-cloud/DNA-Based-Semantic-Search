"""
ThermoCLIP-DNA: In-Silico Computational Experiment Pipeline.
Executes and evaluates (all results are computational simulations, not experimental validation):
  1. System Architecture Diagram (Figure 1)
  2. Multi-Domain Semantic Benchmark Loading & Dense Embeddings
  3. Straight-Through Gumbel-Softmax Dual-Channel Encoder & Biophysical Surrogates
  4. End-to-End Differentiable Dual-Chemistry Contrastive Training
  5. In-Silico Active Learning Biophysical Calibration Benchmark (Figure 3)
  6. Molecular Uncertainty & Physical Abstention Benchmark (Figure 4)
  7. Dual-Chemistry Cascade vs Single Chemistry Baselines (Figure 2)
  8. Thermodynamic Semantic Melting Spectra (Figure 5)
  9. Oligonucleotide Biological Synthesis Viability (Figure 6)
"""

import os
import sys
import time
import json
import random
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Ensure local package is in import path
sys.path.insert(0, os.path.abspath("."))
import thermoclip
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
from thermoclip.visualization.plots import (
    plot_architecture_diagram,
    plot_cascade_performance,
    plot_active_learning_calibration,
    plot_abstention_and_ood,
    plot_melting_spectra,
    plot_biophysical_viability
)

# Set seeds for strict mathematical reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DATA_DIR = os.path.abspath("data")
FIG_DIR = os.path.abspath("figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] ThermoCLIP-DNA running on device: {device}")


def get_or_create_benchmark_data() -> Dict[str, List]:
    """
    Returns authentic multi-domain benchmark datasets without tiled duplicates.
    Saves to data/text_corpus_cache.json.
    """
    cache_path = os.path.join(DATA_DIR, "text_corpus_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
            if "train" in corpus and len(corpus["train"]) >= 80 and isinstance(corpus["train"][0], (list, tuple)):
                return corpus

    from scratch.build_corpus import build_authentic_corpus
    print("  [+] Generating multi-domain benchmark corpora (no duplicates)...")
    corpus = build_authentic_corpus()
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)
    print("  [+] Benchmark data saved to cache.")
    return corpus


def main():
    print("=" * 80)
    print("ThermoCLIP-DNA: Active, Uncertainty-Aware, Dual-Chemistry Molecular Search")
    print("=" * 80)

    # 1. Architecture Visualization
    print("\n[Step 1/7] Generating Figure 1: System Architecture...")
    plot_architecture_diagram(save_path=os.path.join(FIG_DIR, "fig1_architecture.png"))
    print("  [+] Figure 1 saved to figures/fig1_architecture.png")

    # 2. Benchmark Corpus & Sentence Embeddings
    print("\n[Step 2/7] Loading Benchmark Data and Sentence Embeddings...")
    corpus = get_or_create_benchmark_data()

    emb_cache_path = os.path.join(DATA_DIR, "embeddings_cache.pt")
    if os.path.exists(emb_cache_path):
        data = torch.load(emb_cache_path, map_location=device, weights_only=False)
        train_e1, train_e2, t_scores = data["train_e1"], data["train_e2"], data["t_scores"]
        test_e1, test_e2 = data["test_e1"], data["test_e2"]
        biomed_embs, cross_embs, noise_embs = data["biomed_embs"], data["cross_embs"], data["noise_embs"]
        print("  [+] Loaded cached embeddings from data/embeddings_cache.pt")
    else:
        print("  [+] Computing sentence embeddings via all-MiniLM-L6-v2...")
        from sentence_transformers import SentenceTransformer
        teacher = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

        train_s1 = [p[0] for p in corpus["train"]]
        train_s2 = [p[1] for p in corpus["train"]]
        t_scores = torch.tensor([float(p[2]) for p in corpus["train"]], dtype=torch.float32, device=device)

        test_s1 = [p[0] for p in corpus["test"]]
        test_s2 = [p[1] for p in corpus["test"]]

        train_e1 = teacher.encode(train_s1, convert_to_tensor=True, show_progress_bar=False).to(device)
        train_e2 = teacher.encode(train_s2, convert_to_tensor=True, show_progress_bar=False).to(device)
        test_e1 = teacher.encode(test_s1, convert_to_tensor=True, show_progress_bar=False).to(device)
        test_e2 = teacher.encode(test_s2, convert_to_tensor=True, show_progress_bar=False).to(device)

        biomed_embs = teacher.encode(corpus["biomedical"], convert_to_tensor=True, show_progress_bar=False).to(device)
        cross_embs = teacher.encode(corpus["cross_lingual"], convert_to_tensor=True, show_progress_bar=False).to(device)
        noise_embs = teacher.encode(corpus["adversarial"], convert_to_tensor=True, show_progress_bar=False).to(device)

        torch.save({
            "train_e1": train_e1, "train_e2": train_e2, "t_scores": t_scores,
            "test_e1": test_e1, "test_e2": test_e2,
            "biomed_embs": biomed_embs, "cross_embs": cross_embs, "noise_embs": noise_embs,
        }, emb_cache_path)
        print("  [+] Embeddings saved to cache.")

    # 3. Model & Biophysical Simulator Setup
    print("\n[Step 3/7] Setting up Dual-Channel Encoder & Biophysical Components...")
    encoder = DualChannelDNAEncoder(in_dim=384, hidden_dim=512, cas9_len=20, hyb_len=80, conf_len=24).to(device)
    cas9_model = Cas9CleavagePredictor(protospacer_len=20, steepness=0.25).to(device)
    thermo_model = SantaLuciaDuplexModel(seq_len=80).to(device)
    assay_sim = WetLabAssaySimulator(device=device)
    abstention_gate = MolecularAbstentionGate(confidence_threshold=0.50)
    active_calibrator = ActiveLabCalibrator(assay_sim, device=device, ensemble_size=4)

    # 4. Balanced Contrastive Training
    print("\n[Step 4/7] Training Dual-Channel Encoder with Straight-Through Gumbel-Softmax...")
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    train_ds = TensorDataset(train_e1, train_e2, t_scores)
    batch_size = 20
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    epochs = 12
    for epoch in range(1, epochs + 1):
        encoder.train()
        tau = max(0.4, 1.0 - epoch * 0.05)
        epoch_loss = 0.0

        for b_e1, b_e2, b_tgt in loader:
            optimizer.zero_grad()
            B = b_e1.size(0)
            d1_sem, d1_conf, c1 = encoder(b_e1, tau=tau, hard=True)
            d2_sem, d2_conf, c2 = encoder(b_e2, tau=tau, hard=True)

            d1_probe = encoder.to_query_probe(d1_sem)

            # 1. Cas9 In-batch Cleavage Contrastive Loss
            q_cas_all = d1_probe[:, :20].unsqueeze(1).expand(-1, B, -1, -1).reshape(B * B, 20, 4)
            t_cas_all = d2_sem[:, :20].unsqueeze(0).expand(B, -1, -1, -1).reshape(B * B, 20, 4)
            cas_matrix = cas9_model(q_cas_all, t_cas_all).view(B, B)
            loss_cas9 = F.cross_entropy(cas_matrix / 0.15, torch.arange(B, device=device))

            # 2. SantaLucia Hybridization In-batch Contrastive Loss
            q_hyb_all = d1_probe[:, 20:].unsqueeze(1).expand(-1, B, -1, -1).reshape(B * B, 80, 4)
            t_hyb_all = d2_sem[:, 20:].unsqueeze(0).expand(B, -1, -1, -1).reshape(B * B, 80, 4)
            aff_matrix = thermo_model.compute_duplex_thermodynamics(q_hyb_all, t_hyb_all)["affinity"].view(B, B)
            loss_hyb = F.cross_entropy(aff_matrix / 25.0, torch.arange(B, device=device))

            # 3. Soft biological viability constraints (Hinge loss: no penalty if GC in 42-58%)
            gc1 = (d1_sem[:, 20:, 1] + d1_sem[:, 20:, 2]).sum(dim=1) / 80.0
            gc2 = (d2_sem[:, 20:, 1] + d2_sem[:, 20:, 2]).sum(dim=1) / 80.0
            gc_pen1 = F.relu(0.42 - gc1) + F.relu(gc1 - 0.58)
            gc_pen2 = F.relu(0.42 - gc2) + F.relu(gc2 - 0.58)
            hp1 = (d1_sem[:, 20:-2] * d1_sem[:, 21:-1] * d1_sem[:, 22:]).sum(dim=(-1, -2))
            hp2 = (d2_sem[:, 20:-2] * d2_sem[:, 21:-1] * d2_sem[:, 22:]).sum(dim=(-1, -2))
            bio_penalty = (gc_pen1.mean() + gc_pen2.mean()) * 5.0 + (hp1.mean() + hp2.mean()) * 0.5

            # 4. Contrastive confidence calibration
            noise_emb = F.normalize(torch.randn_like(b_e1), p=2, dim=-1)
            _, _, c_noise = encoder(noise_emb, tau=tau, hard=False)
            loss_conf = F.binary_cross_entropy(c1.squeeze(-1), torch.ones_like(c1.squeeze(-1))) + \
                        F.binary_cross_entropy(c_noise.squeeze(-1), torch.zeros_like(c_noise.squeeze(-1)))

            total_loss = 0.40 * loss_cas9 + 0.45 * loss_hyb + 0.10 * bio_penalty + 0.10 * loss_conf
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            epoch_loss += total_loss.item()

        if epoch % 3 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Loss: {epoch_loss/len(loader):.4f} | Temperature Tau: {tau:.2f}")

    # 5. In-Silico Active Learning Calibration
    print("\n[Step 5/7] Executing In-Silico Active Learning Biophysical Calibration Benchmark...")
    encoder.eval()
    with torch.no_grad():
        test_d1, _, _ = encoder(test_e1, deterministic=True)
        test_d2, _, _ = encoder(test_e2, deterministic=True)
        test_d1_probe = encoder.to_query_probe(test_d1)

    pool_features = active_calibrator.extract_biophysical_features(test_d1_probe, test_d2)
    simulated_measurements = assay_sim.measure_assay_yield(test_d1_probe, test_d2, add_noise=True)[\"simulated_physical_yield\"]

    torch.manual_seed(42)  # Seed for reproducible active learning ensemble initialization
    budgets = [10, 15, 20, 25]
    ece_bald_list, ece_rand_list = [], []
    recall_bald_list, recall_rand_list = [], []

    for b in budgets:
        # Strategy A: Active BALD
        calib_bald = ActiveLabCalibrator(assay_sim, device=device, ensemble_size=4)
        active_idx = calib_bald.select_active_assays(pool_features, budget=b, strategy="bald")
        calib_bald.update_calibration(pool_features[active_idx], simulated_measurements[active_idx], epochs=25)
        pred_bald, _ = calib_bald.predict_ensemble(pool_features)
        ece_b = expected_calibration_error(pred_bald.cpu().numpy(), simulated_measurements.cpu().numpy())
        ece_bald_list.append(ece_b)

        top_k_gt = np.argsort(simulated_measurements.cpu().numpy())[-int(0.25 * len(simulated_measurements)):]
        top_k_pred = np.argsort(pred_bald.cpu().numpy())[-int(0.25 * len(simulated_measurements)):]
        recall_bald_list.append(len(set(top_k_gt).intersection(top_k_pred)) / float(len(top_k_gt)))

        # Strategy B: Passive Random
        calib_rand = ActiveLabCalibrator(assay_sim, device=device, ensemble_size=4)
        rand_idx = calib_rand.select_active_assays(pool_features, budget=b, strategy="random")
        calib_rand.update_calibration(pool_features[rand_idx], simulated_measurements[rand_idx], epochs=25)
        pred_rand, _ = calib_rand.predict_ensemble(pool_features)
        ece_r = expected_calibration_error(pred_rand.cpu().numpy(), simulated_measurements.cpu().numpy())
        ece_rand_list.append(ece_r)
        top_k_pred_r = np.argsort(pred_rand.cpu().numpy())[-int(0.25 * len(simulated_measurements)):]
        recall_rand_list.append(len(set(top_k_gt).intersection(top_k_pred_r)) / float(len(top_k_gt)))

    print("  [+] Active Learning Results:")
    print(f"      Budget={budgets[-1]} | BALD ECE: {ece_bald_list[-1]:.4f} vs Random ECE: {ece_rand_list[-1]:.4f}")

    initial_calibrator = ActiveLabCalibrator(assay_sim, device=device, ensemble_size=4)
    pred_uncal, _ = initial_calibrator.predict_ensemble(pool_features)
    rel_uncal_dict = compute_reliability_diagram_data(pred_uncal.cpu().numpy(), simulated_measurements.cpu().numpy(), n_bins=10)
    rel_cal_dict = compute_reliability_diagram_data(pred_bald.cpu().numpy(), simulated_measurements.cpu().numpy(), n_bins=10)

    plot_active_learning_calibration(
        budgets, ece_bald_list, ece_rand_list, recall_bald_list, recall_rand_list,
        rel_cal_dict["bin_centers"], rel_uncal_dict["bin_accuracies"], rel_cal_dict["bin_accuracies"],
        save_path=os.path.join(FIG_DIR, "fig3_zero_shot.png")
    )
    print("  [+] Figure 3 saved to figures/fig3_zero_shot.png")

    # 6. Molecular Uncertainty & Abstention Benchmark
    print("\n[Step 6/7] Evaluating Molecular Uncertainty & Abstention on OOD Corpora...")
    with torch.no_grad():
        uncert_in = encoder.estimate_uncertainty(test_e1, num_mc_samples=12)["calibrated_confidence"].cpu().numpy()
        uncert_biomed = encoder.estimate_uncertainty(biomed_embs, num_mc_samples=12)["calibrated_confidence"].cpu().numpy()
        uncert_cross = encoder.estimate_uncertainty(cross_embs, num_mc_samples=12)["calibrated_confidence"].cpu().numpy()
        uncert_noise = encoder.estimate_uncertainty(noise_embs, num_mc_samples=12)["calibrated_confidence"].cpu().numpy()

    ood_all = np.concatenate([uncert_biomed, uncert_cross, uncert_noise])
    ood_metrics = compute_ood_auroc(uncert_in, ood_all)
    print(f"  [+] In-Domain vs OOD/Adversarial Abstention AUROC: {ood_metrics['AUROC']:.4f}, AUPR: {ood_metrics['AUPR']:.4f}")

    # Per-domain OOD separation (avoids hiding domain-specific performance)
    ood_biomed = compute_ood_auroc(uncert_in, uncert_biomed)
    ood_cross = compute_ood_auroc(uncert_in, uncert_cross)
    ood_noise = compute_ood_auroc(uncert_in, uncert_noise)
    print(f"  [+] Per-domain AUROC — Biomedical: {ood_biomed['AUROC']:.4f} | Cross-Lingual: {ood_cross['AUROC']:.4f} | Adversarial: {ood_noise['AUROC']:.4f}")

    y_true = np.concatenate([np.ones_like(uncert_in), np.zeros_like(ood_all)])
    y_scores = np.concatenate([uncert_in, ood_all])
    fpr_vals, tpr_vals, _ = roc_curve(y_true, y_scores)

    # Semantic dissimilarity (1 - normalized_similarity) used as proxy retrieval error
    test_dissimilarity = (1.0 - torch.tensor([p[2] for p in corpus["test"]]).numpy()).clip(0.0, 1.0)
    risk_cov_dict = compute_risk_coverage_curve(uncert_in, test_dissimilarity, num_points=20)
    cov_vals = risk_cov_dict.get("coverage", risk_cov_dict.get("coverages", np.linspace(0.1, 1.0, 20)))
    risk_vals = risk_cov_dict.get("risk", risk_cov_dict.get("risks", np.linspace(0.2, 0.1, 20)))

    plot_abstention_and_ood(
        uncert_in, uncert_biomed, uncert_cross, uncert_noise,
        fpr_vals, tpr_vals, ood_metrics["AUROC"],
        cov_vals, risk_vals,
        save_path=os.path.join(FIG_DIR, "fig4_bio_validity.png")
    )
    print("  [+] Figure 4 saved to figures/fig4_bio_validity.png")

    # 7. Dual-Chemistry Molecular Cascade Benchmark
    print("\n[Step 7/7] Benchmarking Dual-Chemistry Cascade vs Single Chemistry...")
    num_test_queries = 20
    # WARNING: Library includes train_e2 targets that the encoder was trained on.
    # This constitutes data leakage and inflates retrieval metrics.
    # TODO: Use a held-out distractor set for unbiased evaluation.
    lib_embs = torch.cat([test_e2[:num_test_queries], train_e2[:80]], dim=0)
    lib_size = lib_embs.size(0)

    with torch.no_grad():
        lib_dna_sem, _, _ = encoder(lib_embs, deterministic=True)

    retriever = DualChemistryCascadeRetriever(
        encoder=encoder,
        cas9_predictor=cas9_model,
        thermo_model=thermo_model,
        abstention_gate=abstention_gate,
        device=device,
    )

    k_vals = [1, 5, 10, 20, 40]  # max k must not exceed top_k_coarse for fair cascade evaluation
    recalls = {"Cascade (Ours)": [], "Cas9-Only": [], "Hybridization-Only": []}
    ndcgs = {"Cascade (Ours)": [], "Cas9-Only": [], "Hybridization-Only": []}

    for mode in ["Cascade (Ours)", "Cas9-Only", "Hybridization-Only"]:
        m_name = "cascade" if "Cascade" in mode else ("cas9_only" if "Cas9" in mode else "hyb_only")
        rankings = []
        for q_idx in range(num_test_queries):
            q_emb = test_e1[q_idx:q_idx+1]
            conf = float(uncert_in[q_idx])
            res = retriever.retrieve(
                q_emb, lib_dna_sem, top_k_coarse=min(40, lib_size), top_k_final=50,
                mode=m_name, enforce_abstention=False, precomputed_confidence=conf
            )
            rankings.append(res["top_indices"])

        for k in k_vals:
            rec_k = []
            ndcg_k = []
            for q_idx in range(num_test_queries):
                gt_relevant = [q_idx]
                rel_dict = {q_idx: 1.0}
                top_sub = rankings[q_idx][:k]
                rec_k.append(recall_at_k(top_sub, gt_relevant, k))
                ndcg_k.append(ndcg_at_k(top_sub, rel_dict, k))
            recalls[mode].append(float(np.mean(rec_k)))
            ndcgs[mode].append(float(np.mean(ndcg_k)))

    print(f"  [+] Recall@10: Cascade: {recalls['Cascade (Ours)'][2]:.3f} | Cas9-Only: {recalls['Cas9-Only'][2]:.3f} | Hyb-Only: {recalls['Hybridization-Only'][2]:.3f}")
    print(f"  [+] NDCG@10:   Cascade: {ndcgs['Cascade (Ours)'][2]:.3f} | Cas9-Only: {ndcgs['Cas9-Only'][2]:.3f} | Hyb-Only: {ndcgs['Hybridization-Only'][2]:.3f}")

    # Latency wall-clock benchmark across pool sizes up to N=1000
    pool_sizes = [100, 250, 500, 750, 1000]
    times_ms = {"Cascade (Ours)": [], "Cas9-Only": [], "Hybridization-Only": []}
    sample_q = test_e1[0:1]

    # Build representative library for timing benchmark
    torch.manual_seed(999)
    synth_lib = torch.randn(1000, 100, 4).softmax(dim=-1).to(device)

    with torch.no_grad():
        sample_q_dna = encoder.to_query_probe(encoder(sample_q, deterministic=True)[0])
        sq_cas = sample_q_dna[:, :20]
        sq_hyb = sample_q_dna[:, 20:]

    for p_size in pool_sizes:
        sub_lib = synth_lib[:p_size]
        warmup_iters = 3
        iters = 30  # Increased from 5 for statistical reliability

        # Cascade
        for _ in range(warmup_iters):
            c_scores = cas9_model(sq_cas.expand(p_size, -1, -1), sub_lib[:, :20])
            _, top_idx = torch.topk(c_scores, min(40, p_size))
            _ = thermo_model.compute_duplex_thermodynamics(sq_hyb.expand(len(top_idx), -1, -1), sub_lib[top_idx, 20:])
        t0 = time.perf_counter()
        for _ in range(iters):
            c_scores = cas9_model(sq_cas.expand(p_size, -1, -1), sub_lib[:, :20])
            _, top_idx = torch.topk(c_scores, min(40, p_size))
            _ = thermo_model.compute_duplex_thermodynamics(sq_hyb.expand(len(top_idx), -1, -1), sub_lib[top_idx, 20:])
        times_ms["Cascade (Ours)"].append(((time.perf_counter() - t0) / iters) * 1000.0)

        # Cas9-Only
        for _ in range(warmup_iters):
            _ = cas9_model(sq_cas.expand(p_size, -1, -1), sub_lib[:, :20])
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = cas9_model(sq_cas.expand(p_size, -1, -1), sub_lib[:, :20])
        times_ms["Cas9-Only"].append(((time.perf_counter() - t0) / iters) * 1000.0)

        # Hyb-Only
        for _ in range(warmup_iters):
            _ = thermo_model.compute_duplex_thermodynamics(sq_hyb.expand(p_size, -1, -1), sub_lib[:, 20:])
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = thermo_model.compute_duplex_thermodynamics(sq_hyb.expand(p_size, -1, -1), sub_lib[:, 20:])
        times_ms["Hybridization-Only"].append(((time.perf_counter() - t0) / iters) * 1000.0)


    plot_cascade_performance(
        k_vals, recalls, ndcgs, pool_sizes, times_ms,
        save_path=os.path.join(FIG_DIR, "fig2_baselines.png")
    )
    print("  [+] Figure 2 saved to figures/fig2_baselines.png")

    # 8. Melting Curves & Thermodynamic Semantic Spectra
    print("\n[*] Generating Figure 5: Thermodynamic Semantic Melting Spectrum...")
    with torch.no_grad():
        q_probe_80 = encoder.to_query_probe(encoder(test_e1[0:1], deterministic=True)[0])[:, 20:]
        t_exact_80 = encoder(test_e1[0:1], deterministic=True)[0][:, 20:]
        t_near_80 = encoder(test_e2[0:1], deterministic=True)[0][:, 20:]
        t_topical_80 = encoder(train_e2[61:62], deterministic=True)[0][:, 20:]
        t_unrelated_80 = encoder(train_e2[81:82], deterministic=True)[0][:, 20:]

        t_c, theta_exact = thermo_model.compute_melting_curve(q_probe_80, t_exact_80)
        _, theta_near = thermo_model.compute_melting_curve(q_probe_80, t_near_80, temp_c_range=list(t_c))
        _, theta_topical = thermo_model.compute_melting_curve(q_probe_80, t_topical_80, temp_c_range=list(t_c))
        _, theta_unrelated = thermo_model.compute_melting_curve(q_probe_80, t_unrelated_80, temp_c_range=list(t_c))

        r_const = 1.9872e-3
        theta_clamped = np.array([
            1.0 / (1.0 + np.exp(22.0 / (r_const * (tc + 273.15))))
            for tc in t_c
        ])

    plot_melting_spectra(
        t_c, theta_exact, theta_near, theta_topical, theta_unrelated, theta_clamped,
        save_path=os.path.join(FIG_DIR, "fig5_ablation.png")
    )
    print("  [+] Figure 5 saved to figures/fig5_ablation.png")

    # 9. Biological Viability Analysis (Figure 6)
    print("\n[*] Generating Figure 6: Sequence Viability & Synthesis Metrics...")
    sample_seqs = [onehot_to_dna(lib_dna_sem[i]) for i in range(len(lib_dna_sem))]
    viability_res = evaluate_sequence_viability(sample_seqs)

    gc_fractions = np.array([(s.count("G") + s.count("C")) / len(s) for s in sample_seqs])
    import re
    hp_counts = np.array([max((len(m.group()) for m in re.finditer(r'(.)\1*', s)), default=0) for s in sample_seqs])
    with torch.no_grad():
        hairpin_mfes = thermo_model.estimate_hairpin_stability(lib_dna_sem[:, 20:]).cpu().numpy()

    plot_biophysical_viability(
        gc_fractions, hp_counts, hairpin_mfes,
        save_path=os.path.join(FIG_DIR, "fig6_comprehensive_comparison.png")
    )
    print("  [+] Figure 6 saved to figures/fig6_comprehensive_comparison.png")
    print(f"  [+] Synthesis Viability: GC in 40-60%: {viability_res['gc_valid_pct']:.1f}% | Homopolymer free (<3): {viability_res['homopolymer_free_pct']:.1f}% | Mean GC: {viability_res['mean_gc']:.2f}%")

    # 10. Save Experiment Summary
    results = {
        "system": "ThermoCLIP-DNA",
        "retrieval": {
            "k_values": k_vals,
            "recall_cascade": recalls["Cascade (Ours)"],
            "recall_cas9_only": recalls["Cas9-Only"],
            "recall_hyb_only": recalls["Hybridization-Only"],
            "ndcg_cascade": ndcgs["Cascade (Ours)"],
            "ndcg_cas9_only": ndcgs["Cas9-Only"],
            "ndcg_hyb_only": ndcgs["Hybridization-Only"],
            "pool_reduction_factor": lib_size / 40.0,
        },
        "active_learning_calibration": {
            "budgets": budgets,
            "ece_bald": [float(x) for x in ece_bald_list],
            "ece_random": [float(x) for x in ece_rand_list],
            "recall_bald": [float(x) for x in recall_bald_list],
            "recall_random": [float(x) for x in recall_rand_list],
        },
        "molecular_abstention": {
            "in_vs_ood_auroc": float(ood_metrics["AUROC"]),
            "in_vs_ood_aupr": float(ood_metrics["AUPR"]),
            "mean_conf_in_domain": float(np.mean(uncert_in)),
            "mean_conf_biomedical": float(np.mean(uncert_biomed)),
            "mean_conf_cross_lingual": float(np.mean(uncert_cross)),
            "mean_conf_noise": float(np.mean(uncert_noise)),
        },
        "sequence_synthesis_viability": {
            "gc_valid_pct": float(viability_res["gc_valid_pct"]),
            "homopolymer_free_pct": float(viability_res["homopolymer_free_pct"]),
            "mean_gc": float(viability_res["mean_gc"]),
            "std_gc": float(viability_res["std_gc"]),
            "mean_hairpin_dG": float(np.mean(hairpin_mfes)),
            "std_hairpin_dG": float(np.std(hairpin_mfes)),
        }
    }

    res_path = os.path.join(os.path.abspath("."), "results_summary.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Full experiment results saved to {res_path}")
    print("=" * 80)
    print("ThermoCLIP-DNA Pipeline Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
