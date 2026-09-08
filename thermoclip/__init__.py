"""
ThermoCLIP-DNA: Active, Uncertainty-Aware, Dual-Chemistry Molecular Semantic Retrieval
"""

__version__ = "1.0.0"
__author__ = "Labony Sur"

from thermoclip.models.encoder import DualChannelDNAEncoder
from thermoclip.biophysics.cas9 import Cas9CleavagePredictor
from thermoclip.biophysics.thermodynamics import SantaLuciaDuplexModel
from thermoclip.biophysics.assay_simulator import WetLabAssaySimulator
from thermoclip.models.abstention import MolecularAbstentionGate
from thermoclip.calibration.active_learner import ActiveLabCalibrator
from thermoclip.search.cascade import DualChemistryCascadeRetriever

__all__ = [
    "DualChannelDNAEncoder",
    "Cas9CleavagePredictor",
    "SantaLuciaDuplexModel",
    "WetLabAssaySimulator",
    "MolecularAbstentionGate",
    "ActiveLabCalibrator",
    "DualChemistryCascadeRetriever",
]
