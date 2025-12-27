"""Models module for molecular solubility prediction.

This module provides various model architectures:
- ECFPRandomForest: Random Forest baseline with ECFP fingerprints
- DMPNN: Directed Message Passing Neural Network
- MoleculeTransformer: Transformer trained from scratch
- ChemBERTaForSolubility: Fine-tuned ChemBERTa model
"""

from Models.layers import MCDropout, set_mc_dropout
from Models.random_forest import ECFPRandomForest
from Models.dmpnn import DMPNN, DMPNNEncoder
from Models.transformer import MoleculeTransformer, SMILESTokenizer
from Models.chemberta import ChemBERTaForSolubility

__all__ = [
    "MCDropout",
    "set_mc_dropout",
    "ECFPRandomForest",
    "DMPNN",
    "DMPNNEncoder",
    "MoleculeTransformer",
    "SMILESTokenizer",
    "ChemBERTaForSolubility",
]
