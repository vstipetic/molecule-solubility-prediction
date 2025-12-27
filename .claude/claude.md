# Molecule Solubility Prediction Project

## Project Overview
This project implements molecular solubility prediction using multiple model architectures with uncertainty quantification.

## Project Structure
```
molecule-solubility-prediction/
├── .claude/
│   └── claude.md                    # This file
├── Models/
│   ├── layers.py                    # Shared layers (MCDropout)
│   ├── random_forest.py             # Random Forest with ECFP fingerprints
│   ├── dmpnn.py                     # Chemprop-style D-MPNN
│   ├── transformer.py               # Transformer from scratch
│   └── chemberta.py                 # ChemBERTa-77M fine-tuning
├── DataUtils/
│   ├── utils.py                     # Shared utility functions (load_data, scaffold_split, etc.)
│   ├── datasets.py                  # Dataset classes (AqSolDB, ESOL, ZINCDataset, etc.)
│   ├── graph_preprocessing.py       # SMILES to graph conversion
│   ├── metrics.py                   # Shared metrics (compute_metrics, compute_calibration_metrics)
│   ├── collate.py                   # Collate functions for DataLoaders
│   └── prepare_data.py              # Script to prepare train/val/test CSV splits
├── Train/
│   ├── pretrain_transformer.py      # Pretrain on ZINC-1M
│   ├── finetune_transformer.py      # Fine-tune scratch transformer
│   ├── finetune_chemberta.py        # Fine-tune ChemBERTa
│   ├── train_gnn.py                 # Train D-MPNN
│   ├── train_random_forest.py       # Train Random Forest
│   ├── validate_transformer.py      # Validation for transformers
│   ├── validate_gnn.py              # Validation for GNN
│   └── validate_random_forest.py    # Validation for Random Forest
└── pyproject.toml                   # Dependencies (uv)
```

## Style Guide

### Type Hints
All functions must be type-hinted:
```python
def compute_ecfp(mol: Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    ...
```

### Imports
- Standard library imports first
- Third-party imports second
- Local imports third
- Use absolute imports

### Naming Conventions
- Classes: PascalCase (e.g., `MoleculeTransformer`, `DMPNNEncoder`)
- Functions/methods: snake_case (e.g., `compute_ecfp`, `scaffold_split`)
- Constants: UPPER_SNAKE_CASE (e.g., `ATOM_FEATURES`, `DEFAULT_RADIUS`)
- Private methods/attributes: prefix with underscore (e.g., `_build_vocab`)

### Documentation
- Use docstrings for all public functions and classes
- Include parameter types and return types in docstrings
- Keep inline comments minimal and meaningful

## Key Design Decisions

### MCDropout Layer
Custom dropout layer with independent `mc_active` flag for MC dropout uncertainty estimation:
- Allows models to stay in `eval()` mode during inference
- Only MCDropout layers activate, keeping LayerNorm/BatchNorm in eval mode
- Use `model.enable_mc_dropout()` / `model.disable_mc_dropout()` to toggle

### Scaffold Split
Use Murcko scaffolds for train/val/test splitting to prevent data leakage.

### SMILES Tokenization
Atom-aware regex tokenization for transformers:
```python
SMILES_REGEX = r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
```

## Models

### 1. Random Forest (Baseline)
- ECFP fingerprints (Morgan fingerprints)
- Uncertainty via tree variance

### 2. D-MPNN (GNN)
- Chemprop-style directed message passing
- MCDropout for uncertainty
- Deep ensemble support

### 3. Transformer (from scratch)
- Small model pretrained on ZINC-1M (MLM task)
- Fine-tuned on AqSolDB
- MCDropout for uncertainty

### 4. ChemBERTa (pretrained)
- Fine-tune `seyonec/ChemBERTa-zinc-base-v1` (77M params)
- MCDropout in regression head

## Datasets

### AqSolDB
Main training dataset with aqueous solubility data.
- Columns: SMILES, Solubility (log mol/L)

### ESOL
Held-out distribution shift test set.
- Columns: SMILES, measured log solubility

## Logging
All training and validation scripts use wandb for logging:
- Loss curves
- RMSE/MAE metrics
- Uncertainty calibration
- Model checkpoints
