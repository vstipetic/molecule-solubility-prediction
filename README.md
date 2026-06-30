# Molecule Solubility Prediction

Comparative study of four model architectures for predicting aqueous molecular solubility (log mol/L) from SMILES strings, with calibrated uncertainty estimation on each.

## Overview

Given a molecule as a SMILES string, predict its aqueous solubility and a confidence estimate. Four architectures are compared, ranging from a classical ML baseline to a 77 M-parameter pretrained language model:

| Model | Input representation | Uncertainty |
|---|---|---|
| Random Forest | ECFP4 fingerprints (2048-bit Morgan) | Tree variance across the forest |
| D-MPNN (GNN) | Molecular graph (atom + bond features) | MC Dropout |
| Transformer (scratch) | SMILES tokens; pretrained on ZINC-1M MLM | MC Dropout |
| ChemBERTa | HuggingFace `seyonec/ChemBERTa-zinc-base-v1` fine-tuned | MC Dropout in regression head |

Uncertainty is reported as the **standard deviation** of predictions in the same units as the solubility (log mol/L). For the three deep-learning models, this is computed via Monte Carlo Dropout: the model is kept in `eval()` mode so that BatchNorm/LayerNorm behave deterministically, while only the custom `MCDropout` layers remain stochastic across N forward passes.

## Project Structure

```
molecule-solubility-prediction/
├── Models/
│   ├── layers.py                    # MCDropout, MCDropoutMixin, set_mc_dropout
│   ├── random_forest.py             # ECFPRandomForest wrapper
│   ├── dmpnn.py                     # D-MPNN encoder + FFN head
│   ├── transformer.py               # SMILESTokenizer + MoleculeTransformer
│   └── chemberta.py                 # ChemBERTaForSolubility wrapper
├── DataUtils/
│   ├── utils.py                     # ECFP, atom/bond features, scaffold_split, load_data
│   ├── datasets.py                  # AqSolDBDataset, ESOLDataset, ZINCDataset, etc.
│   ├── graph_preprocessing.py       # SMILES → PyG Data objects
│   ├── metrics.py                   # compute_metrics, compute_calibration_metrics
│   ├── collate.py                   # DataLoader collate functions
│   └── prepare_data.py              # Script to write train/val/test CSV splits
├── Train/
│   ├── pretrain_transformer.py      # MLM pretraining on ZINC-1M
│   ├── finetune_transformer.py      # Fine-tune scratch transformer on AqSolDB
│   ├── finetune_chemberta.py        # Fine-tune ChemBERTa on AqSolDB
│   ├── train_gnn.py                 # Train D-MPNN
│   ├── train_random_forest.py       # Train Random Forest
│   ├── validate_transformer.py      # Validation loop for transformers
│   ├── validate_gnn.py              # Validation loop for D-MPNN
│   ├── validate_random_forest.py    # Validation loop for Random Forest
│   ├── calibrate_conformal.py       # Fit conformal prediction intervals
│   └── wandb_utils.py               # Shared wandb init/finish helpers
├── UQ/
│   ├── conformal.py                 # Split conformal regressor
│   └── evaluate.py                  # Conformal interval metric helpers
├── runpod_runner/
│   ├── pod_manager.py               # RunPod pod lifecycle (runpod SDK)
│   ├── train_on_runpod.py           # Orchestrator: train one model on a pod
│   └── bootstrap.sh                 # In-pod setup + training script
├── Inference/
│   ├── inference_random_forest.py
│   ├── inference_gnn.py
│   ├── inference_transformer.py
│   └── inference_chemberta.py
└── pyproject.toml
```

## Setup

Requires Python >= 3.10. Uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

Key dependencies: `torch`, `torch-geometric`, `rdkit`, `transformers`, `scikit-learn`, `wandb`.

## Datasets

| Dataset | Role | Target column |
|---|---|---|
| AqSolDB | Training + internal val/test | `Solubility` (log mol/L) |
| ESOL (Delaney) | Held-out distribution shift test | `measured log solubility in mols per litre` |
| ZINC-1M | MLM pretraining for the scratch transformer | SMILES only |

### Data splitting

All splits use **Murcko scaffold splitting** to prevent data leakage. Molecules sharing the same scaffold are kept in the same split, so the model is evaluated on genuinely novel chemical scaffolds.

```bash
python -m DataUtils.prepare_data --data-path aqsoldb.csv --output-dir data/splits/
```

## Training Pipeline

### 1. Random Forest

```bash
python -m Train.train_random_forest \
    --train-path data/splits/train.csv \
    --val-path data/splits/val.csv \
    --output-path models/rf.pkl
```

### 2. D-MPNN (GNN)

```bash
python -m Train.train_gnn \
    --train-path data/splits/train.csv \
    --val-path data/splits/val.csv \
    --output-path models/dmpnn.pt
```

### 3. Transformer from scratch

First pretrain on ZINC, then fine-tune:

```bash
python -m Train.pretrain_transformer \
    --zinc-path data/zinc_1m.csv \
    --output-path models/transformer_pretrained.pt

python -m Train.finetune_transformer \
    --pretrained-path models/transformer_pretrained.pt \
    --train-path data/splits/train.csv \
    --val-path data/splits/val.csv \
    --output-path models/transformer.pt
```

### 4. ChemBERTa

```bash
python -m Train.finetune_chemberta \
    --train-path data/splits/train.csv \
    --val-path data/splits/val.csv \
    --output-path models/chemberta.pt
```

All training scripts log to [Weights & Biases](https://wandb.ai): loss curves, RMSE, MAE, and uncertainty calibration metrics.

### Logging

All training entry points initialize wandb through a shared helper (`Train/wandb_utils.py`), which is environment-aware and safe for headless/RunPod runs:

- `WANDB_API_KEY` — when unset and no mode is forced, the run starts in **offline** mode so training never blocks on an interactive login prompt. Set `WANDB_MODE=online` (or provide the key) to stream remotely.
- `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_DIR`, `WANDB_MODE` — optional env overrides; per-run `--wandb-entity`, `--wandb-group`, `--wandb-tags`, `--wandb-mode` CLI flags are available on every training script.

```bash
export WANDB_API_KEY=...        # remote logging
export WANDB_ENTITY=my-team      # optional
uv run python -m Train.train_gnn --data-path data/aqsoldb.csv --wandb-group runpod
```

**`.env` file (gitignored).** Put your keys in `.env` at the repo root (see `.env.example` for the template). Both the local training scripts and the RunPod orchestrator auto-load it — real shell exports still take precedence. `.env` is in `.gitignore` so secrets are never committed.

```dotenv
RUNPOD_API_KEY=...
WANDB_API_KEY=...
WANDB_ENTITY=...        # optional
WANDB_PROJECT=mol-solubility
```

## Training on RunPod

`runpod_runner/train_on_runpod.py` deploys a single transient GPU pod, uploads your data (+ optional extra artifacts), runs the selected training module over SSH, streams the log to your console, then SCPs the checkpoint and `train.log` back and terminates the pod. Teardown is wrapped in `try/finally` so the pod is always released. Random Forest is excluded (it runs locally on CPU per the project plan).

Prerequisites (one-time):

1. Install the RunPod extra: `uv sync --extra runpod`.
2. Set `RUNPOD_API_KEY` (RunPod → Settings → API Keys).
3. Set `WANDB_API_KEY` (and optionally `WANDB_ENTITY`/`WANDB_PROJECT`) — these are injected into the pod so it logs remotely.
4. Add your SSH **public** key under RunPod → Settings → SSH Keys (RunPod injects it into the pod). The matching private key is auto-detected (`~/.ssh/id_ed25519` / `id_rsa` / `id_ecdsa`) or passed via `--ssh-private-key`.

```bash
# Train D-MPNN on an A6000, streaming metrics to wandb
uv run --extra runpod python -m runpod_runner.train_on_runpod \
    --model dmpnn \
    --data-path data/aqsoldb.csv \
    --training-args "--n-epochs 50 --batch-size 64"

# Fine-tune the scratch transformer (upload its pretrained checkpoint too)
uv run --extra runpod python -m runpod_runner.train_on_runpod \
    --model transformer \
    --data-path data/aqsoldb.csv \
    --upload checkpoints/pretrained_model.pt:/workspace/uploads/pretrained.pt \
    --training-args "--pretrained-path /workspace/uploads/pretrained.pt"
```

The checkpoint and `train.log` are downloaded to `./runpod_downloads/<pod_id>/`. Use `--keep-pod` to leave the pod running for debugging, and `--gpu-type` / `--image` / `--volume` to customize the pod. The pod clones the repo from `--repo-url` (default: this project's GitHub origin) on a persistent `/workspace` volume.

## Inference

All inference scripts accept a CSV with a SMILES column and write predictions in a standard format.

```bash
# Random Forest
python -m Inference.inference_random_forest \
    --model-path models/rf.pkl \
    --input-path molecules.csv \
    --output-path predictions.csv

# D-MPNN
python -m Inference.inference_gnn \
    --model-path models/dmpnn.pt \
    --input-path molecules.csv \
    --output-path predictions.csv

# Transformer (scratch)
python -m Inference.inference_transformer \
    --model-path models/transformer.pt \
    --input-path molecules.csv \
    --output-path predictions.csv

# ChemBERTa
python -m Inference.inference_chemberta \
    --model-path models/chemberta.pt \
    --input-path molecules.csv \
    --output-path predictions.csv
```

**Common options:**

| Flag | Default | Description |
|---|---|---|
| `--no-uncertainty` | off | Skip uncertainty estimation (faster) |
| `--n-mc-samples N` | 100 | MC Dropout forward passes |
| `--batch-size N` | 32 | Inference batch size |
| `--device cuda/cpu` | auto | Compute device |

**Output format:**

| Column | Description |
|---|---|
| `SMILES` | Input molecular structure |
| `Solubility` | Predicted solubility (log mol/L) |
| `Uncertainty` | Std of predictions (same units) — omitted with `--no-uncertainty` |

## Key Design Decisions

**MCDropout** — A custom `nn.Module` with an independent `mc_active` flag. This decouples dropout from the model's `training` mode, so `model.eval()` still disables BatchNorm/LayerNorm but dropout can be re-enabled on demand via `model.enable_mc_dropout()` / `model.disable_mc_dropout()`.

**Scaffold split** — Murcko scaffold grouping sorts scaffolds by cluster size (largest first) and greedily fills train, then val, then test. This is the same strategy used in Chemprop and is considered the toughest standard split for molecular ML.

**SMILES tokenization** — The scratch transformer uses an atom-aware regex tokenizer that splits SMILES into chemically meaningful tokens (`Br`, `Cl`, bracketed atoms like `[C@H]`, bonds, ring closures). ChemBERTa uses the RoBERTa BPE tokenizer it was pretrained with.

**Uncertainty calibration** — Beyond RMSE/MAE, all models report coverage at 50%, 90%, and 95% confidence intervals and Gaussian NLL. For well-calibrated models, coverage should match the nominal confidence level.
