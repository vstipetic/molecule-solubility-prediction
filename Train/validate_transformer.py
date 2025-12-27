"""Validation utilities for Transformer models.

This module provides functions to evaluate both the scratch-trained
MoleculeTransformer and fine-tuned ChemBERTa models with MC dropout
uncertainty estimation.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from Models.transformer import MoleculeTransformer, SMILESTokenizer
from Models.chemberta import ChemBERTaForSolubility
from DataUtils.metrics import compute_metrics, compute_calibration_metrics
from DataUtils.datasets import TransformerSolubilityDataset
from DataUtils.collate import transformer_collate_fn


def validate_transformer(
    model: MoleculeTransformer,
    dataloader: DataLoader,
    device: torch.device,
    log_to_wandb: bool = True,
    prefix: str = "val",
) -> Dict[str, float]:
    """Validate MoleculeTransformer model.

    Args:
        model: Trained MoleculeTransformer model.
        dataloader: DataLoader with tokenized SMILES.
        device: Device to run inference on.
        log_to_wandb: Whether to log metrics to wandb.
        prefix: Prefix for metric names.

    Returns:
        Dictionary of metrics.
    """
    model.eval()
    model.to(device)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            predictions = outputs['logits'].squeeze(-1)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    metrics = compute_metrics(predictions, targets)
    metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    if log_to_wandb and wandb.run is not None:
        wandb.log(metrics)

    return metrics


def validate_transformer_with_uncertainty(
    model: MoleculeTransformer,
    dataloader: DataLoader,
    device: torch.device,
    n_samples: int = 100,
    log_to_wandb: bool = True,
    prefix: str = "val",
) -> Dict[str, float]:
    """Validate MoleculeTransformer with MC dropout uncertainty.

    Args:
        model: Trained MoleculeTransformer (with MCDropout layers).
        dataloader: DataLoader with tokenized SMILES.
        device: Device to run inference on.
        n_samples: Number of forward passes for MC dropout.
        log_to_wandb: Whether to log metrics to wandb.
        prefix: Prefix for metric names.

    Returns:
        Dictionary of metrics including calibration.
    """
    model.eval()
    model.to(device)

    # Enable MC dropout
    model.enable_mc_dropout()

    all_predictions = []
    all_stds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Multiple forward passes
            batch_preds = []
            for _ in range(n_samples):
                outputs = model(input_ids, attention_mask)
                predictions = outputs['logits'].squeeze(-1)
                batch_preds.append(predictions)

            batch_preds = torch.stack(batch_preds, dim=0)
            mean_pred = batch_preds.mean(dim=0)
            std_pred = batch_preds.std(dim=0)

            all_predictions.append(mean_pred.cpu().numpy())
            all_stds.append(std_pred.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Disable MC dropout
    model.disable_mc_dropout()

    predictions = np.concatenate(all_predictions, axis=0)
    uncertainties = np.concatenate(all_stds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    metrics = compute_metrics(predictions, targets)
    calibration = compute_calibration_metrics(predictions, uncertainties, targets)
    metrics.update(calibration)

    metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    if log_to_wandb and wandb.run is not None:
        wandb.log(metrics)

    return metrics


def validate_chemberta(
    model: ChemBERTaForSolubility,
    smiles_list: List[str],
    targets: np.ndarray,
    batch_size: int = 32,
    device: torch.device = None,
    log_to_wandb: bool = True,
    prefix: str = "val",
) -> Dict[str, float]:
    """Validate ChemBERTa model.

    Args:
        model: Trained ChemBERTaForSolubility model.
        smiles_list: List of SMILES strings.
        targets: Target values.
        batch_size: Batch size for inference.
        device: Device to run inference on.
        log_to_wandb: Whether to log metrics to wandb.
        prefix: Prefix for metric names.

    Returns:
        Dictionary of metrics.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)

    all_predictions = []

    # Process in batches
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i + batch_size]
        encoded = model.encode_smiles(batch_smiles)

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            predictions = outputs['logits'].squeeze(-1)

        all_predictions.append(predictions.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)

    metrics = compute_metrics(predictions, targets)
    metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    if log_to_wandb and wandb.run is not None:
        wandb.log(metrics)

    return metrics


def validate_chemberta_with_uncertainty(
    model: ChemBERTaForSolubility,
    smiles_list: List[str],
    targets: np.ndarray,
    n_samples: int = 100,
    batch_size: int = 32,
    device: torch.device = None,
    log_to_wandb: bool = True,
    prefix: str = "val",
) -> Dict[str, float]:
    """Validate ChemBERTa with MC dropout uncertainty.

    Args:
        model: Trained ChemBERTaForSolubility model.
        smiles_list: List of SMILES strings.
        targets: Target values.
        n_samples: Number of forward passes for MC dropout.
        batch_size: Batch size for inference.
        device: Device to run inference on.
        log_to_wandb: Whether to log metrics to wandb.
        prefix: Prefix for metric names.

    Returns:
        Dictionary of metrics including calibration.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)
    model.enable_mc_dropout()

    all_predictions = []
    all_stds = []

    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i + batch_size]
        encoded = model.encode_smiles(batch_smiles)

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        batch_preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = model(input_ids, attention_mask)
                predictions = outputs['logits'].squeeze(-1)
                batch_preds.append(predictions)

        batch_preds = torch.stack(batch_preds, dim=0)
        mean_pred = batch_preds.mean(dim=0)
        std_pred = batch_preds.std(dim=0)

        all_predictions.append(mean_pred.cpu().numpy())
        all_stds.append(std_pred.cpu().numpy())

    model.disable_mc_dropout()

    predictions = np.concatenate(all_predictions, axis=0)
    uncertainties = np.concatenate(all_stds, axis=0)

    metrics = compute_metrics(predictions, targets)
    calibration = compute_calibration_metrics(predictions, uncertainties, targets)
    metrics.update(calibration)

    metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    if log_to_wandb and wandb.run is not None:
        wandb.log(metrics)

    return metrics


def create_transformer_dataloader(
    smiles_list: List[str],
    targets: np.ndarray,
    tokenizer: SMILESTokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    shuffle: bool = False,
) -> DataLoader:
    """Create DataLoader for MoleculeTransformer.

    Args:
        smiles_list: List of SMILES strings.
        targets: Target values.
        tokenizer: SMILES tokenizer.
        batch_size: Batch size.
        max_length: Maximum sequence length.
        shuffle: Whether to shuffle.

    Returns:
        DataLoader for transformer model.
    """
    dataset = TransformerSolubilityDataset(
        smiles_list=smiles_list,
        targets=targets,
        tokenizer=tokenizer,
        max_length=max_length,
        is_huggingface=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=transformer_collate_fn,
    )

    return dataloader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate Transformer model")
    parser.add_argument("--model-type", type=str, choices=['scratch', 'chemberta'],
                        default='scratch')
    parser.add_argument("--model-path", type=str, help="Path to saved model")
    parser.add_argument("--data-path", type=str, help="Path to validation data")
    parser.add_argument("--wandb-project", type=str, default="mol-solubility")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    wandb.init(project=args.wandb_project, job_type="validation")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Validation complete. Check wandb for metrics.")
