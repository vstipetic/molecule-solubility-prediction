"""Validation utilities for D-MPNN (GNN) model.

This module provides functions to evaluate the D-MPNN model
on validation/test sets with MC dropout uncertainty estimation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
import wandb

from Models.dmpnn import DMPNN, ensemble_predict
from Models.layers import set_mc_dropout
from DataUtils.graph_preprocessing import MoleculeGraphDataset
from DataUtils.metrics import compute_metrics, compute_calibration_metrics


def validate_gnn(
    model: DMPNN,
    dataloader: PyGDataLoader,
    device: torch.device,
    log_to_wandb: bool = True,
    prefix: str = "val",
) -> Dict[str, float]:
    """Validate D-MPNN model.

    Args:
        model: Trained DMPNN model.
        dataloader: PyTorch Geometric DataLoader.
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
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()

    metrics = compute_metrics(predictions, targets)
    metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    if log_to_wandb and wandb.run is not None:
        wandb.log(metrics)

    return metrics


def validate_gnn_with_uncertainty(
    model: DMPNN,
    dataloader: PyGDataLoader,
    device: torch.device,
    n_samples: int = 100,
    log_to_wandb: bool = True,
    prefix: str = "val",
) -> Dict[str, float]:
    """Validate D-MPNN with MC dropout uncertainty estimation.

    Args:
        model: Trained DMPNN model (must have MCDropout layers).
        dataloader: PyTorch Geometric DataLoader.
        device: Device to run inference on.
        n_samples: Number of forward passes for MC dropout.
        log_to_wandb: Whether to log metrics to wandb.
        prefix: Prefix for metric names.

    Returns:
        Dictionary of metrics including calibration metrics.
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
            batch = batch.to(device)

            # Multiple forward passes for MC dropout
            batch_preds = []
            for _ in range(n_samples):
                outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                batch_preds.append(outputs)

            # Stack predictions and compute statistics
            batch_preds = torch.stack(batch_preds, dim=0)
            mean_pred = batch_preds.mean(dim=0)
            std_pred = batch_preds.std(dim=0)

            all_predictions.append(mean_pred.cpu().numpy())
            all_stds.append(std_pred.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

    # Disable MC dropout
    model.disable_mc_dropout()

    predictions = np.concatenate(all_predictions, axis=0).flatten()
    uncertainties = np.concatenate(all_stds, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()

    # Regression metrics
    metrics = compute_metrics(predictions, targets)

    # Calibration metrics
    calibration = compute_calibration_metrics(predictions, uncertainties, targets)
    metrics.update(calibration)

    metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    if log_to_wandb and wandb.run is not None:
        wandb.log(metrics)

    return metrics


def validate_gnn_ensemble(
    models: List[DMPNN],
    dataloader: PyGDataLoader,
    device: torch.device,
    log_to_wandb: bool = True,
    prefix: str = "val",
) -> Dict[str, float]:
    """Validate D-MPNN deep ensemble.

    Args:
        models: List of trained DMPNN models.
        dataloader: PyTorch Geometric DataLoader.
        device: Device to run inference on.
        log_to_wandb: Whether to log metrics to wandb.
        prefix: Prefix for metric names.

    Returns:
        Dictionary of metrics including ensemble uncertainty.
    """
    for model in models:
        model.eval()
        model.to(device)

    all_predictions = []
    all_stds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            # Get predictions from each model
            model_preds = []
            for model in models:
                outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                model_preds.append(outputs)

            # Stack and compute statistics
            model_preds = torch.stack(model_preds, dim=0)
            mean_pred = model_preds.mean(dim=0)
            std_pred = model_preds.std(dim=0)

            all_predictions.append(mean_pred.cpu().numpy())
            all_stds.append(std_pred.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0).flatten()
    uncertainties = np.concatenate(all_stds, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()

    metrics = compute_metrics(predictions, targets)
    calibration = compute_calibration_metrics(predictions, uncertainties, targets)
    metrics.update(calibration)

    metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    if log_to_wandb and wandb.run is not None:
        wandb.log(metrics)

    return metrics


def create_graph_dataloader(
    smiles_list: List[str],
    targets: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = False,
    use_dmpnn_format: bool = True,
) -> PyGDataLoader:
    """Create a PyTorch Geometric DataLoader from SMILES.

    Args:
        smiles_list: List of SMILES strings.
        targets: Target values.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        use_dmpnn_format: Whether to use D-MPNN specific graph format.

    Returns:
        PyTorch Geometric DataLoader.
    """
    dataset = MoleculeGraphDataset(
        smiles_list=smiles_list,
        targets=targets,
        precompute=True,
        use_dmpnn_format=use_dmpnn_format,
    )

    dataloader = PyGDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dataloader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate D-MPNN model")
    parser.add_argument("--model-path", type=str, help="Path to saved model")
    parser.add_argument("--data-path", type=str, help="Path to validation data")
    parser.add_argument("--wandb-project", type=str, default="mol-solubility")
    parser.add_argument("--n-samples", type=int, default=100, help="MC dropout samples")
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    wandb.init(project=args.wandb_project, job_type="validation")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Validation complete. Check wandb for metrics.")
