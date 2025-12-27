"""Training script for D-MPNN (GNN) model.

This script trains a D-MPNN model on molecular solubility data
with wandb logging and support for MC dropout uncertainty.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGDataLoader
import wandb

from DataUtils.graph_preprocessing import MoleculeGraphDataset
from DataUtils.utils import scaffold_split
from Models.dmpnn import DMPNN, create_dmpnn_ensemble
from Train.validate_gnn import (
    validate_gnn,
    validate_gnn_with_uncertainty,
    validate_gnn_ensemble,
)


def load_data(
    data_path: str,
    smiles_column: str = "SMILES",
    target_column: str = "Solubility",
) -> Tuple[List[str], np.ndarray]:
    """Load data from CSV file."""
    df = pd.read_csv(data_path)
    smiles_list = df[smiles_column].tolist()
    targets = df[target_column].values.astype(np.float32)
    return smiles_list, targets


def create_dataloaders(
    train_smiles: List[str],
    train_targets: np.ndarray,
    val_smiles: List[str],
    val_targets: np.ndarray,
    test_smiles: List[str],
    test_targets: np.ndarray,
    batch_size: int = 32,
    use_dmpnn_format: bool = True,
) -> Tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader]:
    """Create PyTorch Geometric DataLoaders.

    Args:
        train_smiles: Training SMILES.
        train_targets: Training targets.
        val_smiles: Validation SMILES.
        val_targets: Validation targets.
        test_smiles: Test SMILES.
        test_targets: Test targets.
        batch_size: Batch size.
        use_dmpnn_format: Whether to use D-MPNN specific format.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_dataset = MoleculeGraphDataset(
        train_smiles, train_targets, precompute=True, use_dmpnn_format=use_dmpnn_format
    )
    val_dataset = MoleculeGraphDataset(
        val_smiles, val_targets, precompute=True, use_dmpnn_format=use_dmpnn_format
    )
    test_dataset = MoleculeGraphDataset(
        test_smiles, test_targets, precompute=True, use_dmpnn_format=use_dmpnn_format
    )

    train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_epoch(
    model: DMPNN,
    train_loader: PyGDataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_to_wandb: bool = True,
) -> float:
    """Train for one epoch.

    Args:
        model: D-MPNN model.
        train_loader: Training DataLoader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device.
        epoch: Current epoch number.
        log_to_wandb: Whether to log to wandb.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)

        optimizer.zero_grad()
        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(outputs.squeeze(-1), batch.y.squeeze(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if log_to_wandb and wandb.run is not None and batch_idx % 10 == 0:
            wandb.log({
                'train/batch_loss': loss.item(),
                'epoch': epoch,
                'batch': batch_idx,
            })

    avg_loss = total_loss / n_batches
    return avg_loss


def train_gnn(
    train_loader: PyGDataLoader,
    val_loader: PyGDataLoader,
    hidden_size: int = 300,
    depth: int = 3,
    ffn_hidden_size: int = 300,
    ffn_num_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    n_epochs: int = 100,
    patience: int = 10,
    device: torch.device = None,
    log_to_wandb: bool = True,
) -> DMPNN:
    """Train D-MPNN model.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        hidden_size: Hidden size for D-MPNN encoder.
        depth: Number of message passing iterations.
        ffn_hidden_size: Hidden size for FFN.
        ffn_num_layers: Number of FFN layers.
        dropout: Dropout probability.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        n_epochs: Maximum number of epochs.
        patience: Early stopping patience.
        device: Device to train on.
        log_to_wandb: Whether to log to wandb.

    Returns:
        Trained D-MPNN model.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = DMPNN(
        hidden_size=hidden_size,
        depth=depth,
        ffn_hidden_size=ffn_hidden_size,
        ffn_num_layers=ffn_num_layers,
        dropout=dropout,
    ).to(device)

    # Log config
    if log_to_wandb and wandb.run is not None:
        wandb.config.update({
            'hidden_size': hidden_size,
            'depth': depth,
            'ffn_hidden_size': ffn_hidden_size,
            'ffn_num_layers': ffn_num_layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
        })

    # Optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()

    # Training loop
    best_val_rmse = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, log_to_wandb
        )

        # Validate
        val_metrics = validate_gnn(
            model, val_loader, device,
            log_to_wandb=log_to_wandb, prefix='val'
        )

        # Update scheduler
        scheduler.step(val_metrics['val/rmse'])

        # Log
        print(f"Epoch {epoch + 1}/{n_epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val RMSE: {val_metrics['val/rmse']:.4f}")

        if log_to_wandb and wandb.run is not None:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
            })

        # Early stopping
        if val_metrics['val/rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['val/rmse']
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def train_ensemble(
    train_loader: PyGDataLoader,
    val_loader: PyGDataLoader,
    n_models: int = 5,
    device: torch.device = None,
    log_to_wandb: bool = True,
    **model_kwargs,
) -> List[DMPNN]:
    """Train an ensemble of D-MPNN models.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        n_models: Number of models in ensemble.
        device: Device to train on.
        log_to_wandb: Whether to log to wandb.
        **model_kwargs: Arguments passed to train_gnn.

    Returns:
        List of trained D-MPNN models.
    """
    models = []

    for i in range(n_models):
        print(f"\n=== Training ensemble model {i + 1}/{n_models} ===")

        model = train_gnn(
            train_loader, val_loader,
            device=device,
            log_to_wandb=log_to_wandb,
            **model_kwargs,
        )
        models.append(model)

    return models


def main():
    parser = argparse.ArgumentParser(description="Train D-MPNN model")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to training data CSV")
    parser.add_argument("--smiles-column", type=str, default="SMILES")
    parser.add_argument("--target-column", type=str, default="Solubility")
    parser.add_argument("--hidden-size", type=int, default=300)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--ffn-hidden-size", type=int, default=300)
    parser.add_argument("--ffn-num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--ensemble", action="store_true", help="Train ensemble")
    parser.add_argument("--n-models", type=int, default=5, help="Ensemble size")
    parser.add_argument("--mc-samples", type=int, default=100,
                        help="MC dropout samples for uncertainty")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="mol-solubility")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            job_type="train",
        )

    # Load data
    print(f"Loading data from {args.data_path}...")
    smiles_list, targets = load_data(
        args.data_path,
        args.smiles_column,
        args.target_column,
    )
    print(f"Loaded {len(smiles_list)} molecules")

    # Scaffold split
    print("Performing scaffold split...")
    (train_smiles, train_targets), (val_smiles, val_targets), (test_smiles, test_targets) = \
        scaffold_split(smiles_list, targets, args.train_ratio, args.val_ratio)

    print(f"Train: {len(train_smiles)}, Val: {len(val_smiles)}, Test: {len(test_smiles)}")

    # Create dataloaders
    print("Creating graph datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_smiles, train_targets,
        val_smiles, val_targets,
        test_smiles, test_targets,
        batch_size=args.batch_size,
    )

    # Training
    model_kwargs = {
        'hidden_size': args.hidden_size,
        'depth': args.depth,
        'ffn_hidden_size': args.ffn_hidden_size,
        'ffn_num_layers': args.ffn_num_layers,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'n_epochs': args.n_epochs,
        'patience': args.patience,
    }

    if args.ensemble:
        models = train_ensemble(
            train_loader, val_loader,
            n_models=args.n_models,
            device=device,
            log_to_wandb=not args.no_wandb,
            **model_kwargs,
        )

        # Evaluate ensemble
        print("\nEvaluating ensemble on test set...")
        test_metrics = validate_gnn_ensemble(
            models, test_loader, device,
            log_to_wandb=not args.no_wandb, prefix='test'
        )
    else:
        model = train_gnn(
            train_loader, val_loader,
            device=device,
            log_to_wandb=not args.no_wandb,
            **model_kwargs,
        )

        # Evaluate with MC dropout
        print("\nEvaluating with MC dropout on test set...")
        test_metrics = validate_gnn_with_uncertainty(
            model, test_loader, device,
            n_samples=args.mc_samples,
            log_to_wandb=not args.no_wandb, prefix='test'
        )

    print(f"Test RMSE: {test_metrics['test/rmse']:.4f}")
    print(f"Test MAE: {test_metrics['test/mae']:.4f}")

    # Save model
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if args.ensemble:
            for i, m in enumerate(models):
                torch.save(m.state_dict(), save_path.with_suffix(f'.{i}.pt'))
        else:
            torch.save(model.state_dict(), save_path)

        print(f"Model saved to {save_path}")

        if not args.no_wandb and wandb.run is not None:
            wandb.save(str(save_path))

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
