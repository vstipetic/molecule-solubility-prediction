"""Fine-tuning script for ChemBERTa model on solubility prediction.

This script fine-tunes the pretrained ChemBERTa model (seyonec/ChemBERTa-zinc-base-v1)
on the AqSolDB solubility dataset.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from DataUtils.utils import scaffold_split, load_data
from DataUtils.datasets import TransformerSolubilityDataset
from DataUtils.collate import transformer_collate_fn
from Models.chemberta import ChemBERTaForSolubility
from Train.validate_transformer import (
    validate_chemberta,
    validate_chemberta_with_uncertainty,
)


def create_dataloaders(
    train_smiles: List[str],
    train_targets: np.ndarray,
    val_smiles: List[str],
    val_targets: np.ndarray,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 512,
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders for training and validation.

    Args:
        train_smiles: Training SMILES strings.
        train_targets: Training targets.
        val_smiles: Validation SMILES strings.
        val_targets: Validation targets.
        tokenizer: HuggingFace tokenizer from ChemBERTa.
        batch_size: Batch size.
        max_length: Maximum sequence length.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = TransformerSolubilityDataset(
        train_smiles, train_targets, tokenizer, max_length, is_huggingface=True
    )
    val_dataset = TransformerSolubilityDataset(
        val_smiles, val_targets, tokenizer, max_length, is_huggingface=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=transformer_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=transformer_collate_fn
    )

    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_to_wandb: bool = True,
) -> float:
    """Train for one epoch.

    Args:
        model: ChemBERTaForSolubility model.
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
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        predictions = outputs['logits'].squeeze(-1)

        loss = criterion(predictions, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if log_to_wandb and wandb.run is not None and batch_idx % 50 == 0:
            wandb.log({
                'finetune/batch_loss': loss.item(),
                'finetune/epoch': epoch,
            })

    return total_loss / n_batches


def finetune_chemberta(
    model: ChemBERTaForSolubility,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    n_epochs: int = 50,
    patience: int = 10,
    device: torch.device = None,
    log_to_wandb: bool = True,
) -> ChemBERTaForSolubility:
    """Fine-tune ChemBERTa model.

    Args:
        model: ChemBERTaForSolubility model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        n_epochs: Maximum epochs.
        patience: Early stopping patience.
        device: Device.
        log_to_wandb: Whether to log to wandb.

    Returns:
        Fine-tuned model.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, log_to_wandb
        )

        # Validate
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']

                outputs = model(input_ids, attention_mask)
                predictions = outputs['logits'].squeeze(-1)

                val_preds.append(predictions.cpu())
                val_labels.append(labels)

        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()

        val_rmse = np.sqrt(np.mean((val_preds - val_labels) ** 2))
        val_mae = np.mean(np.abs(val_preds - val_labels))

        scheduler.step(val_rmse)

        print(f"Epoch {epoch + 1}/{n_epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val RMSE: {val_rmse:.4f} - "
              f"Val MAE: {val_mae:.4f}")

        if log_to_wandb and wandb.run is not None:
            wandb.log({
                'finetune/epoch': epoch + 1,
                'finetune/train_loss': train_loss,
                'finetune/val_rmse': val_rmse,
                'finetune/val_mae': val_mae,
                'finetune/learning_rate': optimizer.param_groups[0]['lr'],
            })

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
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


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ChemBERTa for solubility")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to training data CSV")
    parser.add_argument("--smiles-column", type=str, default="SMILES")
    parser.add_argument("--target-column", type=str, default="Solubility")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder layers")
    parser.add_argument("--freeze-layers", type=int, default=None,
                        help="Number of encoder layers to freeze")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mc-samples", type=int, default=100,
                        help="MC dropout samples")
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
            name=args.wandb_name or "finetune_chemberta",
            config=vars(args),
            job_type="finetune",
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

    # Create model
    print("Loading ChemBERTa model...")
    model = ChemBERTaForSolubility(
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder,
    )
    tokenizer = model.tokenizer

    if args.freeze_layers is not None:
        model.freeze_encoder_layers(args.freeze_layers)
        print(f"Froze {args.freeze_layers} encoder layers")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_smiles, train_targets,
        val_smiles, val_targets,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Create test dataset
    test_dataset = TransformerSolubilityDataset(
        test_smiles, test_targets, tokenizer, args.max_length, is_huggingface=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=transformer_collate_fn
    )

    # Fine-tune
    print("\nFine-tuning ChemBERTa...")
    model = finetune_chemberta(
        model, train_loader, val_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        patience=args.patience,
        device=device,
        log_to_wandb=not args.no_wandb,
    )

    # Evaluate on test set with uncertainty
    print("\nEvaluating on test set with MC dropout...")
    test_metrics = validate_chemberta_with_uncertainty(
        model, test_smiles, test_targets,
        n_samples=args.mc_samples,
        batch_size=args.batch_size,
        device=device,
        log_to_wandb=not args.no_wandb,
        prefix='test'
    )

    print(f"Test RMSE: {test_metrics['test/rmse']:.4f}")
    print(f"Test MAE: {test_metrics['test/mae']:.4f}")

    # Save model
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_path))
        print(f"Model saved to {save_path}")

        if not args.no_wandb and wandb.run is not None:
            wandb.save(str(save_path))

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
