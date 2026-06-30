"""Fine-tuning script for scratch-trained Transformer on solubility prediction.

This script fine-tunes a MoleculeTransformer that was pretrained on ZINC-1M
for the AqSolDB solubility prediction task.
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

from DataUtils.utils import load_splits
from DataUtils.datasets import TransformerSolubilityDataset
from DataUtils.collate import transformer_collate_fn
from Models.transformer import MoleculeTransformer, SMILESTokenizer
from Train.validate_transformer import (
    validate_transformer,
    validate_transformer_with_uncertainty,
    create_transformer_dataloader,
)
from Train.wandb_utils import init_run, finish_run


def create_dataloaders(
    train_smiles: List[str],
    train_targets: np.ndarray,
    val_smiles: List[str],
    val_targets: np.ndarray,
    tokenizer: SMILESTokenizer,
    batch_size: int = 32,
    max_length: int = 512,
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders for training and validation.

    Args:
        train_smiles: Training SMILES strings.
        train_targets: Training targets.
        val_smiles: Validation SMILES strings.
        val_targets: Validation targets.
        tokenizer: SMILES tokenizer.
        batch_size: Batch size.
        max_length: Maximum sequence length.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = TransformerSolubilityDataset(
        train_smiles, train_targets, tokenizer, max_length, is_huggingface=False
    )
    val_dataset = TransformerSolubilityDataset(
        val_smiles, val_targets, tokenizer, max_length, is_huggingface=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=transformer_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=transformer_collate_fn
    )

    return train_loader, val_loader


def train_epoch(
    model: MoleculeTransformer,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_to_wandb: bool = True,
) -> float:
    """Train for one epoch.

    Args:
        model: MoleculeTransformer model.
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


def finetune_transformer(
    model: MoleculeTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    n_epochs: int = 50,
    patience: int = 10,
    device: torch.device = None,
    log_to_wandb: bool = True,
) -> MoleculeTransformer:
    """Fine-tune transformer model.

    Args:
        model: MoleculeTransformer model.
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
        optimizer, mode='min', factor=0.5, patience=3
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


def load_pretrained_transformer(
    checkpoint_path: str,
    device: torch.device = None,
) -> Tuple[MoleculeTransformer, SMILESTokenizer]:
    """Load pretrained transformer from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint.
        device: Device.

    Returns:
        Tuple of (model, tokenizer).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Restore tokenizer
    tokenizer = SMILESTokenizer(vocab=checkpoint['tokenizer_vocab'])

    # Create model with saved config
    config = checkpoint['config']
    model = MoleculeTransformer(
        vocab_size=tokenizer.vocab_size,
        **config
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Transformer for solubility")
    parser.add_argument("--pretrained-path", type=str, required=True,
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Directory with train.csv, val.csv, and test.csv")
    parser.add_argument("--smiles-column", type=str, default="SMILES")
    parser.add_argument("--target-column", type=str, default="Solubility")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--mc-samples", type=int, default=100,
                        help="MC dropout samples")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="mol-solubility")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="wandb entity/team (default: WANDB_ENTITY env)")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="wandb group name (e.g. to group runpod runs)")
    parser.add_argument("--wandb-tags", type=str, default=None,
                        help="Comma/space separated wandb tags")
    parser.add_argument("--wandb-mode", type=str, default=None,
                        choices=["online", "offline", "disabled"],
                        help="Force wandb mode (default: online if WANDB_API_KEY "
                             "set, else offline)")
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize wandb
    if not args.no_wandb:
        init_run(
            project=args.wandb_project,
            name=args.wandb_name or "finetune_transformer",
            config=vars(args),
            job_type="finetune",
            tags=args.wandb_tags,
            group=args.wandb_group,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
        )

    # Load pre-built splits
    print(f"Loading splits from {args.data_path}...")
    (train_smiles, train_targets), (val_smiles, val_targets), (test_smiles, test_targets) = \
        load_splits(args.data_path, args.smiles_column, args.target_column)

    print(f"Train: {len(train_smiles)}, Val: {len(val_smiles)}, Test: {len(test_smiles)}")

    # Load pretrained model
    print(f"Loading pretrained transformer from {args.pretrained_path}...")
    model, tokenizer = load_pretrained_transformer(args.pretrained_path, device)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_smiles, train_targets,
        val_smiles, val_targets,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Create test loader
    test_loader = create_transformer_dataloader(
        test_smiles, test_targets, tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
    )

    # Fine-tune
    print("\nFine-tuning model...")
    model = finetune_transformer(
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
    test_metrics = validate_transformer_with_uncertainty(
        model, test_loader, device,
        n_samples=args.mc_samples,
        log_to_wandb=not args.no_wandb,
        prefix='test'
    )

    print(f"Test RMSE: {test_metrics['test/rmse']:.4f}")
    print(f"Test MAE: {test_metrics['test/mae']:.4f}")

    # Save model
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer_vocab': tokenizer.vocab,
        }, save_path)

        print(f"Model saved to {save_path}")

        if not args.no_wandb and wandb.run is not None:
            wandb.save(str(save_path))

    if not args.no_wandb:
        finish_run()


if __name__ == "__main__":
    main()
