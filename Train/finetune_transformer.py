"""Fine-tuning script for Transformer models on solubility prediction.

This script fine-tunes either:
1. A scratch-trained MoleculeTransformer (pretrained on ZINC)
2. A pretrained ChemBERTa model

Both models are fine-tuned on the AqSolDB solubility dataset.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb

from DataUtils.utils import scaffold_split
from Models.transformer import MoleculeTransformer, SMILESTokenizer
from Models.chemberta import ChemBERTaForSolubility
from Train.validate_transformer import (
    validate_transformer,
    validate_transformer_with_uncertainty,
    validate_chemberta,
    validate_chemberta_with_uncertainty,
    create_transformer_dataloader,
)


class SolubilityDataset(Dataset):
    """Dataset for solubility regression with transformers."""

    def __init__(
        self,
        smiles_list: List[str],
        targets: np.ndarray,
        tokenizer: Union[SMILESTokenizer, None],
        max_length: int = 512,
        is_chemberta: bool = False,
    ) -> None:
        self.smiles_list = smiles_list
        self.targets = targets.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_chemberta = is_chemberta

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        smiles = self.smiles_list[idx]
        target = self.targets[idx]

        if self.is_chemberta:
            # ChemBERTa uses HuggingFace tokenizer
            encoded = self.tokenizer(
                smiles,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            )
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'labels': torch.tensor(target, dtype=torch.float32),
            }
        else:
            encoded = self.tokenizer.encode(smiles, max_length=self.max_length)
            return {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': torch.tensor(target, dtype=torch.float32),
            }


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
    tokenizer: Union[SMILESTokenizer, any],
    batch_size: int = 32,
    max_length: int = 512,
    is_chemberta: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders for training and validation."""

    def collate_fn(batch):
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
        }

    train_dataset = SolubilityDataset(
        train_smiles, train_targets, tokenizer, max_length, is_chemberta
    )
    val_dataset = SolubilityDataset(
        val_smiles, val_targets, tokenizer, max_length, is_chemberta
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
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
    """Train for one epoch."""
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
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    n_epochs: int = 50,
    patience: int = 10,
    device: torch.device = None,
    log_to_wandb: bool = True,
) -> nn.Module:
    """Fine-tune transformer model.

    Args:
        model: MoleculeTransformer or ChemBERTaForSolubility model.
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
    parser.add_argument("--model-type", type=str, choices=['scratch', 'chemberta'],
                        default='chemberta', help="Model type to fine-tune")
    parser.add_argument("--pretrained-path", type=str, default=None,
                        help="Path to pretrained model (for scratch)")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to training data CSV")
    parser.add_argument("--smiles-column", type=str, default="SMILES")
    parser.add_argument("--target-column", type=str, default="Solubility")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder layers (ChemBERTa only)")
    parser.add_argument("--freeze-layers", type=int, default=None,
                        help="Number of encoder layers to freeze (ChemBERTa)")
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
            name=args.wandb_name or f"finetune_{args.model_type}",
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

    # Create model and tokenizer
    if args.model_type == 'chemberta':
        print("Loading ChemBERTa model...")
        model = ChemBERTaForSolubility(
            dropout=args.dropout,
            freeze_encoder=args.freeze_encoder,
        )
        tokenizer = model.tokenizer

        if args.freeze_layers is not None:
            model.freeze_encoder_layers(args.freeze_layers)
            print(f"Froze {args.freeze_layers} encoder layers")

        is_chemberta = True
    else:
        if args.pretrained_path is None:
            raise ValueError("--pretrained-path required for scratch model")

        print(f"Loading pretrained transformer from {args.pretrained_path}...")
        model, tokenizer = load_pretrained_transformer(args.pretrained_path, device)
        is_chemberta = False

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_smiles, train_targets,
        val_smiles, val_targets,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        is_chemberta=is_chemberta,
    )

    # Create test loader
    test_dataset = SolubilityDataset(
        test_smiles, test_targets, tokenizer, args.max_length, is_chemberta
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda batch: {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
        }
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
    if is_chemberta:
        test_metrics = validate_chemberta_with_uncertainty(
            model, test_smiles, test_targets,
            n_samples=args.mc_samples,
            batch_size=args.batch_size,
            device=device,
            log_to_wandb=not args.no_wandb,
            prefix='test'
        )
    else:
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

        if is_chemberta:
            model.save_pretrained(str(save_path))
        else:
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer_vocab': tokenizer.vocab,
            }, save_path)

        print(f"Model saved to {save_path}")

        if not args.no_wandb and wandb.run is not None:
            wandb.save(str(save_path))

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
