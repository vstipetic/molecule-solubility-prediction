"""Pretraining script for Transformer model on ZINC-1M.

This script pretrains a MoleculeTransformer using masked language modeling (MLM)
on SMILES strings from the ZINC dataset.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb

from Models.transformer import (
    MoleculeTransformer,
    SMILESTokenizer,
    create_mlm_inputs,
)
from DataUtils.datasets import ZINCDataset
from DataUtils.utils import load_zinc_data
from DataUtils.collate import transformer_collate_fn


def build_vocab_from_data(smiles_list: List[str], tokenizer: SMILESTokenizer) -> SMILESTokenizer:
    """Build vocabulary from training data.

    Args:
        smiles_list: List of SMILES strings.
        tokenizer: Initial tokenizer.

    Returns:
        Updated tokenizer with expanded vocabulary.
    """
    new_tokens = set()
    for smiles in smiles_list:
        tokens = tokenizer.tokenize(smiles)
        for token in tokens:
            if token not in tokenizer.vocab:
                new_tokens.add(token)

    if new_tokens:
        print(f"Adding {len(new_tokens)} new tokens to vocabulary")
        tokenizer.add_tokens(list(new_tokens))

    return tokenizer


def train_epoch_mlm(
    model: MoleculeTransformer,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    tokenizer: SMILESTokenizer,
    device: torch.device,
    epoch: int,
    mask_prob: float = 0.15,
    log_to_wandb: bool = True,
) -> float:
    """Train for one epoch with MLM objective.

    Args:
        model: MoleculeTransformer model.
        train_loader: Training DataLoader.
        optimizer: Optimizer.
        tokenizer: SMILES tokenizer.
        device: Device.
        epoch: Current epoch number.
        mask_prob: Probability of masking tokens.
        log_to_wandb: Whether to log to wandb.

    Returns:
        Average training loss.
    """
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Create masked inputs
        masked_input_ids, labels = create_mlm_inputs(input_ids, tokenizer, mask_prob)
        masked_input_ids = masked_input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass with MLM head
        mlm_logits = model.forward_mlm(masked_input_ids, attention_mask)

        # Compute loss
        loss = criterion(
            mlm_logits.view(-1, model.vocab_size),
            labels.view(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if log_to_wandb and wandb.run is not None and batch_idx % 100 == 0:
            wandb.log({
                'pretrain/batch_loss': loss.item(),
                'pretrain/epoch': epoch,
                'pretrain/batch': batch_idx,
            })

        if batch_idx % 500 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / n_batches
    return avg_loss


def evaluate_mlm(
    model: MoleculeTransformer,
    val_loader: DataLoader,
    tokenizer: SMILESTokenizer,
    device: torch.device,
    mask_prob: float = 0.15,
) -> Dict[str, float]:
    """Evaluate MLM model.

    Args:
        model: MoleculeTransformer model.
        val_loader: Validation DataLoader.
        tokenizer: SMILES tokenizer.
        device: Device.
        mask_prob: Probability of masking tokens.

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            masked_input_ids, labels = create_mlm_inputs(input_ids, tokenizer, mask_prob)
            masked_input_ids = masked_input_ids.to(device)
            labels = labels.to(device)

            mlm_logits = model.forward_mlm(masked_input_ids, attention_mask)

            loss = criterion(
                mlm_logits.view(-1, model.vocab_size),
                labels.view(-1)
            )

            # Compute accuracy on masked tokens
            predictions = mlm_logits.argmax(dim=-1)
            mask = labels != -100
            correct = (predictions == labels) & mask
            total_correct += correct.sum().item()
            total_masked += mask.sum().item()

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    accuracy = total_correct / max(total_masked, 1)

    return {'val_loss': avg_loss, 'val_accuracy': accuracy}


def pretrain_transformer(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    tokenizer: SMILESTokenizer,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 6,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    n_epochs: int = 10,
    mask_prob: float = 0.15,
    device: torch.device = None,
    log_to_wandb: bool = True,
    save_every: int = 1,
    save_dir: Optional[str] = None,
) -> MoleculeTransformer:
    """Pretrain transformer with MLM objective.

    Args:
        train_loader: Training DataLoader.
        val_loader: Optional validation DataLoader.
        tokenizer: SMILES tokenizer.
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer layers.
        dim_feedforward: FFN dimension.
        dropout: Dropout probability.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        n_epochs: Number of epochs.
        mask_prob: MLM mask probability.
        device: Device.
        log_to_wandb: Whether to log to wandb.
        save_every: Save checkpoint every N epochs.
        save_dir: Directory to save checkpoints.

    Returns:
        Pretrained MoleculeTransformer.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = MoleculeTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    # Log config
    if log_to_wandb and wandb.run is not None:
        wandb.config.update({
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'mask_prob': mask_prob,
            'vocab_size': tokenizer.vocab_size,
        })

    # Optimizer with warmup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * n_epochs
    warmup_steps = min(10000, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 1 - (step - warmup_steps) / (total_steps - warmup_steps))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    for epoch in range(n_epochs):
        print(f"\n=== Epoch {epoch + 1}/{n_epochs} ===")

        # Train
        train_loss = train_epoch_mlm(
            model, train_loader, optimizer, tokenizer, device,
            epoch, mask_prob, log_to_wandb
        )

        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        if val_loader is not None:
            val_metrics = evaluate_mlm(model, val_loader, tokenizer, device, mask_prob)
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Accuracy: {val_metrics['val_accuracy']:.4f}")

            if log_to_wandb and wandb.run is not None:
                wandb.log({
                    'pretrain/epoch': epoch + 1,
                    'pretrain/train_loss': train_loss,
                    'pretrain/val_loss': val_metrics['val_loss'],
                    'pretrain/val_accuracy': val_metrics['val_accuracy'],
                    'pretrain/learning_rate': scheduler.get_last_lr()[0],
                })
        else:
            if log_to_wandb and wandb.run is not None:
                wandb.log({
                    'pretrain/epoch': epoch + 1,
                    'pretrain/train_loss': train_loss,
                    'pretrain/learning_rate': scheduler.get_last_lr()[0],
                })

        # Save checkpoint
        if save_dir and (epoch + 1) % save_every == 0:
            checkpoint_path = Path(save_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Pretrain Transformer on ZINC")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to ZINC SMILES data")
    parser.add_argument("--val-path", type=str, default=None,
                        help="Path to validation data")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum training samples")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=1)
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
            name=args.wandb_name or "pretrain_transformer",
            config=vars(args),
            job_type="pretrain",
        )

    # Load data
    print(f"Loading data from {args.data_path}...")
    train_smiles = load_zinc_data(args.data_path, args.max_samples)
    print(f"Loaded {len(train_smiles)} SMILES for training")

    val_smiles = None
    if args.val_path:
        val_smiles = load_zinc_data(args.val_path)
        print(f"Loaded {len(val_smiles)} SMILES for validation")

    # Create tokenizer and expand vocabulary
    tokenizer = SMILESTokenizer(max_length=args.max_length)
    tokenizer = build_vocab_from_data(train_smiles, tokenizer)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create datasets
    train_dataset = ZINCDataset(train_smiles, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=transformer_collate_fn, num_workers=4, pin_memory=True
    )

    val_loader = None
    if val_smiles:
        val_dataset = ZINCDataset(val_smiles, tokenizer, args.max_length)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=transformer_collate_fn, num_workers=4
        )

    # Pretrain
    model = pretrain_transformer(
        train_loader, val_loader, tokenizer,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        mask_prob=args.mask_prob,
        device=device,
        log_to_wandb=not args.no_wandb,
        save_every=args.save_every,
        save_dir=args.save_dir,
    )

    # Save final model
    if args.save_dir:
        save_path = Path(args.save_dir) / "pretrained_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer_vocab': tokenizer.vocab,
            'config': {
                'd_model': args.d_model,
                'nhead': args.nhead,
                'num_layers': args.num_layers,
                'dim_feedforward': args.dim_feedforward,
                'dropout': args.dropout,
            }
        }, save_path)
        print(f"Final model saved to {save_path}")

        if not args.no_wandb and wandb.run is not None:
            wandb.save(str(save_path))

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
