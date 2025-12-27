"""ChemBERTa model wrapper for molecular property prediction.

This module provides a fine-tuning wrapper around the pretrained ChemBERTa-77M
model from HuggingFace for solubility prediction.

Model: seyonec/ChemBERTa-zinc-base-v1
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, RobertaModel

from Models.layers import MCDropout, MCDropoutMixin


class ChemBERTaForSolubility(nn.Module, MCDropoutMixin):
    """ChemBERTa model fine-tuned for solubility prediction.

    Wraps the pretrained ChemBERTa-77M model and adds a regression head
    with MCDropout for uncertainty estimation.

    Args:
        model_name: HuggingFace model name or path.
        dropout: Dropout probability for regression head.
        freeze_encoder: Whether to freeze the encoder layers initially.
        num_outputs: Number of output values.

    Attributes:
        encoder: Pretrained ChemBERTa encoder.
        tokenizer: ChemBERTa tokenizer.
        regression_head: MLP for regression with MCDropout.

    Example:
        >>> model = ChemBERTaForSolubility()
        >>> smiles = ["CCO", "CC(=O)O"]
        >>> outputs = model.predict_from_smiles(smiles)
    """

    DEFAULT_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        num_outputs: int = 1,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.freeze_encoder = freeze_encoder

        # Load pretrained model and tokenizer
        self.encoder: RobertaModel = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Get hidden size from config
        hidden_size = self.encoder.config.hidden_size

        # Regression head with MCDropout
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            MCDropout(dropout),
            nn.Linear(hidden_size, num_outputs),
        )

        # Optionally freeze encoder
        if freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_encoder_layers(self, num_layers: Optional[int] = None) -> None:
        """Freeze a specific number of encoder layers.

        Args:
            num_layers: Number of layers to freeze from the bottom.
                       If None, freezes all layers.
        """
        if num_layers is None:
            self._freeze_encoder()
            return

        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        # Freeze specified number of layers
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters including encoder."""
        self._unfreeze_encoder()
        for param in self.regression_head.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            output_hidden_states: Whether to return hidden states.

        Returns:
            Dictionary with 'logits' and optionally 'hidden_states'.
        """
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        # Use [CLS] token representation (first token)
        cls_output = encoder_outputs.last_hidden_state[:, 0, :]

        # Regression head
        logits = self.regression_head(cls_output)

        output = {'logits': logits}
        if output_hidden_states:
            output['hidden_states'] = encoder_outputs.hidden_states

        return output

    def encode_smiles(
        self,
        smiles_list: List[str],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Encode SMILES strings using the tokenizer.

        Args:
            smiles_list: List of SMILES strings.
            max_length: Maximum sequence length.
            padding: Whether to pad sequences.
            truncation: Whether to truncate long sequences.

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'.
        """
        encoded = self.tokenizer(
            smiles_list,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors='pt',
        )
        return encoded

    def predict_from_smiles(
        self,
        smiles_list: List[str],
        max_length: int = 512,
    ) -> torch.Tensor:
        """Predict solubility directly from SMILES strings.

        Args:
            smiles_list: List of SMILES strings.
            max_length: Maximum sequence length.

        Returns:
            Predictions [batch_size, num_outputs].
        """
        encoded = self.encode_smiles(smiles_list, max_length)

        # Move to same device as model
        device = next(self.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

        return outputs['logits']

    def predict_with_uncertainty(
        self,
        smiles_list: List[str],
        n_samples: int = 100,
        max_length: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with MC dropout uncertainty estimation.

        Args:
            smiles_list: List of SMILES strings.
            n_samples: Number of forward passes for MC dropout.
            max_length: Maximum sequence length.

        Returns:
            Tuple of (mean_predictions, std_predictions).
        """
        encoded = self.encode_smiles(smiles_list, max_length)

        device = next(self.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # Enable MC dropout
        self.eval()
        self.enable_mc_dropout()

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.forward(input_ids, attention_mask)
                predictions.append(outputs['logits'])

        self.disable_mc_dropout()

        predictions_tensor = torch.stack(predictions, dim=0)
        mean = predictions_tensor.mean(dim=0)
        std = predictions_tensor.std(dim=0)

        return mean, std

    def get_embeddings(
        self,
        smiles_list: List[str],
        max_length: int = 512,
    ) -> torch.Tensor:
        """Get [CLS] token embeddings for SMILES strings.

        Args:
            smiles_list: List of SMILES strings.
            max_length: Maximum sequence length.

        Returns:
            Embeddings [batch_size, hidden_size].
        """
        encoded = self.encode_smiles(smiles_list, max_length)

        device = next(self.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        return encoder_outputs.last_hidden_state[:, 0, :]

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        model_name: str = DEFAULT_MODEL_NAME,
        **kwargs,
    ) -> "ChemBERTaForSolubility":
        """Load a fine-tuned model from checkpoint.

        Args:
            model_path: Path to saved model weights.
            model_name: Original HuggingFace model name.
            **kwargs: Additional arguments for model initialization.

        Returns:
            Loaded ChemBERTaForSolubility model.
        """
        model = cls(model_name=model_name, **kwargs)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_path: str) -> None:
        """Save model weights.

        Args:
            save_path: Path to save model weights.
        """
        torch.save(self.state_dict(), save_path)


def load_chemberta_tokenizer(
    model_name: str = ChemBERTaForSolubility.DEFAULT_MODEL_NAME,
) -> AutoTokenizer:
    """Load the ChemBERTa tokenizer.

    Args:
        model_name: HuggingFace model name.

    Returns:
        ChemBERTa tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name)
