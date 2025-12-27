"""Transformer model for molecular property prediction from SMILES.

This module implements a Transformer encoder trained from scratch on SMILES
strings, with support for MC dropout uncertainty estimation.

The model can be pretrained on ZINC-1M using masked language modeling (MLM)
and then fine-tuned on solubility prediction.
"""

import math
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.layers import MCDropout, MCDropoutMixin


# Atom-aware SMILES tokenization regex pattern
SMILES_REGEX = r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


class SMILESTokenizer:
    """Atom-aware tokenizer for SMILES strings.

    Uses regex to split SMILES into chemically meaningful tokens like atoms,
    bonds, and special characters.

    Args:
        vocab: Optional pre-built vocabulary dictionary.
        max_length: Maximum sequence length (includes special tokens).

    Attributes:
        vocab: Token to index mapping.
        inv_vocab: Index to token mapping.
        pad_token: Padding token string.
        unk_token: Unknown token string.
        cls_token: Classification token string.
        mask_token: Mask token for MLM pretraining.

    Example:
        >>> tokenizer = SMILESTokenizer()
        >>> tokens = tokenizer.tokenize("CCO")
        >>> print(tokens)  # ['[CLS]', 'C', 'C', 'O']
        >>> ids = tokenizer.encode("CCO", max_length=10)
    """

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        max_length: int = 512,
    ) -> None:
        self.max_length = max_length
        self.pattern = re.compile(SMILES_REGEX)

        # Special tokens
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"

        if vocab is not None:
            self.vocab = vocab
        else:
            self._build_default_vocab()

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        # Token IDs
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.cls_token_id = self.vocab[self.cls_token]
        self.sep_token_id = self.vocab[self.sep_token]
        self.mask_token_id = self.vocab[self.mask_token]

    def _build_default_vocab(self) -> None:
        """Build default vocabulary with common SMILES tokens."""
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.cls_token,
            self.sep_token,
            self.mask_token,
        ]

        # Common SMILES tokens
        atoms = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si',
                 'c', 'n', 'o', 's', 'p']
        bonds = ['=', '#', '-', '/', '\\', ':']
        numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        brackets = ['(', ')', '[', ']']
        special = ['.', '+', '@', '@@', '%']

        # Common bracketed atoms
        bracketed = [
            '[H]', '[C]', '[N]', '[O]', '[S]', '[P]', '[F]', '[Cl]', '[Br]', '[I]',
            '[B]', '[Si]', '[Se]', '[Te]', '[nH]', '[NH]', '[NH2]', '[NH3+]',
            '[N+]', '[N-]', '[O-]', '[S-]', '[S+]', '[OH]', '[O+]',
            '[C@H]', '[C@@H]', '[C@]', '[C@@]',
            '[n+]', '[n-]', '[Na]', '[K]', '[Ca]', '[Mg]', '[Fe]', '[Zn]', '[Cu]',
            '[Na+]', '[K+]', '[Ca+2]', '[Mg+2]',
        ]

        all_tokens = (
            special_tokens + atoms + bonds + numbers + brackets + special + bracketed
        )

        self.vocab = {token: idx for idx, token in enumerate(all_tokens)}

    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize a SMILES string into tokens.

        Args:
            smiles: SMILES string to tokenize.

        Returns:
            List of tokens (includes [CLS] at start).
        """
        tokens = [self.cls_token]
        matches = self.pattern.findall(smiles)
        tokens.extend(matches)
        return tokens

    def encode(
        self,
        smiles: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        return_tensors: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Encode a SMILES string to token IDs.

        Args:
            smiles: SMILES string to encode.
            max_length: Maximum length (default: self.max_length).
            padding: Whether to pad to max_length.
            return_tensors: Whether to return PyTorch tensors.

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'.
        """
        if max_length is None:
            max_length = self.max_length

        tokens = self.tokenize(smiles)

        # Convert to IDs
        input_ids = []
        for token in tokens[:max_length]:
            if token in self.vocab:
                input_ids.append(self.vocab[token])
            else:
                input_ids.append(self.unk_token_id)

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        # Padding
        if padding:
            pad_length = max_length - len(input_ids)
            input_ids.extend([self.pad_token_id] * pad_length)
            attention_mask.extend([0] * pad_length)

        if return_tensors:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            }
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

    def batch_encode(
        self,
        smiles_list: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Encode a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings.
            max_length: Maximum length.
            padding: Whether to pad.

        Returns:
            Dictionary with batched 'input_ids' and 'attention_mask'.
        """
        encoded = [
            self.encode(smiles, max_length, padding, return_tensors=False)
            for smiles in smiles_list
        ]

        input_ids = torch.tensor([e['input_ids'] for e in encoded], dtype=torch.long)
        attention_mask = torch.tensor(
            [e['attention_mask'] for e in encoded], dtype=torch.long
        )

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to SMILES string.

        Args:
            token_ids: List of token IDs.

        Returns:
            Reconstructed SMILES string.
        """
        tokens = []
        for tid in token_ids:
            if tid in self.inv_vocab:
                token = self.inv_vocab[tid]
                if token not in [self.pad_token, self.cls_token, self.sep_token]:
                    tokens.append(token)
        return ''.join(tokens)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    def add_tokens(self, tokens: List[str]) -> int:
        """Add new tokens to vocabulary.

        Args:
            tokens: List of new tokens to add.

        Returns:
            Number of tokens added.
        """
        added = 0
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        return added


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = MCDropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor [batch_size, seq_len, d_model].

        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with MCDropout.

    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # MC Dropout layers
        self.dropout1 = MCDropout(dropout)
        self.dropout2 = MCDropout(dropout)
        self.dropout_ffn = MCDropout(dropout)

        self.activation = nn.GELU()

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of transformer encoder layer.

        Args:
            src: Input tensor [batch_size, seq_len, d_model].
            src_mask: Attention mask.
            src_key_padding_mask: Padding mask [batch_size, seq_len].

        Returns:
            Output tensor [batch_size, seq_len, d_model].
        """
        # Self-attention
        attn_output, _ = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feedforward
        ff_output = self.linear2(self.dropout_ffn(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src


class MoleculeTransformer(nn.Module, MCDropoutMixin):
    """Transformer model for molecular property prediction.

    This model can be used for:
    1. Masked language modeling (pretraining on ZINC-1M)
    2. Regression (fine-tuning on solubility)

    Args:
        vocab_size: Size of vocabulary.
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer layers.
        dim_feedforward: FFN hidden dimension.
        max_seq_length: Maximum sequence length.
        dropout: Dropout probability.
        num_outputs: Number of output values (1 for regression).

    Example:
        >>> tokenizer = SMILESTokenizer()
        >>> model = MoleculeTransformer(vocab_size=tokenizer.vocab_size)
        >>> encoded = tokenizer.encode("CCO")
        >>> output = model(encoded['input_ids'].unsqueeze(0))
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        num_outputs: int = 1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Output heads
        self.mlm_head = nn.Linear(d_model, vocab_size)  # For pretraining
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            MCDropout(dropout),
            nn.Linear(d_model, num_outputs),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.mlm_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for regression task.

        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Padding mask [batch_size, seq_len].
            output_hidden_states: Whether to return hidden states.

        Returns:
            Dictionary with 'logits' and optionally 'hidden_states'.
        """
        # Embedding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Create padding mask for attention (True where padded)
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None

        # Transformer encoder
        hidden_states = []
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
            if output_hidden_states:
                hidden_states.append(x)

        # Use [CLS] token (first token) for regression
        cls_output = x[:, 0, :]
        logits = self.regression_head(cls_output)

        output = {'logits': logits}
        if output_hidden_states:
            output['hidden_states'] = hidden_states

        return output

    def forward_mlm(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for masked language modeling.

        Args:
            input_ids: Token IDs (with some masked) [batch_size, seq_len].
            attention_mask: Padding mask [batch_size, seq_len].

        Returns:
            MLM logits [batch_size, seq_len, vocab_size].
        """
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None

        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)

        mlm_logits = self.mlm_head(x)
        return mlm_logits

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get [CLS] token embeddings.

        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Padding mask [batch_size, seq_len].

        Returns:
            Embeddings [batch_size, d_model].
        """
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None

        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)

        return x[:, 0, :]


def create_mlm_inputs(
    input_ids: torch.Tensor,
    tokenizer: SMILESTokenizer,
    mask_prob: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create masked inputs for MLM pretraining.

    Args:
        input_ids: Original token IDs [batch_size, seq_len].
        tokenizer: SMILES tokenizer.
        mask_prob: Probability of masking a token.

    Returns:
        Tuple of (masked_input_ids, labels).
        Labels have -100 for non-masked positions (ignored in loss).
    """
    labels = input_ids.clone()
    masked_input_ids = input_ids.clone()

    # Create mask (don't mask special tokens)
    special_tokens = {
        tokenizer.pad_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
    }

    probability_matrix = torch.full(labels.shape, mask_prob)

    # Don't mask special tokens
    for special_id in special_tokens:
        probability_matrix[input_ids == special_id] = 0.0

    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Set labels to -100 for non-masked positions
    labels[~masked_indices] = -100

    # 80% of time, replace with [MASK]
    indices_replaced = torch.bernoulli(
        torch.full(labels.shape, 0.8)
    ).bool() & masked_indices
    masked_input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% of time, replace with random token
    indices_random = torch.bernoulli(
        torch.full(labels.shape, 0.5)
    ).bool() & masked_indices & ~indices_replaced
    random_tokens = torch.randint(
        len(special_tokens), tokenizer.vocab_size, labels.shape, dtype=torch.long
    )
    masked_input_ids[indices_random] = random_tokens[indices_random]

    # 10% of time, keep original (already done, nothing to change)

    return masked_input_ids, labels
