"""Dataset classes for molecular solubility prediction.

This module provides PyTorch Dataset classes for:
- AqSolDB: Main training dataset
- ESOL: Held-out distribution shift test set
"""

from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from DataUtils.utils import compute_ecfp_from_smiles, smiles_to_mol


class AqSolDBDataset(Dataset):
    """Dataset class for the AqSolDB aqueous solubility dataset.

    This dataset contains aqueous solubility measurements for various compounds.
    Supports multiple output modes: raw SMILES, fingerprints, or preprocessed
    for specific model types.

    Args:
        data_path: Path to the CSV file containing the dataset.
        smiles_column: Name of the column containing SMILES strings.
        target_column: Name of the column containing solubility values.
        mode: Output mode - 'smiles', 'fingerprint', or 'index'.
        fingerprint_radius: Radius for ECFP fingerprint computation.
        fingerprint_bits: Number of bits for fingerprint.
        transform: Optional transform to apply to SMILES.
        target_transform: Optional transform to apply to targets.

    Attributes:
        smiles: List of SMILES strings.
        targets: Numpy array of solubility values.
        mode: Current output mode.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        smiles_column: str = "SMILES",
        target_column: str = "Solubility",
        mode: Literal["smiles", "fingerprint", "index"] = "smiles",
        fingerprint_radius: int = 2,
        fingerprint_bits: int = 2048,
        transform: Optional[Callable[[str], torch.Tensor]] = None,
        target_transform: Optional[Callable[[float], torch.Tensor]] = None,
    ) -> None:
        super().__init__()

        self.data_path = Path(data_path)
        self.mode = mode
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_bits = fingerprint_bits
        self.transform = transform
        self.target_transform = target_transform

        # Load data
        df = pd.read_csv(self.data_path)

        # Filter valid SMILES
        valid_indices = []
        self.smiles: List[str] = []
        targets: List[float] = []

        for idx, row in df.iterrows():
            smiles = row[smiles_column]
            target = row[target_column]

            # Check if SMILES is valid
            mol = smiles_to_mol(smiles)
            if mol is not None and pd.notna(target):
                self.smiles.append(smiles)
                targets.append(float(target))
                valid_indices.append(idx)

        self.targets = np.array(targets, dtype=np.float32)

        # Precompute fingerprints if needed
        self._fingerprints: Optional[np.ndarray] = None
        if mode == "fingerprint":
            self._precompute_fingerprints()

    def _precompute_fingerprints(self) -> None:
        """Precompute fingerprints for all molecules."""
        fingerprints = []
        for smiles in self.smiles:
            fp = compute_ecfp_from_smiles(
                smiles,
                radius=self.fingerprint_radius,
                n_bits=self.fingerprint_bits,
            )
            if fp is not None:
                fingerprints.append(fp)
            else:
                # This shouldn't happen since we filtered invalid SMILES
                fingerprints.append(np.zeros(self.fingerprint_bits, dtype=np.float32))

        self._fingerprints = np.stack(fingerprints)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[str, float],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[int, str, float],
    ]:
        """Get a single item from the dataset.

        Returns:
            Depending on mode:
            - 'smiles': (smiles_string, target)
            - 'fingerprint': (fingerprint_tensor, target_tensor)
            - 'index': (index, smiles_string, target)
        """
        smiles = self.smiles[idx]
        target = self.targets[idx]

        if self.mode == "smiles":
            if self.transform is not None:
                smiles_out = self.transform(smiles)
            else:
                smiles_out = smiles

            if self.target_transform is not None:
                target_out = self.target_transform(target)
            else:
                target_out = target

            return smiles_out, target_out

        elif self.mode == "fingerprint":
            if self._fingerprints is None:
                self._precompute_fingerprints()

            fp = torch.from_numpy(self._fingerprints[idx])
            target_tensor = torch.tensor(target, dtype=torch.float32)

            return fp, target_tensor

        elif self.mode == "index":
            return idx, smiles, target

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_all_smiles(self) -> List[str]:
        """Get all SMILES strings in the dataset."""
        return self.smiles.copy()

    def get_all_targets(self) -> np.ndarray:
        """Get all target values in the dataset."""
        return self.targets.copy()

    def set_mode(self, mode: Literal["smiles", "fingerprint", "index"]) -> None:
        """Change the output mode of the dataset.

        Args:
            mode: New output mode.
        """
        self.mode = mode
        if mode == "fingerprint" and self._fingerprints is None:
            self._precompute_fingerprints()


class ESOLDataset(Dataset):
    """Dataset class for the ESOL (Delaney) solubility dataset.

    This dataset is used as a held-out test set to evaluate distribution shift.
    Contains measured aqueous solubility for drug-like compounds.

    Args:
        data_path: Path to the CSV file containing the dataset.
        smiles_column: Name of the column containing SMILES strings.
        target_column: Name of the column containing solubility values.
        mode: Output mode - 'smiles', 'fingerprint', or 'index'.
        fingerprint_radius: Radius for ECFP fingerprint computation.
        fingerprint_bits: Number of bits for fingerprint.
        transform: Optional transform to apply to SMILES.
        target_transform: Optional transform to apply to targets.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        smiles_column: str = "smiles",
        target_column: str = "measured log solubility in mols per litre",
        mode: Literal["smiles", "fingerprint", "index"] = "smiles",
        fingerprint_radius: int = 2,
        fingerprint_bits: int = 2048,
        transform: Optional[Callable[[str], torch.Tensor]] = None,
        target_transform: Optional[Callable[[float], torch.Tensor]] = None,
    ) -> None:
        super().__init__()

        self.data_path = Path(data_path)
        self.mode = mode
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_bits = fingerprint_bits
        self.transform = transform
        self.target_transform = target_transform

        # Load data
        df = pd.read_csv(self.data_path)

        # Filter valid SMILES
        self.smiles: List[str] = []
        targets: List[float] = []

        for idx, row in df.iterrows():
            smiles = row[smiles_column]
            target = row[target_column]

            mol = smiles_to_mol(smiles)
            if mol is not None and pd.notna(target):
                self.smiles.append(smiles)
                targets.append(float(target))

        self.targets = np.array(targets, dtype=np.float32)

        self._fingerprints: Optional[np.ndarray] = None
        if mode == "fingerprint":
            self._precompute_fingerprints()

    def _precompute_fingerprints(self) -> None:
        """Precompute fingerprints for all molecules."""
        fingerprints = []
        for smiles in self.smiles:
            fp = compute_ecfp_from_smiles(
                smiles,
                radius=self.fingerprint_radius,
                n_bits=self.fingerprint_bits,
            )
            if fp is not None:
                fingerprints.append(fp)
            else:
                fingerprints.append(np.zeros(self.fingerprint_bits, dtype=np.float32))

        self._fingerprints = np.stack(fingerprints)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[str, float],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[int, str, float],
    ]:
        """Get a single item from the dataset."""
        smiles = self.smiles[idx]
        target = self.targets[idx]

        if self.mode == "smiles":
            if self.transform is not None:
                smiles_out = self.transform(smiles)
            else:
                smiles_out = smiles

            if self.target_transform is not None:
                target_out = self.target_transform(target)
            else:
                target_out = target

            return smiles_out, target_out

        elif self.mode == "fingerprint":
            if self._fingerprints is None:
                self._precompute_fingerprints()

            fp = torch.from_numpy(self._fingerprints[idx])
            target_tensor = torch.tensor(target, dtype=torch.float32)

            return fp, target_tensor

        elif self.mode == "index":
            return idx, smiles, target

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_all_smiles(self) -> List[str]:
        """Get all SMILES strings in the dataset."""
        return self.smiles.copy()

    def get_all_targets(self) -> np.ndarray:
        """Get all target values in the dataset."""
        return self.targets.copy()

    def set_mode(self, mode: Literal["smiles", "fingerprint", "index"]) -> None:
        """Change the output mode of the dataset."""
        self.mode = mode
        if mode == "fingerprint" and self._fingerprints is None:
            self._precompute_fingerprints()


def create_subset_dataset(
    smiles_list: List[str],
    targets: np.ndarray,
    mode: Literal["smiles", "fingerprint", "index"] = "smiles",
    fingerprint_radius: int = 2,
    fingerprint_bits: int = 2048,
) -> "SubsetDataset":
    """Create a dataset from a subset of SMILES and targets.

    Useful for creating train/val/test splits.

    Args:
        smiles_list: List of SMILES strings.
        targets: Numpy array of target values.
        mode: Output mode.
        fingerprint_radius: Radius for ECFP fingerprint.
        fingerprint_bits: Number of bits for fingerprint.

    Returns:
        SubsetDataset instance.
    """
    return SubsetDataset(
        smiles_list=smiles_list,
        targets=targets,
        mode=mode,
        fingerprint_radius=fingerprint_radius,
        fingerprint_bits=fingerprint_bits,
    )


class SubsetDataset(Dataset):
    """Dataset created from a subset of SMILES and targets.

    Used for creating train/val/test splits after scaffold splitting.
    """

    def __init__(
        self,
        smiles_list: List[str],
        targets: np.ndarray,
        mode: Literal["smiles", "fingerprint", "index"] = "smiles",
        fingerprint_radius: int = 2,
        fingerprint_bits: int = 2048,
    ) -> None:
        super().__init__()

        self.smiles = smiles_list
        self.targets = targets.astype(np.float32)
        self.mode = mode
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_bits = fingerprint_bits

        self._fingerprints: Optional[np.ndarray] = None
        if mode == "fingerprint":
            self._precompute_fingerprints()

    def _precompute_fingerprints(self) -> None:
        """Precompute fingerprints for all molecules."""
        fingerprints = []
        for smiles in self.smiles:
            fp = compute_ecfp_from_smiles(
                smiles,
                radius=self.fingerprint_radius,
                n_bits=self.fingerprint_bits,
            )
            if fp is not None:
                fingerprints.append(fp)
            else:
                fingerprints.append(np.zeros(self.fingerprint_bits, dtype=np.float32))

        self._fingerprints = np.stack(fingerprints)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[str, float],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[int, str, float],
    ]:
        """Get a single item from the dataset."""
        smiles = self.smiles[idx]
        target = self.targets[idx]

        if self.mode == "smiles":
            return smiles, target

        elif self.mode == "fingerprint":
            if self._fingerprints is None:
                self._precompute_fingerprints()

            fp = torch.from_numpy(self._fingerprints[idx])
            target_tensor = torch.tensor(target, dtype=torch.float32)

            return fp, target_tensor

        elif self.mode == "index":
            return idx, smiles, target

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_all_smiles(self) -> List[str]:
        """Get all SMILES strings in the dataset."""
        return self.smiles.copy()

    def get_all_targets(self) -> np.ndarray:
        """Get all target values in the dataset."""
        return self.targets.copy()

    def set_mode(self, mode: Literal["smiles", "fingerprint", "index"]) -> None:
        """Change the output mode of the dataset."""
        self.mode = mode
        if mode == "fingerprint" and self._fingerprints is None:
            self._precompute_fingerprints()


class ZINCDataset(Dataset):
    """Dataset for ZINC SMILES strings for MLM pretraining.

    Used for masked language modeling pretraining of transformers.

    Args:
        smiles_list: List of SMILES strings.
        tokenizer: SMILESTokenizer instance for encoding SMILES.
        max_length: Maximum sequence length for tokenization.
    """

    def __init__(
        self,
        smiles_list: List[str],
        tokenizer: "SMILESTokenizer",  # Forward reference to avoid circular import
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get encoded SMILES for MLM training.

        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors.
        """
        smiles = self.smiles_list[idx]
        encoded = self.tokenizer.encode(smiles, max_length=self.max_length)
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
        }


class TransformerSolubilityDataset(Dataset):
    """Dataset for solubility regression with transformer tokenizers.

    Supports both scratch-trained SMILESTokenizer and HuggingFace tokenizers
    (e.g., ChemBERTa). This unifies SolubilityDataset and SMILESDataset.

    Args:
        smiles_list: List of SMILES strings.
        targets: Numpy array of target values.
        tokenizer: Tokenizer instance (SMILESTokenizer or HuggingFace).
        max_length: Maximum sequence length for tokenization.
        is_huggingface: Whether tokenizer is HuggingFace-based (uses different API).
    """

    def __init__(
        self,
        smiles_list: List[str],
        targets: np.ndarray,
        tokenizer: Union["SMILESTokenizer", "PreTrainedTokenizer"],  # Forward refs
        max_length: int = 512,
        is_huggingface: bool = False,
    ) -> None:
        super().__init__()
        self.smiles_list = smiles_list
        self.targets = targets.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_huggingface = is_huggingface

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized SMILES with target for solubility prediction.

        Returns:
            Dictionary with 'input_ids', 'attention_mask', and 'labels' tensors.
        """
        smiles = self.smiles_list[idx]
        target = self.targets[idx]

        if self.is_huggingface:
            # HuggingFace tokenizers return BatchEncoding
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
            # Custom SMILESTokenizer
            encoded = self.tokenizer.encode(smiles, max_length=self.max_length)
            return {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': torch.tensor(target, dtype=torch.float32),
            }
