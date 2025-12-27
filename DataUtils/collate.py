"""Collate functions for DataLoaders.

This module provides collate functions for batching data in PyTorch DataLoaders.
"""

from typing import Dict, List

import torch


def transformer_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for transformer-based models.

    Stacks input_ids, attention_mask, and optionally labels from batch items.

    Args:
        batch: List of dictionaries with 'input_ids', 'attention_mask',
               and optionally 'labels'.

    Returns:
        Dictionary with batched tensors.
    """
    result = {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
    }

    if 'labels' in batch[0]:
        result['labels'] = torch.stack([item['labels'] for item in batch])

    return result
