"""Shared neural network layers for molecular models.

This module provides custom layers used across different model architectures,
most importantly the MCDropout layer for MC dropout uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MCDropout(nn.Module):
    """Monte Carlo Dropout layer with independent activation flag.

    This dropout layer can be activated independently of the model's training
    mode, allowing for MC dropout inference while keeping other layers
    (like LayerNorm, BatchNorm) in eval mode.

    Args:
        p: Dropout probability. Default: 0.1.

    Attributes:
        p: Dropout probability.
        mc_active: Whether MC dropout is active (independent of training mode).

    Example:
        >>> layer = MCDropout(p=0.2)
        >>> model.eval()  # LayerNorm/BatchNorm in eval mode
        >>> layer.enable_mc()  # Only dropout is active
        >>> # Multiple forward passes for uncertainty estimation
        >>> predictions = [model(x) for _ in range(100)]
        >>> layer.disable_mc()
    """

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Dropout probability must be between 0 and 1, got {p}")
        self.p = p
        self.mc_active = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout to input tensor.

        Dropout is applied when either:
        - The module is in training mode (self.training == True), or
        - MC dropout is explicitly activated (self.mc_active == True)

        Args:
            x: Input tensor.

        Returns:
            Tensor with dropout applied (if active) or unchanged.
        """
        if self.training or self.mc_active:
            return F.dropout(x, p=self.p, training=True)
        return x

    def enable_mc(self) -> None:
        """Enable MC dropout mode."""
        self.mc_active = True

    def disable_mc(self) -> None:
        """Disable MC dropout mode."""
        self.mc_active = False

    def extra_repr(self) -> str:
        return f"p={self.p}, mc_active={self.mc_active}"


def set_mc_dropout(model: nn.Module, active: bool) -> None:
    """Recursively enable or disable MC dropout in all MCDropout layers.

    This function traverses all submodules of a model and sets the mc_active
    flag on any MCDropout layers found.

    Args:
        model: PyTorch model containing MCDropout layers.
        active: Whether to enable (True) or disable (False) MC dropout.

    Example:
        >>> model.eval()  # Put model in eval mode
        >>> set_mc_dropout(model, True)  # Enable MC dropout only
        >>> # Perform multiple forward passes for uncertainty
        >>> predictions = torch.stack([model(x) for _ in range(100)])
        >>> mean = predictions.mean(dim=0)
        >>> std = predictions.std(dim=0)
        >>> set_mc_dropout(model, False)  # Disable MC dropout
    """
    for module in model.modules():
        if isinstance(module, MCDropout):
            if active:
                module.enable_mc()
            else:
                module.disable_mc()


class MCDropoutMixin:
    """Mixin class providing MC dropout enable/disable methods.

    Inherit from this class to add enable_mc_dropout() and disable_mc_dropout()
    methods to your model.

    Example:
        >>> class MyModel(nn.Module, MCDropoutMixin):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.dropout = MCDropout(0.1)
        ...         self.linear = nn.Linear(10, 1)
    """

    def enable_mc_dropout(self) -> None:
        """Enable MC dropout in all MCDropout layers."""
        set_mc_dropout(self, True)

    def disable_mc_dropout(self) -> None:
        """Disable MC dropout in all MCDropout layers."""
        set_mc_dropout(self, False)


def mc_dropout_inference(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 100,
    return_individual: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform MC dropout inference for uncertainty estimation.

    Args:
        model: Model with MCDropout layers.
        x: Input tensor.
        n_samples: Number of forward passes.
        return_individual: Whether to return individual predictions.

    Returns:
        Tuple of (mean_prediction, std_prediction) or
        (mean_prediction, std_prediction, all_predictions) if return_individual=True.
    """
    model.eval()
    set_mc_dropout(model, True)

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x)
            predictions.append(pred)

    set_mc_dropout(model, False)

    predictions_tensor = torch.stack(predictions, dim=0)
    mean = predictions_tensor.mean(dim=0)
    std = predictions_tensor.std(dim=0)

    if return_individual:
        return mean, std, predictions_tensor
    return mean, std
