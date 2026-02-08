import torch
import torch.nn as nn
from typing import Optional

class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization (AdaLN) with conditioning.

    This layer modulates the normalized input using scale and shift parameters
    derived from a condition embedding.

    Args:
        embedding_dim: The size of each embedding vector.
        norm_eps: A value added to the denominator for numerical stability.
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Disable elementwise_affine since scale and shift are generated dynamically.
        self.norm = nn.LayerNorm(embedding_dim, eps=norm_eps, elementwise_affine=False)
        
        self.silu = nn.SiLU()
        # Output dimension is 2x embedding_dim to generate both scale and shift.
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        
        self._reset_parameters()

    def _reset_parameters(self):
        """Initializes weights to zero to ensure identity mapping at the start."""
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(
        self, 
        x: torch.Tensor, 
        condition_embed: torch.Tensor
    ) -> torch.Tensor:
        """Applies adaptive normalization.

        Args:
            x: Input tensor of shape [B, L, D].
            condition_embed: Condition embedding of shape [B, D].

        Returns:
            Modulated tensor of shape [B, L, D].
        """
        # Split the projected condition into scale and shift components.
        scale, shift = self.linear(self.silu(condition_embed)).chunk(2, dim=1)
        
        # Reshape for broadcasting: [B, D] -> [B, 1, D].
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        
        # Apply normalization and residual modulation (standard AdaLN formulation).
        x = self.norm(x)
        x = x * (1 + scale) + shift
        
        return x


class Normalize(nn.Module):
    """Reversible Instance Normalization (RevIN).

    Args:
        num_features: The number of features or channels.
        eps: A value added for numerical stability.
        affine: If True, learnable affine parameters are applied.
        subtract_last: If True, subtracts the last element of the sequence instead of the mean.
        non_norm: If True, bypasses normalization (identity).
    """

    def __init__(
        self, 
        num_features: int, 
        eps: float = 1e-5, 
        affine: bool = False, 
        subtract_last: bool = False, 
        non_norm: bool = False
    ):
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Applies normalization or denormalization.

        Args:
            x: Input tensor.
            mode: Either 'norm' or 'denorm'.

        Returns:
            Processed tensor.
        
        Raises:
            NotImplementedError: If mode is not 'norm' or 'denorm'.
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: torch.Tensor):
        # Calculate statistics over all dimensions except batch and feature.
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.non_norm:
            return x
        
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
            
        x = x / self.stdev
        
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.non_norm:
            return x
            
        if self.affine:
            x = x - self.affine_bias
            # Note: eps * eps might be intended, but standard is usually just eps.
            x = x / (self.affine_weight + self.eps * self.eps)
            
        x = x * self.stdev
        
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
