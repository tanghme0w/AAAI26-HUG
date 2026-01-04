"""
Uncertainty Estimator Module for HUG

This module implements the lightweight Transformer-based uncertainty estimator
that predicts uncertainty (variance) for visual, textual, and multimodal features.

According to the paper, this is a 1-layer lightweight Transformer block.
"""

import torch
import torch.nn as nn


class UncertaintyEstimator(nn.Module):
    """
    Lightweight Transformer-based uncertainty estimator.

    This module takes in feature embeddings (mean representations) and
    estimates the uncertainty (standard deviation) for each dimension.

    Args:
        hidden_dim: Dimension of the input features
        num_heads: Number of attention heads in the Transformer layer
        dropout: Dropout rate
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # Single-layer Transformer encoder
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        # Projection head to predict uncertainty
        # Using Softplus to ensure positive values
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus()  # Ensures positive uncertainty values
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to estimate uncertainty.

        Args:
            features: Input features of shape [batch_size, num_queries, hidden_dim]
                     (e.g., K=32 query tokens from Q-Former)

        Returns:
            uncertainty: Estimated standard deviation of shape [batch_size, num_queries, hidden_dim]
        """
        # Apply Transformer layer for feature refinement
        transformed_features = self.transformer_layer(features)

        # Predict uncertainty (standard deviation)
        uncertainty = self.uncertainty_head(transformed_features)

        return uncertainty


class SharedVisualUncertaintyEstimator(nn.Module):
    """
    Shared uncertainty estimator for visual modality.

    According to the paper, g_V (visual uncertainty estimator) is shared
    for both reference image and target image.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.estimator = UncertaintyEstimator(hidden_dim, num_heads, dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Estimate visual uncertainty"""
        return self.estimator(features)
