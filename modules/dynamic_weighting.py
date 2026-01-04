"""
Dynamic Weighting Module for HUG

This module implements the dynamic weighting mechanism described in Equation (11)
of the HUG paper. It combines uncertainties from visual, textual, and multimodal
components using an uncertainty-based weighting scheme.
"""

import torch
import torch.nn as nn


class DynamicWeighting(nn.Module):
    """
    Dynamic weighting mechanism for combining heterogeneous uncertainties.

    According to Equation (11) in the paper:
    w_r = exp(-Ã_r²) / (exp(-Ã_r²) + exp(-Ã_t²) + exp(-Ã_m²))
    w_t = exp(-Ã_t²) / (exp(-Ã_r²) + exp(-Ã_t²) + exp(-Ã_m²))
    w_m = exp(-Ã_m²) / (exp(-Ã_r²) + exp(-Ã_t²) + exp(-Ã_m²))

    Then, according to Equation (10):
    Ã_q² = w_r·Ã_r² + w_t·Ã_t² + w_m·Ã_m²
    """

    def __init__(self, eps: float = 1e-6):
        """
        Args:
            eps: Small epsilon value for numerical stability
        """
        super().__init__()
        self.eps = eps

    def forward(
        self,
        sigma_r: torch.Tensor,
        sigma_t: torch.Tensor,
        sigma_m: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dynamically weighted uncertainty for the query side.

        Args:
            sigma_r: Visual quality uncertainty (reference image)
                    Shape: [batch_size, num_queries, hidden_dim]
            sigma_t: Text quality uncertainty (modification text)
                    Shape: [batch_size, num_queries, hidden_dim]
            sigma_m: Multi-modal coordination uncertainty
                    Shape: [batch_size, num_queries, hidden_dim]

        Returns:
            sigma_q: Combined query uncertainty
                    Shape: [batch_size, num_queries, hidden_dim]
        """
        # Compute squared uncertainties
        sigma_r_sq = sigma_r ** 2
        sigma_t_sq = sigma_t ** 2
        sigma_m_sq = sigma_m ** 2

        # Compute unnormalized weights using exp(-Ã²)
        # Lower uncertainty -> Higher weight
        w_r_unnorm = torch.exp(-sigma_r_sq)
        w_t_unnorm = torch.exp(-sigma_t_sq)
        w_m_unnorm = torch.exp(-sigma_m_sq)

        # Normalize weights (Equation 11)
        denominator = w_r_unnorm + w_t_unnorm + w_m_unnorm + self.eps
        w_r = w_r_unnorm / denominator
        w_t = w_t_unnorm / denominator
        w_m = w_m_unnorm / denominator

        # Combine uncertainties using dynamic weights (Equation 10)
        sigma_q_sq = w_r * sigma_r_sq + w_t * sigma_t_sq + w_m * sigma_m_sq

        # Return standard deviation (not variance)
        sigma_q = torch.sqrt(sigma_q_sq + self.eps)

        return sigma_q

    def get_weights(
        self,
        sigma_r: torch.Tensor,
        sigma_t: torch.Tensor,
        sigma_m: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the dynamic weights without computing the combined uncertainty.
        Useful for analysis and visualization.

        Returns:
            (w_r, w_t, w_m): Normalized weights for each modality
        """
        sigma_r_sq = sigma_r ** 2
        sigma_t_sq = sigma_t ** 2
        sigma_m_sq = sigma_m ** 2

        w_r_unnorm = torch.exp(-sigma_r_sq)
        w_t_unnorm = torch.exp(-sigma_t_sq)
        w_m_unnorm = torch.exp(-sigma_m_sq)

        denominator = w_r_unnorm + w_t_unnorm + w_m_unnorm + self.eps
        w_r = w_r_unnorm / denominator
        w_t = w_t_unnorm / denominator
        w_m = w_m_unnorm / denominator

        return w_r, w_t, w_m
