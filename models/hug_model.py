"""
HUG Model: Heterogeneous Uncertainty-Guided Composed Image Retrieval

This is the main model that implements the HUG architecture, including:
- BLIP-2 Q-Former backbone
- Learnable query tokens (K=32)
- Three uncertainty estimators (visual, text, multimodal)
- Dynamic weighting mechanism
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from .blip_backbone import BLIPBackbone
from .uncertainty_head import UncertaintyEstimator
from modules.dynamic_weighting import DynamicWeighting


class HUGModel(nn.Module):
    """
    Main HUG model for Composed Image Retrieval with probabilistic embeddings.

    Architecture:
    - Uses BLIP-2 Q-Former as the backbone
    - K=32 learnable query tokens (representing 32 Gaussian distributions)
    - Three uncertainty estimators:
        * g_V: Shared visual uncertainty estimator (for ref & target images)
        * g_T: Text uncertainty estimator
        * g_M: Multimodal coordination uncertainty estimator
    - Dynamic weighting to combine uncertainties

    Args:
        num_queries: Number of learnable query tokens (K=32 in paper)
        hidden_dim: Hidden dimension of the Q-Former
        blip_model_name: Name of the pretrained BLIP-2 model
        freeze_vision_encoder: Whether to freeze the vision encoder
        uncertainty_num_heads: Number of attention heads in uncertainty estimators
        uncertainty_dropout: Dropout rate for uncertainty estimators
    """

    def __init__(
        self,
        num_queries: int = 32,
        hidden_dim: int = 768,
        blip_model_name: str = "pretrain",  # LAVIS model type (pretrain, pretrain_vitL, coco)
        freeze_vision_encoder: bool = True,
        uncertainty_num_heads: int = 4,
        uncertainty_dropout: float = 0.1
    ):
        super().__init__()

        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # BLIP-2 Q-Former backbone
        self.blip_backbone = BLIPBackbone(
            model_type=blip_model_name,  # LAVIS model type
            device="cuda" if torch.cuda.is_available() else "cpu",
            freeze_vision_encoder=freeze_vision_encoder,
            freeze_qformer=False  # Q-Former should be trainable
        )

        # Learnable query tokens (K=32 Gaussian distributions)
        # Shape: [1, num_queries, hidden_dim]
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim) * 0.02
        )

        # Three uncertainty estimators
        # g_V: Shared visual uncertainty estimator (for both reference and target)
        self.visual_uncertainty_estimator = UncertaintyEstimator(
            hidden_dim, uncertainty_num_heads, uncertainty_dropout
        )

        # g_T: Text uncertainty estimator
        self.text_uncertainty_estimator = UncertaintyEstimator(
            hidden_dim, uncertainty_num_heads, uncertainty_dropout
        )

        # g_M: Multimodal coordination uncertainty estimator
        self.multimodal_uncertainty_estimator = UncertaintyEstimator(
            hidden_dim, uncertainty_num_heads, uncertainty_dropout
        )

        # Dynamic weighting module
        self.dynamic_weighting = DynamicWeighting()

    def extract_target_features(
        self,
        target_pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract uni-modal features from target image.

        Args:
            target_pixel_values: Target image tensor [batch_size, 3, H, W]

        Returns:
            mu_c: Target mean features [batch_size, num_queries, hidden_dim]
            sigma_c: Target uncertainty [batch_size, num_queries, hidden_dim]
        """
        batch_size = target_pixel_values.size(0)

        # Expand query tokens for batch
        query_embeds = self.query_tokens.expand(batch_size, -1, -1)

        # Extract image-only features (mean)
        mu_c = self.blip_backbone.extract_image_features(
            pixel_values=target_pixel_values,
            query_embeds=query_embeds
        )

        # Estimate uncertainty using visual uncertainty estimator
        sigma_c = self.visual_uncertainty_estimator(mu_c)

        return mu_c, sigma_c

    def extract_query_components(
        self,
        ref_pixel_values: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all components for the query side (reference image + modification text).

        This includes:
        - Visual quality uncertainty (sigma_r)
        - Text quality uncertainty (sigma_t)
        - Multimodal features (mu_m) and coordination uncertainty (sigma_m)

        Args:
            ref_pixel_values: Reference image tensor [batch_size, 3, H, W]
            text_input_ids: Text token IDs [batch_size, seq_len]
            text_attention_mask: Text attention mask [batch_size, seq_len]

        Returns:
            Dictionary containing:
                - mu_m: Multimodal mean features
                - sigma_r: Visual quality uncertainty
                - sigma_t: Text quality uncertainty
                - sigma_m: Multimodal coordination uncertainty
                - sigma_q: Combined query uncertainty (after dynamic weighting)
        """
        batch_size = ref_pixel_values.size(0)
        query_embeds = self.query_tokens.expand(batch_size, -1, -1)

        # 1. Extract multimodal features (mu_m) and coordination uncertainty (sigma_m)
        mu_m = self.blip_backbone.extract_multimodal_features(
            pixel_values=ref_pixel_values,
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            query_embeds=query_embeds
        )
        sigma_m = self.multimodal_uncertainty_estimator(mu_m)

        # 2. Extract visual-only features for visual quality uncertainty (sigma_r)
        feat_r = self.blip_backbone.extract_image_features(
            pixel_values=ref_pixel_values,
            query_embeds=query_embeds
        )
        sigma_r = self.visual_uncertainty_estimator(feat_r)

        # 3. Extract text-only features for text quality uncertainty (sigma_t)
        feat_t = self.blip_backbone.extract_text_features(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            query_embeds=query_embeds
        )
        sigma_t = self.text_uncertainty_estimator(feat_t)

        # 4. Dynamic weighting to get final query uncertainty (sigma_q)
        sigma_q = self.dynamic_weighting(sigma_r, sigma_t, sigma_m)

        return {
            'mu_m': mu_m,
            'sigma_r': sigma_r,
            'sigma_t': sigma_t,
            'sigma_m': sigma_m,
            'sigma_q': sigma_q
        }

    def forward(
        self,
        ref_pixel_values: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        target_pixel_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for training.

        Args:
            ref_pixel_values: Reference image [batch_size, 3, H, W]
            text_input_ids: Modification text token IDs [batch_size, seq_len]
            text_attention_mask: Text attention mask [batch_size, seq_len]
            target_pixel_values: Target image [batch_size, 3, H, W]

        Returns:
            Dictionary containing all features and uncertainties needed for loss computation
        """
        # Extract query side components
        query_components = self.extract_query_components(
            ref_pixel_values, text_input_ids, text_attention_mask
        )

        # Extract target side features
        mu_c, sigma_c = self.extract_target_features(target_pixel_values)

        return {
            # Query side
            'mu_q': query_components['mu_m'],  # Using multimodal mean as query mean
            'sigma_q': query_components['sigma_q'],  # Combined uncertainty
            'sigma_r': query_components['sigma_r'],  # Visual quality uncertainty
            'sigma_t': query_components['sigma_t'],  # Text quality uncertainty
            'sigma_m': query_components['sigma_m'],  # Multimodal coordination uncertainty
            # Target side
            'mu_c': mu_c,
            'sigma_c': sigma_c
        }

    def encode_query(
        self,
        ref_pixel_values: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode query (reference image + modification text) for inference.

        Args:
            ref_pixel_values: Reference image
            text_input_ids: Modification text token IDs
            text_attention_mask: Text attention mask

        Returns:
            (mu_q, sigma_q): Query mean and uncertainty
        """
        query_components = self.extract_query_components(
            ref_pixel_values, text_input_ids, text_attention_mask
        )
        return query_components['mu_m'], query_components['sigma_q']

    def encode_target(
        self,
        target_pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode target image for inference.

        Args:
            target_pixel_values: Target image

        Returns:
            (mu_c, sigma_c): Target mean and uncertainty
        """
        return self.extract_target_features(target_pixel_values)
