"""
BLIP-2 Backbone Wrapper for HUG (LAVIS-based)

This module wraps the LAVIS BLIP-2 Q-Former model for use in HUG.
The Q-Former serves as the core feature extractor for both visual and textual modalities.

Based on LAVIS implementation which properly handles:
- Image-only feature extraction (32 query tokens)
- Text-only feature extraction (via Q-Former)
- Multimodal feature extraction (cross-attention between image and text)

This wrapper provides BOTH:
1. Native LAVIS-style methods (for clean usage)
2. HUG-compatible wrapper methods (for backward compatibility)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Add LAVIS to path
lavis_path = os.path.expanduser("~/AAAI26-HUG/ref/LAVIS")
if lavis_path not in sys.path:
    sys.path.insert(0, lavis_path)

from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.common.registry import registry


class BLIPBackbone(nn.Module):
    """
    Wrapper for BLIP-2 Q-Former backbone (LAVIS-based).

    This class encapsulates the BLIP-2 model and provides a unified interface
    for extracting features from images, text, or both (multimodal).

    Uses LAVIS implementation which properly handles all three extraction modes:
    - Image mode: Vision encoder → Q-Former → 32 query tokens
    - Text mode: Q-Former (text-only) → projected features
    - Multimodal mode: Q-Former with cross-attention between image and text

    Args:
        model_type: BLIP2 model type ("pretrain", "pretrain_vitL", "coco")
        device: Device to load model on ("cuda" or "cpu")
        freeze_vision_encoder: Whether to freeze the vision encoder (default: True)
        freeze_qformer: Whether to freeze the Q-Former (default: False for fine-tuning)
    """

    def __init__(
        self,
        model_type: str = "pretrain",
        device: str = "cuda",
        freeze_vision_encoder: bool = True,
        freeze_qformer: bool = False
    ):
        super().__init__()

        self.device = device
        self.model_type = model_type

        # Load LAVIS BLIP2 model using the registry
        print(f"Loading BLIP2 model ({model_type}) from LAVIS...")

        # Build model from config
        model_cls = registry.get_model_class("blip2")
        config_path = model_cls.default_config_path(model_type=model_type)

        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config_path)

        # Override checkpoint paths to use local weights
        cfg.model.pretrained = "/mnt/jfs/hypertext-du/tanghaomiao/blip2_weights/blip2_pretrained.pth"

        # MONKEY PATCH to avoid BERT download
        # We'll load the model directly from our checkpoint
        from lavis.models.blip2_models.Qformer import BertConfig
        import transformers
        original_from_pretrained = transformers.BertLMHeadModel.from_pretrained

        def mock_from_pretrained(*args, **kwargs):
            # Return an empty model without loading weights
            if 'config' in kwargs:
                config = kwargs['config']
            elif len(args) > 1:
                config = args[1]
            else:
                config = BertConfig.from_pretrained(args[0])
            return transformers.BertLMHeadModel(config)

        # Patch during model creation
        transformers.BertLMHeadModel.from_pretrained = mock_from_pretrained

        model = model_cls.from_config(cfg.model)

        # Restore original method
        transformers.BertLMHeadModel.from_pretrained = original_from_pretrained

        # Now load our checkpoint (has full Q-Former weights)
        model.load_checkpoint_from_config(cfg.model)
        model = model.to(self.device)
        model.eval()

        self.blip_model = model

        # On CPU, convert vision encoder to float32 for compatibility
        # (fp16 Conv2d operations are not well-supported on CPU)
        if self.device == "cpu":
            print("Converting vision encoder to float32 for CPU compatibility")
            self.blip_model.visual_encoder = self.blip_model.visual_encoder.float()

        # Freeze vision encoder if specified (common practice)
        if freeze_vision_encoder:
            for param in self.blip_model.visual_encoder.parameters():
                param.requires_grad = False

        # Optionally freeze Q-Former
        if freeze_qformer:
            for param in self.blip_model.Qformer.parameters():
                param.requires_grad = False

        # Model dimensions
        self.num_query_tokens = 32  # BLIP2 uses 32 query tokens
        self.hidden_dim = self.blip_model.Qformer.config.hidden_size  # 768
        self.proj_dim = 256  # vision_proj and text_proj output dimension

        # Get tokenizer from model
        self.tokenizer = self.blip_model.tokenizer

        # Get image processor from LAVIS
        from lavis.processors import load_processor
        self.vis_processor = load_processor('blip_image_eval').build(image_size=224)

        print(f"Loaded BLIP2: {self.num_query_tokens} query tokens, "
              f"hidden_dim={self.hidden_dim}, proj_dim={self.proj_dim}")

    # =========================================================================
    # LAVIS-style methods (clean, recommended usage)
    # =========================================================================

    @torch.no_grad()
    def extract_image_features_lavis(
        self,
        pixel_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from images only (LAVIS style).

        Uses BLIP2's image mode: Vision encoder → Q-Former → 32 query tokens

        Args:
            pixel_values: Image tensor of shape [batch_size, 3, 224, 224]

        Returns:
            image_embeds: Raw Q-Former query tokens, shape [batch, 32, 768]
            image_embeds_proj: L2-normalized projected features, shape [batch, 32, 256]
        """
        # Convert input to match vision encoder dtype
        # On GPU: use fp16 (as configured in BLIP2)
        # On CPU: use float32 (converted in __init__ for compatibility)
        if self.device == "cuda" and pixel_values.dtype != torch.float16:
            pixel_values = pixel_values.to(torch.float16)
        elif self.device == "cpu" and pixel_values.dtype != torch.float32:
            pixel_values = pixel_values.to(torch.float32)

        # Extract features using LAVIS method
        feat = self.blip_model.extract_features({"image": pixel_values}, mode="image")

        # feat.image_embeds: [batch, 32, 768] - Q-Former output (unnormalized)
        # feat.image_embeds_proj: [batch, 32, 256] - Projected & L2-normalized
        image_embeds = feat.image_embeds
        image_embeds_proj = feat.image_embeds_proj

        return image_embeds, image_embeds_proj

    @torch.no_grad()
    def extract_text_features_lavis(
        self,
        texts: list
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract features from text only (LAVIS style).

        Uses BLIP2's text mode: Q-Former (text-only) → projection

        Args:
            texts: List of text strings

        Returns:
            text_embeds: Raw Q-Former output, shape [batch, seq_len, 768]
            text_embeds_proj: L2-normalized projected features, shape [batch, seq_len, 256]
            text_cls: [CLS] token feature (single vector), shape [batch, 256]
        """
        # Extract features using LAVIS method
        feat = self.blip_model.extract_features({"text_input": texts}, mode="text")

        # feat.text_embeds: [batch, seq_len, 768] - Q-Former output
        # feat.text_embeds_proj: [batch, seq_len, 256] - Projected & normalized
        text_embeds = feat.text_embeds
        text_embeds_proj = feat.text_embeds_proj

        # Extract [CLS] token (first token after projection)
        text_cls = feat.text_embeds_proj[:, 0, :]  # [batch, 256]

        return text_embeds, text_embeds_proj, text_cls

    @torch.no_grad()
    def extract_multimodal_features_lavis(
        self,
        pixel_values: torch.Tensor,
        texts: list
    ) -> torch.Tensor:
        """
        Extract multimodal features from image and text (LAVIS style).

        Uses BLIP2's multimodal mode: Q-Former with cross-attention

        Args:
            pixel_values: Image tensor [batch, 3, 224, 224]
            texts: List of text strings

        Returns:
            multimodal_embeds: Query tokens after cross-attention, shape [batch, 32, 768]
        """
        # Convert input to match vision encoder dtype
        if self.device == "cuda" and pixel_values.dtype != torch.float16:
            pixel_values = pixel_values.to(torch.float16)
        elif self.device == "cpu" and pixel_values.dtype != torch.float32:
            pixel_values = pixel_values.to(torch.float32)

        # Extract features using LAVIS multimodal mode
        feat = self.blip_model.extract_features(
            {"image": pixel_values, "text_input": texts},
            mode="multimodal"
        )

        # feat.multimodal_embeds: [batch, 32, 768]
        # Query tokens after cross-attention with both image and text
        multimodal_embeds = feat.multimodal_embeds

        return multimodal_embeds

    # =========================================================================
    # HUG-compatible wrapper methods (backward compatible)
    # =========================================================================

    @torch.no_grad()
    def extract_image_features(
        self,
        pixel_values: torch.Tensor,
        query_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract features from images only (HUG-compatible interface).

        Args:
            pixel_values: Image tensor [batch, 3, H, W]
            query_embeds: Ignored (kept for backward compatibility)

        Returns:
            image_embeds: Raw Q-Former query tokens [batch, 32, 768]
        """
        image_embeds, _ = self.extract_image_features_lavis(pixel_values)
        return image_embeds

    @torch.no_grad()
    def extract_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        query_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract features from text only (HUG-compatible interface).

        Uses BLIP2's text CLS token (following BLIP2's image-text contrastive design).
        The CLS token provides a global text representation, which is then expanded
        to 32 tokens to match HUG's query token architecture.

        Args:
            input_ids: Text token IDs [batch, seq_len]
            attention_mask: Text attention mask [batch, seq_len]
            query_embeds: Ignored (kept for backward compatibility)

        Returns:
            text_embeds: CLS token expanded to 32 tokens [batch, 32, 768]
        """
        # Convert input_ids back to text for LAVIS processor
        texts = []
        for ids in input_ids:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(text)

        # Extract features using LAVIS method
        # text_embeds: [batch, seq_len, 768] - all text tokens
        # text_embeds_proj: [batch, seq_len, 256] - projected tokens
        # text_cls: [batch, 256] - CLS token (global text representation)
        text_embeds, text_embeds_proj, text_cls = self.extract_text_features_lavis(texts)

        # Use the UNPROJECTED CLS token (hidden_dim=768, not 256)
        # Extract CLS from text_embeds (before projection)
        text_cls_unproj = text_embeds[:, 0, :]  # [batch, 768]

        # Expand CLS token to 32 tokens to match HUG's architecture
        # This broadcasts the same global text representation across all query tokens
        batch_size = text_cls_unproj.size(0)
        text_embeds = text_cls_unproj.unsqueeze(1).expand(batch_size, 32, 768)  # [batch, 32, 768]

        return text_embeds

    @torch.no_grad()
    def extract_multimodal_features(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        query_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract multimodal features from image and text (HUG-compatible interface).

        Args:
            pixel_values: Image tensor [batch, 3, H, W]
            input_ids: Text token IDs [batch, seq_len]
            attention_mask: Text attention mask [batch, seq_len]
            query_embeds: Ignored (kept for backward compatibility)

        Returns:
            multimodal_embeds: Query tokens after cross-attention [batch, 32, 768]
        """
        # Convert input_ids back to text for LAVIS processor
        texts = []
        for ids in input_ids:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(text)

        # Extract features using LAVIS multimodal mode
        return self.extract_multimodal_features_lavis(pixel_values, texts)

    def compute_similarity(
        self,
        image_feats_proj: torch.Tensor,
        text_cls: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute image-text similarity using BLIP2's MAX pooling strategy.

        Args:
            image_feats_proj: [batch, 32, 256] - Image features (projected)
            text_cls: [batch, 256] - Text [CLS] features

        Returns:
            similarities: [batch, batch] - Similarity matrix
        """
        # Compute per-token similarity: [batch, batch, 32]
        sim = torch.matmul(image_feats_proj.unsqueeze(1), text_cls.T.unsqueeze(-1))
        sim = sim.squeeze(-1)  # [batch, batch, 32]

        # BLIP2 uses MAX pooling across query tokens
        similarities, _ = sim.max(dim=-1)  # [batch, batch]

        return similarities


def create_blip_backbone(
    model_type: str = "pretrain",
    device: str = "cuda",
    freeze_vision_encoder: bool = True,
    freeze_qformer: bool = False
) -> BLIPBackbone:
    """
    Helper function to create a BLIP backbone.

    Args:
        model_type: BLIP2 model type ("pretrain", "pretrain_vitL", "coco")
        device: Device to load model on
        freeze_vision_encoder: Whether to freeze vision encoder
        freeze_qformer: Whether to freeze Q-Former

    Returns:
        BLIPBackbone instance
    """
    return BLIPBackbone(
        model_type=model_type,
        device=device,
        freeze_vision_encoder=freeze_vision_encoder,
        freeze_qformer=freeze_qformer
    )


if __name__ == "__main__":
    # Test the BLIP backbone
    import sys
    sys.path.insert(0, os.path.expanduser("~/AAAI26-HUG"))

    from models.blip_backbone import create_blip_backbone

    # Create backbone (test on CPU to avoid dtype issues)
    print("Creating BLIP backbone...")
    blip = create_blip_backbone(device="cpu")

    # Verify model attributes
    print("\n=== Verifying Model Components ===")
    print(f"✓ Vision encoder: {type(blip.blip_model.visual_encoder).__name__}")
    print(f"✓ Q-Former: {type(blip.blip_model.Qformer).__name__}")
    print(f"✓ Query tokens shape: {blip.blip_model.query_tokens.shape}")
    print(f"✓ Vision projection: {type(blip.blip_model.vision_proj).__name__}")
    print(f"✓ Text projection: {type(blip.blip_model.text_proj).__name__}")

    print(f"\n✅ BLIP backbone successfully loaded with LAVIS!")
    print(f"   - Model type: {blip.model_type}")
    print(f"   - Query tokens: {blip.num_query_tokens}")
    print(f"   - Hidden dim: {blip.hidden_dim}")
    print(f"   - Projection dim: {blip.proj_dim}")
