"""
Dataset Classes for Composed Image Retrieval

This module implements dataset loaders for CIR tasks:
- Fashion-IQ
- CIRR (Composed Image Retrieval on Real-world images)

Each dataset returns triplets: (reference_image, modification_text, target_image)
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import torch
from torch.utils.data import Dataset


class FashionIQDataset(Dataset):
    """
    Fashion-IQ dataset for Composed Image Retrieval.

    Fashion-IQ contains fashion images with natural language modification descriptions.
    Dataset structure:
    - images/
    - captions/
        - cap.{dress|shirt|toptee}.train.json
        - cap.{dress|shirt|toptee}.val.json
        - cap.{dress|shirt|toptee}.test.json

    Each JSON entry contains:
    {
        "candidate": "B00BHPC5IU.jpg",
        "target": "B00AEG3ZNG.jpg",
        "captions": ["has longer sleeves", "is more gray"]
    }
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        category: str = 'dress',  # dress, shirt, or toptee
        processor: Optional[Any] = None,  # LAVIS text processor (for preprocessing)
        tokenizer: Optional[Any] = None,  # BLIP2 tokenizer (for tokenization)
        image_transform=None
    ):
        """
        Args:
            data_root: Root directory of Fashion-IQ dataset
            split: 'train', 'val', or 'test'
            category: Fashion category ('dress', 'shirt', or 'toptee')
            processor: LAVIS text processor for text preprocessing (optional)
            tokenizer: BLIP2 tokenizer for text tokenization (required)
            image_transform: Transform function for images
        """
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.category = category
        self.processor = processor  # LAVIS text processor (preprocessing only)
        self.tokenizer = tokenizer  # BLIP2 tokenizer (actual tokenization)
        self.image_transform = image_transform

        # Load annotations
        caption_file = os.path.join(
            data_root, 'captions', f'cap.{category}.{split}.json'
        )
        with open(caption_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_dir = os.path.join(data_root, 'images')

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.

        Returns:
            Dictionary containing:
                - ref_image: Reference image tensor
                - target_image: Target image tensor
                - text_input_ids: Tokenized modification text
                - text_attention_mask: Attention mask for text
        """
        ann = self.annotations[idx]

        # Load images - Fashion-IQ uses .png extension (not included in JSON)
        ref_image_path = os.path.join(self.image_dir, f"{ann['candidate']}.png")
        target_image_path = os.path.join(self.image_dir, f"{ann['target']}.png")

        ref_image = Image.open(ref_image_path).convert('RGB')
        target_image = Image.open(target_image_path).convert('RGB')

        # Apply transforms
        if self.image_transform:
            ref_image = self.image_transform(ref_image)
            target_image = self.image_transform(target_image)

        # Combine captions (Fashion-IQ provides multiple captions per triplet)
        # Join with comma or use first caption
        modification_text = ', '.join(ann['captions'])

        # Preprocess text with LAVIS processor (optional, adds prompts/cleaning)
        if self.processor:
            modification_text = self.processor(modification_text)

        # Tokenize text with BLIP2 tokenizer
        if self.tokenizer:
            text_inputs = self.tokenizer(
                modification_text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=77  # Standard CLIP text length
            )
            text_input_ids = text_inputs['input_ids'].squeeze(0)
            text_attention_mask = text_inputs['attention_mask'].squeeze(0)
        else:
            # Fallback: return raw text (not recommended for training)
            text_input_ids = modification_text
            text_attention_mask = None

        return {
            'ref_image': ref_image,
            'target_image': target_image,
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'ref_image_id': ann['candidate'],
            'target_image_id': ann['target']
        }


class CIRRDataset(Dataset):
    """
    CIRR (Composed Image Retrieval on Real-world images) dataset.

    CIRR contains real-world images with modification texts.
    Dataset structure:
    - images/
    - captions/
        - cap.rc2.train.json
        - cap.rc2.val.json
        - cap.rc2.test1.json

    Each JSON entry contains:
    {
        "pairid": 12345,
        "reference": "dev-123-1-img0.png",
        "target_hard": "dev-456-2-img1.png",
        "caption": "remove the person on the left",
        "target_soft": ["dev-456-2-img1.png", "dev-789-3-img2.png", ...]
    }
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        processor: Optional[Any] = None,  # LAVIS text processor (for preprocessing)
        tokenizer: Optional[Any] = None,  # BLIP2 tokenizer (for tokenization)
        image_transform=None
    ):
        """
        Args:
            data_root: Root directory of CIRR dataset
            split: 'train', 'val', or 'test'
            processor: LAVIS text processor for text preprocessing (optional)
            tokenizer: BLIP2 tokenizer for text tokenization (required)
            image_transform: Transform function for images
        """
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.processor = processor  # LAVIS text processor (preprocessing only)
        self.tokenizer = tokenizer  # BLIP2 tokenizer (actual tokenization)
        self.image_transform = image_transform

        # Load annotations
        caption_file = os.path.join(
            data_root, 'captions', f'cap.rc2.{split}.json'
        )
        with open(caption_file, 'r') as f:
            self.annotations = json.load(f)

        # CIRR has images in dev/, train/, test1/ directories
        # Build image path cache for faster lookup
        print(f"Building CIRR image path cache for {split} split...")
        self._build_image_cache()

    def _build_image_cache(self):
        """Build a cache of image paths for faster lookup."""
        self.image_cache = {}

        # CIRR images are in split-specific directories with nested subdirs
        # e.g., train/16/train-9146-1-img0.png
        split_dirs = {
            'train': 'train',
            'val': 'dev',  # validation uses dev directory
            'test': 'test1'
        }

        split_dir = split_dirs.get(self.split, self.split)
        base_dir = os.path.join(self.data_root, split_dir)

        # Walk through all subdirectories
        for root, dirs, files in os.walk(base_dir):
            for fname in files:
                if fname.endswith('.png'):
                    # Remove .png extension to get the key
                    img_name = fname[:-4]
                    self.image_cache[img_name] = os.path.join(root, fname)

        print(f"  Cached {len(self.image_cache)} images")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.

        Returns:
            Dictionary containing reference image, target image, and modification text
        """
        ann = self.annotations[idx]

        # Load images using cached paths
        ref_image_name = ann['reference']
        target_image_name = ann['target_hard']

        ref_image_path = self.image_cache.get(ref_image_name)
        target_image_path = self.image_cache.get(target_image_name)

        if ref_image_path is None:
            raise FileNotFoundError(f"Reference image not found: {ref_image_name}")
        if target_image_path is None:
            raise FileNotFoundError(f"Target image not found: {target_image_name}")

        ref_image = Image.open(ref_image_path).convert('RGB')
        target_image = Image.open(target_image_path).convert('RGB')

        # Apply transforms
        if self.image_transform:
            ref_image = self.image_transform(ref_image)
            target_image = self.image_transform(target_image)

        # Get modification text
        modification_text = ann['caption']

        # Preprocess text with LAVIS processor (optional, adds prompts/cleaning)
        if self.processor:
            modification_text = self.processor(modification_text)

        # Tokenize text with BLIP2 tokenizer
        if self.tokenizer:
            text_inputs = self.tokenizer(
                modification_text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=77
            )
            text_input_ids = text_inputs['input_ids'].squeeze(0)
            text_attention_mask = text_inputs['attention_mask'].squeeze(0)
        else:
            # Fallback: return raw text (not recommended for training)
            text_input_ids = modification_text
            text_attention_mask = None

        return {
            'ref_image': ref_image,
            'target_image': target_image,
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'ref_image_id': ann['reference'],
            'target_image_id': ann['target_hard'],
            'pair_id': ann['pairid']
        }


class CIRDatasetWrapper(Dataset):
    """
    Generic wrapper for CIR datasets that handles data shuffling for L_Cord loss.

    This wrapper can create mismatched (image, text) pairs needed for
    computing the multi-modal coordination loss.
    """

    def __init__(self, base_dataset: Dataset):
        """
        Args:
            base_dataset: Base CIR dataset (FashionIQ or CIRR)
        """
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict:
        return self.base_dataset[idx]

    def get_mismatched_text(self, idx: int, seed: Optional[int] = None) -> Dict:
        """
        Get a mismatched text for computing L_Cord.

        Args:
            idx: Index of the current sample
            seed: Random seed for reproducibility

        Returns:
            Mismatched text from a different sample
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Get a random different index
        random_idx = torch.randint(0, len(self), (1,)).item()
        while random_idx == idx:
            random_idx = torch.randint(0, len(self), (1,)).item()

        # Get text from different sample
        mismatched_sample = self.base_dataset[random_idx]

        return mismatched_sample['text_input_ids'], mismatched_sample['text_attention_mask']


class FashionIQQueryDataset(Dataset):
    """
    Fashion-IQ Query dataset for retrieval evaluation.

    This dataset loads the query triplets (candidate + captions → target)
    for encoding queries during evaluation.

    Each entry returns: (reference_image, modification_text, target_id, candidate_id)
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'val',
        category: str = 'dress',
        processor: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        image_transform=None
    ):
        """
        Args:
            data_root: Root directory of Fashion-IQ dataset
            split: 'val' or 'test'
            category: Fashion category ('dress', 'shirt', or 'toptee')
            processor: LAVIS text processor for text preprocessing
            tokenizer: BLIP2 tokenizer for text tokenization
            image_transform: Transform function for images
        """
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.category = category
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_transform = image_transform

        # Load query annotations
        caption_file = os.path.join(
            data_root, 'captions', f'cap.{category}.{split}.json'
        )
        with open(caption_file, 'r') as f:
            self.queries = json.load(f)

        self.image_dir = os.path.join(data_root, 'images')

        # Build target ID to index mapping for evaluation
        self.target_to_idx = {q['target']: idx for idx, q in enumerate(self.queries)}

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single query (reference image + text)"""
        query = self.queries[idx]

        # Load reference image
        ref_image_path = os.path.join(self.image_dir, f"{query['candidate']}.png")
        ref_image = Image.open(ref_image_path).convert('RGB')

        # Apply transforms
        if self.image_transform:
            ref_image = self.image_transform(ref_image)

        # Combine captions
        modification_text = ', '.join(query['captions'])

        # Preprocess text with LAVIS processor
        if self.processor:
            modification_text = self.processor(modification_text)

        # Tokenize text
        if self.tokenizer:
            text_inputs = self.tokenizer(
                modification_text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=77
            )
            text_input_ids = text_inputs['input_ids'].squeeze(0)
            text_attention_mask = text_inputs['attention_mask'].squeeze(0)
        else:
            text_input_ids = modification_text
            text_attention_mask = None

        return {
            'ref_image': ref_image,
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'target_id': query['target'],
            'candidate_id': query['candidate']  # Also return candidate ID for exclusion
        }


class FashionIQGalleryDataset(Dataset):
    """
    Fashion-IQ Gallery dataset for retrieval evaluation.

    This dataset loads ALL images in the validation/test split
    to serve as the retrieval gallery.

    Each entry returns: (image, image_id)
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'val',
        category: str = 'dress',
        image_transform=None
    ):
        """
        Args:
            data_root: Root directory of Fashion-IQ dataset
            split: 'val' or 'test'
            category: Fashion category
            image_transform: Transform function for images
        """
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.category = category
        self.image_transform = image_transform

        # Load gallery image IDs from split file
        split_file = os.path.join(
            data_root, 'image_splits', f'split.{category}.{split}.json'
        )
        with open(split_file, 'r') as f:
            self.gallery_ids = json.load(f)

        self.image_dir = os.path.join(data_root, 'images')

        # Build ID to index mapping for evaluation
        self.id_to_idx = {img_id: idx for idx, img_id in enumerate(self.gallery_ids)}

    def __len__(self) -> int:
        return len(self.gallery_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single gallery image"""
        img_id = self.gallery_ids[idx]

        # Load image
        image_path = os.path.join(self.image_dir, f"{img_id}.png")
        image = Image.open(image_path).convert('RGB')

        # Apply transforms
        if self.image_transform:
            image = self.image_transform(image)

        return {
            'image': image,
            'image_id': img_id
        }


class CIRRQueryDataset(Dataset):
    """
    CIRR Query dataset for retrieval evaluation.

    Returns: (reference_image, modification_text, target_id, reference_id)
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'val',
        processor: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        image_transform=None
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_transform = image_transform

        # Load query annotations
        caption_file = os.path.join(
            data_root, 'captions', f'cap.rc2.{split}.json'
        )
        with open(caption_file, 'r') as f:
            self.queries = json.load(f)

        # Build image path cache
        self._build_image_cache()

        # Build target ID to index mapping
        self.target_to_idx = {q['target_hard']: idx for idx, q in enumerate(self.queries)}

    def _build_image_cache(self):
        """Build cache of image paths"""
        self.image_cache = {}
        split_dir = 'dev' if self.split == 'val' else ('test1' if self.split == 'test' else 'train')
        base_dir = os.path.join(self.data_root, split_dir)

        for root, dirs, files in os.walk(base_dir):
            for fname in files:
                if fname.endswith('.png'):
                    img_name = fname[:-4]
                    self.image_cache[img_name] = os.path.join(root, fname)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single query (reference image + text)"""
        query = self.queries[idx]

        # Load reference image
        ref_image_name = query['reference']
        ref_image_path = self.image_cache.get(ref_image_name)
        if ref_image_path is None:
            raise FileNotFoundError(f"Reference image not found: {ref_image_name}")

        ref_image = Image.open(ref_image_path).convert('RGB')

        # Apply transforms
        if self.image_transform:
            ref_image = self.image_transform(ref_image)

        # Get modification text
        modification_text = query['caption']

        # Preprocess text
        if self.processor:
            modification_text = self.processor(modification_text)

        # Tokenize text
        if self.tokenizer:
            text_inputs = self.tokenizer(
                modification_text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=77
            )
            text_input_ids = text_inputs['input_ids'].squeeze(0)
            text_attention_mask = text_inputs['attention_mask'].squeeze(0)
        else:
            text_input_ids = modification_text
            text_attention_mask = None

        return {
            'ref_image': ref_image,
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'target_id': query['target_hard'],
            'candidate_id': query['reference']  # Reference image to exclude
        }


class CIRRGalleryDataset(Dataset):
    """
    CIRR Gallery dataset for retrieval evaluation.

    Uses the split files from image_splits/ directory to load the gallery.

    Returns: (image, image_id)
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'val',
        image_transform=None
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.image_transform = image_transform

        # Load gallery from split file (dictionary: image_id -> relative_path)
        split_name = 'rc2.' + split  # e.g., 'rc2.val'
        split_file = os.path.join(data_root, 'image_splits', f'split.{split_name}.json')

        with open(split_file, 'r') as f:
            split_data = json.load(f)

        # split_data is a dict: {'dev-244-0-img0': './dev/dev-244-0-img0.png', ...}
        self.gallery_ids = list(split_data.keys())
        self.image_cache = {}

        # Build full paths
        for img_id, rel_path in split_data.items():
            # Remove leading './' if present and construct full path
            if rel_path.startswith('./'):
                rel_path = rel_path[2:]
            full_path = os.path.join(data_root, rel_path)
            self.image_cache[img_id] = full_path

        # Build ID to index mapping
        self.id_to_idx = {img_id: idx for idx, img_id in enumerate(self.gallery_ids)}

    def __len__(self) -> int:
        return len(self.gallery_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single gallery image"""
        img_id = self.gallery_ids[idx]
        image_path = self.image_cache[img_id]
        image = Image.open(image_path).convert('RGB')

        if self.image_transform:
            image = self.image_transform(image)

        return {
            'image': image,
            'image_id': img_id
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for CIR datasets.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched dictionary with stacked tensors
    """
    ref_images = torch.stack([item['ref_image'] for item in batch])
    target_images = torch.stack([item['target_image'] for item in batch])
    text_input_ids = torch.stack([item['text_input_ids'] for item in batch])
    text_attention_mask = torch.stack([item['text_attention_mask'] for item in batch])

    result = {
        'ref_images': ref_images,
        'target_images': target_images,
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'ref_image_ids': [item['ref_image_id'] for item in batch],
        'target_image_ids': [item['target_image_id'] for item in batch]
    }

    # Add optional fields if present (e.g., pair_id for CIRR)
    if 'pair_id' in batch[0]:
        result['pair_ids'] = [item['pair_id'] for item in batch]

    return result


def collate_fn_query(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for query dataset"""
    ref_images = torch.stack([item['ref_image'] for item in batch])
    text_input_ids = torch.stack([item['text_input_ids'] for item in batch])
    text_attention_mask = torch.stack([item['text_attention_mask'] for item in batch])

    return {
        'ref_images': ref_images,
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'target_ids': [item['target_id'] for item in batch],
        'candidate_ids': [item['candidate_id'] for item in batch]  # Include for exclusion
    }


def collate_fn_gallery(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for gallery dataset"""
    images = torch.stack([item['image'] for item in batch])

    return {
        'images': images,
        'image_ids': [item['image_id'] for item in batch]
    }
