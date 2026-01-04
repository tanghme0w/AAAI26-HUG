"""
Evaluation Script for HUG Model

This script evaluates the trained HUG model on test datasets using
standard CIR metrics: Recall@K, MRR, and Median Rank.
"""

import os
import sys

# Add LAVIS to path for processor
lavis_path = os.path.expanduser("~/AAAI26-HUG/ref/LAVIS")
if lavis_path not in sys.path:
    sys.path.insert(0, lavis_path)

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from models import HUGModel
from modules import CIRMetrics, uncertainty_aware_distance
from data import (
    FashionIQDataset, FashionIQQueryDataset, FashionIQGalleryDataset,
    CIRRDataset, CIRRQueryDataset, CIRRGalleryDataset,
    collate_fn, collate_fn_query, collate_fn_gallery, get_transform
)

# Import LAVIS processor (instead of HuggingFace Blip2Processor)
from lavis.processors import load_processor
from transformers import BertTokenizer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate HUG model')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='fashion-iq',
                       choices=['fashion-iq', 'cirr'],
                       help='Dataset to use')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the dataset')
    parser.add_argument('--category', type=str, default='dress',
                       choices=['dress', 'shirt', 'toptee'],
                       help='Fashion-IQ category')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save evaluation results (JSON)')

    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model arguments from checkpoint
    model_args = checkpoint.get('args', {})

    # Create model (use LAVIS model names)
    model = HUGModel(
        num_queries=model_args.get('num_queries', 32),
        hidden_dim=model_args.get('hidden_dim', 768),
        blip_model_name=model_args.get('blip_model', 'pretrain'),
        freeze_vision_encoder=True
    ).to(device)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    return model


def create_dataloader(args, split='test'):
    """Create dataloader for evaluation"""
    # Get image transform (no augmentation for eval)
    image_transform = get_transform(image_size=224, is_train=False)

    # Get text processor (LAVIS for text preprocessing)
    processor = load_processor('blip_caption').build()

    # Get BLIP2 tokenizer (for actual tokenization)
    # BLIP2 uses BertTokenizer with special tokens
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation_side='right')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})

    # Create dataset
    if args.dataset == 'fashion-iq':
        dataset = FashionIQDataset(
            data_root=args.data_root,
            split=split,
            category=args.category,
            processor=processor,
            tokenizer=tokenizer,
            image_transform=image_transform
        )
    elif args.dataset == 'cirr':
        dataset = CIRRDataset(
            data_root=args.data_root,
            split=split,
            processor=processor,
            tokenizer=tokenizer,
            image_transform=image_transform
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return dataloader


def create_retrieval_dataloaders(args, split='val'):
    """
    Create query and gallery dataloaders for retrieval evaluation.

    For Fashion-IQ:
    - Query dataloader: Loads triplets (candidate + captions → target_id)
    - Gallery dataloader: Loads ALL images in the split as retrieval candidates

    Returns:
        query_loader, gallery_loader, gallery_id_to_idx
    """
    # Get image transform (no augmentation for eval)
    image_transform = get_transform(image_size=224, is_train=False)

    # Get text processor (LAVIS for text preprocessing)
    processor = load_processor('blip_caption').build()

    # Get BLIP2 tokenizer (for actual tokenization)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation_side='right')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})

    # Create query dataset (triplets with reference + text → target)
    if args.dataset == 'fashion-iq':
        query_dataset = FashionIQQueryDataset(
            data_root=args.data_root,
            split=split,
            category=args.category,
            processor=processor,
            tokenizer=tokenizer,
            image_transform=image_transform
        )

        # Create gallery dataset (all images in the split)
        gallery_dataset = FashionIQGalleryDataset(
            data_root=args.data_root,
            split=split,
            category=args.category,
            image_transform=image_transform
        )

        gallery_id_to_idx = gallery_dataset.id_to_idx

    elif args.dataset == 'cirr':
        query_dataset = CIRRQueryDataset(
            data_root=args.data_root,
            split=split,
            processor=processor,
            tokenizer=tokenizer,
            image_transform=image_transform
        )

        # Create gallery dataset (all images in the split)
        gallery_dataset = CIRRGalleryDataset(
            data_root=args.data_root,
            split=split,
            image_transform=image_transform
        )

        gallery_id_to_idx = gallery_dataset.id_to_idx
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Create dataloaders
    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_query,
        pin_memory=True
    )

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_gallery,
        pin_memory=True
    )

    return query_loader, gallery_loader, gallery_id_to_idx


@torch.no_grad()
def extract_query_features(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """
    Extract query features (reference image + text).

    Returns:
        query_mu, query_sigma, target_ids, candidate_ids
    """
    query_mu_list = []
    query_sigma_list = []
    target_ids_list = []
    candidate_ids_list = []

    print("Extracting query features...")
    for batch in tqdm(dataloader):
        ref_images = batch['ref_images'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)

        # Encode query (reference + text)
        mu_q, sigma_q = model.encode_query(
            ref_pixel_values=ref_images,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask
        )

        query_mu_list.append(mu_q.cpu())
        query_sigma_list.append(sigma_q.cpu())
        target_ids_list.extend(batch['target_ids'])
        candidate_ids_list.extend(batch.get('candidate_ids', batch['target_ids']))  # Fallback for compatibility

    # Concatenate all features
    query_mu = torch.cat(query_mu_list, dim=0)
    query_sigma = torch.cat(query_sigma_list, dim=0)

    return query_mu, query_sigma, target_ids_list, candidate_ids_list


@torch.no_grad()
def extract_gallery_features(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """
    Extract gallery image features.

    Returns:
        gallery_mu, gallery_sigma, gallery_ids
    """
    gallery_mu_list = []
    gallery_sigma_list = []
    gallery_ids_list = []

    print("Extracting gallery features...")
    for batch in tqdm(dataloader):
        images = batch['images'].to(device)

        # Encode target image
        mu_c, sigma_c = model.encode_target(target_pixel_values=images)

        gallery_mu_list.append(mu_c.cpu())
        gallery_sigma_list.append(sigma_c.cpu())
        gallery_ids_list.extend(batch['image_ids'])

    # Concatenate all features
    gallery_mu = torch.cat(gallery_mu_list, dim=0)
    gallery_sigma = torch.cat(gallery_sigma_list, dim=0)

    return gallery_mu, gallery_sigma, gallery_ids_list


@torch.no_grad()
def extract_all_features(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """
    Extract all query and target features from the dataset.

    Returns:
        query_features: Dictionary with 'mu' and 'sigma'
        target_features: Dictionary with 'mu' and 'sigma'
    """
    query_mu_list = []
    query_sigma_list = []
    target_mu_list = []
    target_sigma_list = []

    print("Extracting features...")
    for batch in tqdm(dataloader):
        # Move batch to device
        ref_images = batch['ref_images'].to(device)
        target_images = batch['target_images'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)

        # Extract query features
        mu_q, sigma_q = model.encode_query(
            ref_pixel_values=ref_images,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask
        )
        query_mu_list.append(mu_q.cpu())
        query_sigma_list.append(sigma_q.cpu())

        # Extract target features
        mu_c, sigma_c = model.encode_target(target_pixel_values=target_images)
        target_mu_list.append(mu_c.cpu())
        target_sigma_list.append(sigma_c.cpu())

    # Concatenate all features
    query_mu = torch.cat(query_mu_list, dim=0)
    query_sigma = torch.cat(query_sigma_list, dim=0)
    target_mu = torch.cat(target_mu_list, dim=0)
    target_sigma = torch.cat(target_sigma_list, dim=0)

    return {
        'query_mu': query_mu,
        'query_sigma': query_sigma,
        'target_mu': target_mu,
        'target_sigma': target_sigma
    }


def compute_distance_matrix(features: dict, device: torch.device):
    """
    Compute pairwise distance matrix using uncertainty-aware distance.

    Args:
        features: Dictionary with query and target features

    Returns:
        Distance matrix of shape [num_queries, num_targets]
    """
    query_mu = features['query_mu']
    query_sigma = features['query_sigma']
    target_mu = features['target_mu']
    target_sigma = features['target_sigma']

    num_queries = query_mu.size(0)
    num_targets = target_mu.size(0)

    print(f"Computing distance matrix ({num_queries} x {num_targets})...")

    # Compute distances in batches to avoid OOM
    batch_size = 100
    distance_matrix = torch.zeros(num_queries, num_targets)

    for i in tqdm(range(0, num_queries, batch_size)):
        end_i = min(i + batch_size, num_queries)
        batch_query_mu = query_mu[i:end_i].to(device)
        batch_query_sigma = query_sigma[i:end_i].to(device)

        for j in range(0, num_targets, batch_size):
            end_j = min(j + batch_size, num_targets)
            batch_target_mu = target_mu[j:end_j].to(device)
            batch_target_sigma = target_sigma[j:end_j].to(device)

            # Compute pairwise distances
            for qi in range(batch_query_mu.size(0)):
                for ti in range(batch_target_mu.size(0)):
                    dist = uncertainty_aware_distance(
                        batch_query_mu[qi:qi+1],
                        batch_query_sigma[qi:qi+1],
                        batch_target_mu[ti:ti+1],
                        batch_target_sigma[ti:ti+1],
                        aggregate=True
                    )
                    distance_matrix[i + qi, j + ti] = dist.item()

    return distance_matrix


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """
    Evaluate model on test set (direct triplet matching).

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    # Extract all features
    features = extract_all_features(model, dataloader, device)

    # Compute distance matrix
    distance_matrix = compute_distance_matrix(features, device)

    # Compute metrics
    metric_computer = CIRMetrics(
        recall_k_values=[1, 5, 10, 50],
        recall_subset_k_values=[1, 2, 3]
    )

    metrics = metric_computer.compute_all_metrics(distance_matrix)

    return metrics


def evaluate_retrieval(model: nn.Module, query_loader: DataLoader,
                       gallery_loader: DataLoader, gallery_id_to_idx: dict,
                       device: torch.device):
    """
    Evaluate model using retrieval setup (query → gallery).

    This is the proper Fashion-IQ evaluation setup where:
    - Queries: reference image + modification text
    - Gallery: ALL images in the validation/test split
    - For each query, find rank of its true target in gallery
    - Query image (candidate) is excluded from the gallery for ranking

    Args:
        model: HUG model
        query_loader: Dataloader for query triplets
        gallery_loader: Dataloader for gallery images
        gallery_id_to_idx: Mapping from image ID to gallery index
        device: torch device

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    # Extract query features (reference + text)
    query_mu, query_sigma, target_ids, candidate_ids = extract_query_features(model, query_loader, device)

    # Extract gallery features (all images)
    gallery_mu, gallery_sigma, gallery_ids = extract_gallery_features(model, gallery_loader, device)

    num_queries = query_mu.size(0)
    num_gallery = gallery_mu.size(0)

    print(f"\nComputing query-gallery distance matrix ({num_queries} x {num_gallery})...")

    # Compute distance matrix between queries and gallery
    # For each query, we compute distance to ALL gallery images
    batch_size = 100
    distance_matrix = torch.zeros(num_queries, num_gallery)

    for i in tqdm(range(0, num_queries, batch_size)):
        end_i = min(i + batch_size, num_queries)
        batch_query_mu = query_mu[i:end_i].to(device)
        batch_query_sigma = query_sigma[i:end_i].to(device)

        # Compute distances to all gallery images at once
        # Gallery features: [num_gallery, 32, 768]
        batch_gallery_mu = gallery_mu.to(device)
        batch_gallery_sigma = gallery_sigma.to(device)

        for qi in range(batch_query_mu.size(0)):
            # Compute distance from query[i] to all gallery images
            for gi in range(num_gallery):
                dist = uncertainty_aware_distance(
                    batch_query_mu[qi:qi+1],
                    batch_query_sigma[qi:qi+1],
                    batch_gallery_mu[gi:gi+1],
                    batch_gallery_sigma[gi:gi+1],
                    aggregate=True
                )
                distance_matrix[i + qi, gi] = dist.item()

    # Efficiently exclude candidate images from rankings
    # For each query, set distance to its own candidate image to infinity
    print("Excluding query images from gallery...")
    for query_idx, candidate_id in enumerate(candidate_ids):
        if candidate_id in gallery_id_to_idx:
            candidate_gallery_idx = gallery_id_to_idx[candidate_id]
            distance_matrix[query_idx, candidate_gallery_idx] = float('inf')

    # Convert to rankings (lower distance = better rank = smaller rank number)
    # For each query, sort gallery indices by distance
    rankings = torch.argsort(distance_matrix, dim=1)  # [num_queries, num_gallery]

    # Compute metrics
    recall_k_values = [1, 5, 10, 50]
    recall_scores = {k: 0 for k in recall_k_values}

    reciprocal_ranks = []
    correct_ranks = []

    for query_idx, target_id in enumerate(target_ids):
        # Get gallery index of true target
        target_gallery_idx = gallery_id_to_idx[target_id]

        # Find rank of target (0-indexed)
        rank = (rankings[query_idx] == target_gallery_idx).nonzero(as_tuple=True)[0].item()

        # Update recall@K
        for k in recall_k_values:
            if rank < k:
                recall_scores[k] += 1

        # Reciprocal rank (1-indexed)
        reciprocal_ranks.append(1.0 / (rank + 1))
        correct_ranks.append(rank + 1)  # 1-indexed for reporting

    # Compute final metrics
    num_queries = len(target_ids)
    metrics = {}
    for k in recall_k_values:
        metrics[f'recall@{k}'] = (recall_scores[k] / num_queries) * 100

    metrics['mrr'] = sum(reciprocal_ranks) / num_queries
    metrics['median_rank'] = sorted(correct_ranks)[num_queries // 2]

    return metrics


def main():
    """Main evaluation function"""
    args = parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_checkpoint(args.checkpoint, device)

    # Choose evaluation method based on dataset and split
    # For Fashion-IQ val/test and CIRR val/test: use retrieval evaluation (query → gallery)
    # For train split or other: use direct triplet matching
    use_retrieval = (args.split in ['val', 'test'])

    if use_retrieval:
        # Retrieval evaluation (proper Fashion-IQ setup)
        print("Creating query and gallery dataloaders...")
        query_loader, gallery_loader, gallery_id_to_idx = create_retrieval_dataloaders(args, split=args.split)
        print(f"Queries: {len(query_loader.dataset)}")
        print(f"Gallery size: {len(gallery_loader.dataset)}")

        # Evaluate
        print("Starting retrieval evaluation...")
        metrics = evaluate_retrieval(model, query_loader, gallery_loader, gallery_id_to_idx, device)
    else:
        # Direct triplet matching (for train split or other datasets)
        print("Creating dataloader...")
        test_loader = create_dataloader(args, split=args.split)
        print(f"Test samples: {len(test_loader.dataset)}")

        # Evaluate
        print("Starting evaluation...")
        metrics = evaluate(model, test_loader, device)

    # Print results
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    print("="*50)

    # Save results to file
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == '__main__':
    main()
