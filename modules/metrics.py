"""
Evaluation Metrics for Composed Image Retrieval

This module implements evaluation metrics commonly used for CIR tasks:
- Recall@K
- Recall_subset@K (for datasets like CIRR)
"""

import torch
import numpy as np
from typing import List, Dict


def compute_recall_at_k(
    distances: torch.Tensor,
    k_values: List[int] = [1, 5, 10, 50]
) -> Dict[str, float]:
    """
    Compute Recall@K for image retrieval.

    Args:
        distances: Distance matrix of shape [num_queries, num_targets]
                  Lower distance = more similar
        k_values: List of K values to compute Recall@K

    Returns:
        Dictionary mapping "recall@K" to the computed value
    """
    num_queries = distances.shape[0]

    # Get sorted indices (ascending order - lower distance first)
    sorted_indices = torch.argsort(distances, dim=1)

    # Ground truth is that query i matches target i (diagonal)
    results = {}

    for k in k_values:
        # Check if ground truth (index i) is in top-k predictions
        top_k_indices = sorted_indices[:, :k]
        ground_truth = torch.arange(num_queries, device=distances.device).unsqueeze(1)

        # Check if ground truth is in top-k
        correct = (top_k_indices == ground_truth).any(dim=1).float()
        recall = correct.mean().item()

        results[f'recall@{k}'] = recall * 100  # Convert to percentage

    return results


def compute_recall_subset_at_k(
    distances: torch.Tensor,
    subset_indices: List[List[int]],
    k_values: List[int] = [1, 2, 3]
) -> Dict[str, float]:
    """
    Compute Recall_subset@K metric used in CIRR dataset.

    In CIRR, each query has a small subset of relevant candidates.
    This metric checks if the ground truth is in the top-K within that subset.

    Args:
        distances: Distance matrix of shape [num_queries, num_targets]
        subset_indices: List of lists, where subset_indices[i] contains
                       the indices of the candidate subset for query i
        k_values: List of K values to compute Recall_subset@K

    Returns:
        Dictionary mapping "recall_subset@K" to the computed value
    """
    num_queries = distances.shape[0]
    results = {}

    for k in k_values:
        correct_count = 0

        for query_idx in range(num_queries):
            # Get distances only within the subset
            subset_idx = subset_indices[query_idx]
            subset_distances = distances[query_idx, subset_idx]

            # Sort subset by distance
            sorted_subset_indices = torch.argsort(subset_distances)

            # Check if ground truth (query_idx) is in top-k of the subset
            # Assuming ground truth is at position query_idx in the subset
            if query_idx in [subset_idx[i] for i in sorted_subset_indices[:k].tolist()]:
                correct_count += 1

        recall = correct_count / num_queries
        results[f'recall_subset@{k}'] = recall * 100

    return results


def compute_mean_reciprocal_rank(distances: torch.Tensor) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Args:
        distances: Distance matrix of shape [num_queries, num_targets]

    Returns:
        MRR value
    """
    num_queries = distances.shape[0]

    # Get sorted indices
    sorted_indices = torch.argsort(distances, dim=1)

    # Find rank of ground truth (diagonal element)
    ground_truth = torch.arange(num_queries, device=distances.device)

    ranks = []
    for i in range(num_queries):
        rank = (sorted_indices[i] == ground_truth[i]).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(1.0 / rank)

    return np.mean(ranks)


def compute_median_rank(distances: torch.Tensor) -> float:
    """
    Compute Median Rank of ground truth.

    Args:
        distances: Distance matrix of shape [num_queries, num_targets]

    Returns:
        Median rank value
    """
    num_queries = distances.shape[0]

    # Get sorted indices
    sorted_indices = torch.argsort(distances, dim=1)

    # Find rank of ground truth
    ground_truth = torch.arange(num_queries, device=distances.device)

    ranks = []
    for i in range(num_queries):
        rank = (sorted_indices[i] == ground_truth[i]).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    return np.median(ranks)


class CIRMetrics:
    """
    Wrapper class for computing all CIR metrics at once.
    """

    def __init__(
        self,
        recall_k_values: List[int] = [1, 5, 10, 50],
        recall_subset_k_values: List[int] = [1, 2, 3]
    ):
        """
        Args:
            recall_k_values: K values for Recall@K
            recall_subset_k_values: K values for Recall_subset@K
        """
        self.recall_k_values = recall_k_values
        self.recall_subset_k_values = recall_subset_k_values

    def compute_all_metrics(
        self,
        distances: torch.Tensor,
        subset_indices: List[List[int]] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            distances: Distance matrix [num_queries, num_targets]
            subset_indices: Optional subset indices for CIRR-style evaluation

        Returns:
            Dictionary with all computed metrics
        """
        metrics = {}

        # Recall@K
        recall_metrics = compute_recall_at_k(distances, self.recall_k_values)
        metrics.update(recall_metrics)

        # Recall_subset@K (if subset indices provided)
        if subset_indices is not None:
            recall_subset_metrics = compute_recall_subset_at_k(
                distances, subset_indices, self.recall_subset_k_values
            )
            metrics.update(recall_subset_metrics)

        # MRR
        mrr = compute_mean_reciprocal_rank(distances)
        metrics['mrr'] = mrr

        # Median Rank
        median_rank = compute_median_rank(distances)
        metrics['median_rank'] = median_rank

        return metrics
