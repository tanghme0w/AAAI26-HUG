# AAAI26-HUG
Official Implementation for AAAI26 Oral: Heterogeneous Uncertainty-Guided Composed Image Retrieval

## Overview

**HUG (Heterogeneous Uncertainty-Guided Composed Image Retrieval)** is a novel approach for Composed Image Retrieval (CIR) that uses probabilistic embeddings to handle data noise and multi-modal coordination challenges. The model represents images and queries as series of Gaussian distributions and employs three types of uncertainty estimators to capture:

1. **Visual Quality Uncertainty** (σ_r): Uncertainty from reference image
2. **Text Quality Uncertainty** (σ_t): Uncertainty from modification text
3. **Multi-Modal Coordination Uncertainty** (σ_m): Uncertainty from cross-modal alignment

These uncertainties are dynamically combined using a learned weighting mechanism to produce robust query representations.

## Architecture

```
Reference Image + Modification Text → BLIP-2 Q-Former → K=32 Query Tokens
                                                       ↓
                                            Three Uncertainty Estimators
                                                       ↓
                                            Dynamic Weighting (Eq. 11)
                                                       ↓
                                            Final Query Representation
```

### Key Components

- **BLIP-2 Q-Former Backbone**: Pretrained multimodal encoder
- **32 Learnable Query Tokens**: Each representing a Gaussian distribution
- **Lightweight Uncertainty Estimators**: 1-layer Transformer blocks
- **Dynamic Weighting Module**: Combines heterogeneous uncertainties
- **Three-Part Loss Function**: L_HC + λ_FC·L_FC + λ_Cord·L_Cord

## Installation

### Prerequisites

- Python >= 3.8
- CUDA >= 11.8 (for GPU training)

### Setup

This project uses LAVIS (Salesforce BLIP-2 implementation).

```bash
# Clone the repository
git clone https://github.com/your-username/AAAI26-HUG.git
cd AAAI26-HUG

# Install LAVIS (includes UV environment with PyTorch, Transformers, etc.)
cd ref/LAVIS
pip install -e .
cd ..

# Install additional HUG-specific dependencies
pip install -r requirements.txt
```

## Data Preparation

### Fashion-IQ Dataset

1. Download Fashion-IQ from [official website](https://github.com/XiaoxiaoGuo/fashion-iq)
2. Organize the dataset as follows:

```
fashion-iq/
├── images/
│   ├── B00001.jpg
│   ├── B00002.jpg
│   └── ...
└── captions/
    ├── cap.dress.train.json
    ├── cap.dress.val.json
    ├── cap.dress.test.json
    ├── cap.shirt.train.json
    └── ...
```

### CIRR Dataset

1. Download CIRR from [official website](https://github.com/Cuberick-Orion/CIRR)
2. Organize the dataset as follows:

```
cirr/
├── images/
│   ├── dev-0-0-img0.png
│   ├── dev-0-1-img1.png
│   └── ...
└── captions/
    ├── cap.rc2.train.json
    ├── cap.rc2.val.json
    └── cap.rc2.test1.json
```

## Training

### Fashion-IQ

```bash
python train.py \
    --dataset fashion-iq \
    --data_root /path/to/fashion-iq \
    --category dress \
    --batch_size 32 \
    --num_epochs 30 \
    --lr 3e-5 \
    --lambda_fc 0.5 \
    --lambda_cord 0.1 \
    --output_dir ./checkpoints/fashion_iq \
    --use_wandb
```

Train on all three categories:
```bash
for category in dress shirt toptee; do
    python train.py \
        --dataset fashion-iq \
        --data_root /path/to/fashion-iq \
        --category $category \
        --batch_size 32 \
        --num_epochs 30 \
        --output_dir ./checkpoints/fashion_iq_${category}
done
```

### CIRR

```bash
python train.py \
    --dataset cirr \
    --data_root /path/to/cirr \
    --batch_size 32 \
    --num_epochs 30 \
    --lr 3e-5 \
    --lambda_fc 0.5 \
    --lambda_cord 0.1 \
    --output_dir ./checkpoints/cirr \
    --use_wandb
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | fashion-iq | Dataset to use (fashion-iq or cirr) |
| `--data_root` | Required | Root directory of dataset |
| `--category` | dress | Fashion-IQ category (dress/shirt/toptee) |
| `--batch_size` | 32 | Batch size per GPU |
| `--num_epochs` | 30 | Number of training epochs |
| `--lr` | 3e-5 | Learning rate |
| `--lambda_fc` | 0.5 | Weight for fine-grained contrastive loss |
| `--lambda_cord` | 0.1 | Weight for coordination loss |
| `--num_queries` | 32 | Number of learnable query tokens (K) |
| `--use_wandb` | False | Enable Weights & Biases logging |

## Evaluation

```bash
python eval.py \
    --dataset fashion-iq \
    --data_root /path/to/fashion-iq \
    --category dress \
    --checkpoint ./checkpoints/fashion_iq/checkpoint_final.pth \
    --batch_size 64 \
    --output_file ./results/fashion_iq_dress_results.json
```

### Evaluation Metrics

The evaluation script computes:
- **Recall@K** (K=1, 5, 10, 50): Standard retrieval metric
- **Recall_subset@K** (K=1, 2, 3): CIRR-specific metric
- **Mean Reciprocal Rank (MRR)**
- **Median Rank**

## Project Structure

```
AAAI26-HUG/
├── config/
│   ├── fashion_iq.yaml       # Fashion-IQ configuration
│   └── cirr.yaml             # CIRR configuration
├── data/
│   ├── dataset.py            # Dataset loaders for CIR
│   └── transforms.py         # Image preprocessing
├── models/
│   ├── hug_model.py          # Main HUG model
│   ├── blip_backbone.py      # BLIP-2 Q-Former wrapper
│   └── uncertainty_head.py   # Uncertainty estimator
├── modules/
│   ├── losses.py             # L_HC, L_FC, L_Cord
│   ├── dynamic_weighting.py  # Dynamic weighting (Eq. 11)
│   └── metrics.py            # Evaluation metrics
├── train.py                  # Training script
├── eval.py                   # Evaluation script
└── requirements.txt          # Dependencies
```

## Citation

If you find this project helpful to your research, please cite:

```bibtex
@inproceedings{hug2025,
  title={Heterogeneous Uncertainty-Guided Composed Image Retrieval},
  author={Your Name},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
