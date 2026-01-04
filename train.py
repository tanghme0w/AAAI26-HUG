"""
Training Script for HUG Model

This script implements the training procedure for HUG following the paper specifications:
- Learning rate: 3e-5
- Optimizer: AdamW (²1=0.9, ²2=0.999)
- Batch size: 32
- Loss weights: lambda_FC=0.5, lambda_Cord=0.1
"""

import os
import sys

# Add LAVIS to path for processor
lavis_path = os.path.expanduser("~/AAAI26-HUG/ref/LAVIS")
if lavis_path not in sys.path:
    sys.path.insert(0, lavis_path)

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("wandb not installed, logging to file only")

from models import HUGModel
from modules import HUGLoss
from data import FashionIQDataset, CIRRDataset, collate_fn, get_transform

# Import LAVIS processor (instead of HuggingFace Blip2Processor)
from lavis.processors import load_processor
from transformers import BertTokenizer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train HUG model for Composed Image Retrieval')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='fashion-iq',
                       choices=['fashion-iq', 'cirr'],
                       help='Dataset to use')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the dataset')
    parser.add_argument('--category', type=str, default='dress',
                       choices=['dress', 'shirt', 'toptee'],
                       help='Fashion-IQ category (only for Fashion-IQ)')

    # Model arguments
    parser.add_argument('--num_queries', type=int, default=32,
                       help='Number of learnable query tokens (K)')
    parser.add_argument('--hidden_dim', type=int, default=768,
                       help='Hidden dimension')
    parser.add_argument('--blip_model', type=str, default='pretrain',
                       choices=['pretrain', 'pretrain_vitL', 'coco'],
                       help='BLIP-2 LAVIS model type')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                       help='Number of warmup epochs')

    # Loss arguments
    parser.add_argument('--lambda_fc', type=float, default=0.5,
                       help='Weight for fine-grained contrastive loss')
    parser.add_argument('--lambda_cord', type=float, default=0.1,
                       help='Weight for coordination loss')

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log every N steps')
    parser.add_argument('--save_interval', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


def create_dataloader(args, split='train'):
    """Create dataloader for training or validation"""
    # Get image transform
    image_transform = get_transform(image_size=224, is_train=(split == 'train'))

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
        shuffle=(split == 'train'),
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return dataloader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: HUGLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args
):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_loss_hc = 0
    total_loss_fc = 0
    total_loss_cord = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for step, batch in enumerate(pbar):
        # Move batch to device
        ref_images = batch['ref_images'].to(device)
        target_images = batch['target_images'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)

        # Forward pass
        outputs = model(
            ref_pixel_values=ref_images,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            target_pixel_values=target_images
        )

        # For L_Cord, we need mismatched pairs
        # Simple approach: shuffle text within batch
        shuffled_indices = torch.randperm(text_input_ids.size(0))
        mismatched_text_ids = text_input_ids[shuffled_indices]
        mismatched_text_mask = text_attention_mask[shuffled_indices]

        # Get sigma_m for mismatched pairs
        mismatched_components = model.extract_query_components(
            ref_pixel_values=ref_images,
            text_input_ids=mismatched_text_ids,
            text_attention_mask=mismatched_text_mask
        )
        sigma_m_mismatched = mismatched_components['sigma_m']

        # Compute loss
        loss_dict = criterion(
            mu_q=outputs['mu_q'],
            sigma_q=outputs['sigma_q'],
            mu_c=outputs['mu_c'],
            sigma_c=outputs['sigma_c'],
            sigma_m_matched=outputs['sigma_m'],
            sigma_m_mismatched=sigma_m_mismatched
        )

        loss = loss_dict['total_loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update statistics
        total_loss += loss.item()
        total_loss_hc += loss_dict['loss_hc'].item()
        total_loss_fc += loss_dict['loss_fc'].item()
        total_loss_cord += loss_dict['loss_cord'].item()

        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'loss_hc': loss_dict['loss_hc'].item(),
            'loss_fc': loss_dict['loss_fc'].item(),
            'loss_cord': loss_dict['loss_cord'].item()
        })

        # Log to wandb
        if args.use_wandb and step % args.log_interval == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/loss_hc': loss_dict['loss_hc'].item(),
                'train/loss_fc': loss_dict['loss_fc'].item(),
                'train/loss_cord': loss_dict['loss_cord'].item(),
                'train/step': epoch * len(dataloader) + step
            })

    # Return average losses
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'loss_hc': total_loss_hc / num_batches,
        'loss_fc': total_loss_fc / num_batches,
        'loss_cord': total_loss_cord / num_batches
    }


def save_checkpoint(model, optimizer, epoch, args, filename='checkpoint.pth'):
    """Save model checkpoint"""
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, filename)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)
    }, checkpoint_path)

    print(f"Checkpoint saved to {checkpoint_path}")


def main():
    """Main training function"""
    args = parse_args()
    set_seed(args.seed)

    # Initialize wandb
    if args.use_wandb:
        wandb.init(project='hug-cir', config=vars(args))

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = create_dataloader(args, split='train')
    print(f"Training samples: {len(train_loader.dataset)}")

    # Create model
    print("Creating model...")
    model = HUGModel(
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
        blip_model_name=args.blip_model,
        freeze_vision_encoder=True
    ).to(device)

    # Create loss function
    criterion = HUGLoss(
        lambda_fc=args.lambda_fc,
        lambda_cord=args.lambda_cord
    ).to(device)

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )

    # Create learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )

    # Training loop
    print("Starting training...")
    for epoch in range(1, args.num_epochs + 1):
        # Train one epoch
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args
        )

        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.num_epochs} Summary:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Loss HC: {train_metrics['loss_hc']:.4f}")
        print(f"  Loss FC: {train_metrics['loss_fc']:.4f}")
        print(f"  Loss Cord: {train_metrics['loss_cord']:.4f}")

        # Log epoch metrics
        if args.use_wandb:
            wandb.log({
                'train/epoch_loss': train_metrics['loss'],
                'train/epoch_loss_hc': train_metrics['loss_hc'],
                'train/epoch_loss_fc': train_metrics['loss_fc'],
                'train/epoch_loss_cord': train_metrics['loss_cord'],
                'train/epoch': epoch,
                'train/lr': optimizer.param_groups[0]['lr']
            })

        # Update learning rate
        scheduler.step()

        # Save checkpoint
        if epoch % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, args, f'checkpoint_epoch_{epoch}.pth')

    # Save final checkpoint
    save_checkpoint(model, optimizer, args.num_epochs, args, 'checkpoint_final.pth')

    print("Training completed!")

    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
