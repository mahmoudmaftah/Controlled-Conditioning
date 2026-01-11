import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from model import create_model, count_parameters
from dataset import (
    get_dataloader, get_conditioning_channels, is_vector_conditioning,
    get_dataset_config, list_available_conditionings, print_dataset_info
)
import os
import argparse
import json
from datetime import datetime

# ============================================================================
# Diffusion Utilities
# ============================================================================

def linear_beta_schedule(timesteps=1000, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps=1000, s=0.008):
    """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def get_index_from_list(vals, t, x_shape):
    """Extract values from vals at indices t, reshape for broadcasting"""
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class DiffusionTrainer:
    def __init__(self, model, device='cuda', timesteps=1000, schedule='linear',
                 image_size=28, in_channels=1):
        self.model = model.to(device)
        self.device = device
        self.timesteps = timesteps
        self.image_size = image_size
        self.in_channels = in_channels
        
        # Define beta schedule
        if schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps).to(device)
        else:
            self.betas = cosine_beta_schedule(timesteps).to(device)
            
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        
        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: add noise to x_start at timestep t"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
    
    def train_step(self, batch, optimizer, is_conditional=True):
        """Single training step"""
        optimizer.zero_grad()
        
        x = batch['image'].to(self.device)
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=self.device).long()
        
        # Add noise
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise)
        
        # Predict noise
        if is_conditional and hasattr(self.model, 'control_net'):
            condition = batch['condition'].to(self.device)
            noise_pred = self.model(x_noisy, t, condition)
        elif is_conditional and hasattr(self.model, 'unet'):
            # ConcatConditionUNet
            condition = batch['condition'].to(self.device)
            noise_pred = self.model(x_noisy, t, condition)
        else:
            noise_pred = self.model(x_noisy, t)
        
        # MSE loss (DDPM objective: predict noise)
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, condition=None, n_samples=8, is_conditional=True):
        """DDPM sampling (reverse diffusion)"""
        self.model.eval()
        
        # Start from noise - use model's image size and channels
        x = torch.randn(n_samples, self.in_channels, self.image_size, self.image_size).to(self.device)
        
        if condition is not None:
            condition = condition.to(self.device)
        
        for i in tqdm(reversed(range(self.timesteps)), desc='Sampling', leave=False):
            t = torch.full((n_samples,), i, device=self.device, dtype=torch.long)
            
            # Predict noise
            if is_conditional and condition is not None:
                if hasattr(self.model, 'control_net') or hasattr(self.model, 'unet'):
                    noise_pred = self.model(x, t, condition)
                else:
                    noise_pred = self.model(x, t)
            else:
                noise_pred = self.model(x, t)
            
            # Get coefficients
            beta_t = get_index_from_list(self.betas, t, x.shape)
            alpha_t = get_index_from_list(self.alphas, t, x.shape)
            alpha_cumprod_t = get_index_from_list(self.alphas_cumprod, t, x.shape)
            sqrt_recip_alpha_t = get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
            
            # Compute mean
            model_mean = sqrt_recip_alpha_t * (
                x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * noise_pred
            )
            
            # Add noise (except for t=0)
            if i > 0:
                posterior_var_t = get_index_from_list(self.posterior_variance, t, x.shape)
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(posterior_var_t) * noise
            else:
                x = model_mean
        
        self.model.train()
        return x.clamp(-1, 1)


# Need to import F for pad
import torch.nn.functional as F

# ============================================================================
# Training Loop
# ============================================================================

def train(model_type='controlnet', rank=4, inject_layer='mid', 
          conditioning_type='edge', epochs=10, batch_size=128, lr=1e-4,
          timesteps=1000, save_every=1, checkpoint_dir='checkpoints',
          freeze_base=False, dataset_name='mnist', mask_type='random_rect',
          mask_percentage=50):
    """
    Main training function.
    
    Args:
        model_type: 'baseline', 'concat', 'controlnet', 'lora', 'lora_vector'
        rank: LoRA rank for ablation (1, 2, 4, 8, 16)
        inject_layer: 'early', 'mid', 'all'
        conditioning_type: 'edge', 'sobel', 'skeleton', 'inpainting', 
                          'stroke_thickness', 'center_scale', 'color_histogram',
                          'bounding_box', 'segmentation', 'object_positions'
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        timesteps: Number of diffusion timesteps
        save_every: Save checkpoint every N epochs
        checkpoint_dir: Directory for checkpoints
        freeze_base: Whether to freeze base UNet (for LoRA training)
        dataset_name: 'mnist', 'svhn', or 'clevr'
        mask_type: For inpainting - 'random_rect', 'digit_percentage', 'digit_contiguous', etc.
        mask_percentage: For digit-based masks, what percentage to hide (0-100)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on {device}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Model: {model_type}, Rank: {rank}, Inject: {inject_layer}")
    print(f"Conditioning: {conditioning_type}")
    if conditioning_type == 'inpainting':
        print(f"Mask type: {mask_type}, Mask percentage: {mask_percentage}%")
    
    # Get dataset configuration
    dataset_config = get_dataset_config(dataset_name)
    image_size = dataset_config['image_size']
    in_channels = dataset_config['channels']
    
    # Validate conditioning type for this dataset
    available_cond = list_available_conditionings(dataset_name)
    if conditioning_type not in available_cond:
        print(f"Warning: {conditioning_type} not in recommended conditionings for {dataset_name}")
        print(f"Available: {available_cond}")
    
    # Determine conditioning channels
    cond_channels = get_conditioning_channels(conditioning_type, dataset_name)
    is_vector_cond = is_vector_conditioning(conditioning_type)
    
    print(f"Image size: {image_size}x{image_size}, Channels: {in_channels}")
    print(f"Conditioning channels: {cond_channels}, Vector: {is_vector_cond}")
    
    # Create model
    if is_vector_cond:
        model_type_actual = 'lora_vector' if model_type == 'lora' else model_type
    else:
        model_type_actual = model_type
    
    model = create_model(
        model_type=model_type_actual, 
        rank=rank, 
        inject_layer=inject_layer,
        conditioning_channels=cond_channels,
        conditioning_type=conditioning_type,
        freeze_base=freeze_base,
        image_size=image_size,
        in_channels=in_channels
    )
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Parameter efficiency: {trainable_params/total_params*100:.1f}%")
    
    # Data
    train_loader = get_dataloader(
        batch_size=batch_size, 
        train=True, 
        conditioning=conditioning_type,
        mask_type=mask_type,
        mask_percentage=mask_percentage,
        num_workers=0,  # Windows compatibility
        dataset_name=dataset_name
    )
    
    # Training
    trainer = DiffusionTrainer(model, device=device, timesteps=timesteps,
                               image_size=image_size, in_channels=in_channels)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Determine if model is conditional
    is_conditional = model_type != 'baseline'
    
    # Training loop
    os.makedirs(checkpoint_dir, exist_ok=True)
    history = {'loss': [], 'lr': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            loss = trainer.train_step(batch, optimizer, is_conditional=is_conditional)
            epoch_loss += loss
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        history['loss'].append(avg_loss)
        history['lr'].append(current_lr)
        
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        scheduler.step()
        
        # Save checkpoint - include mask info for inpainting
        if (epoch + 1) % save_every == 0:
            if conditioning_type == 'inpainting':
                ckpt_name = f'{model_type}_rank{rank}_{inject_layer}_{conditioning_type}_{mask_type}{mask_percentage}_{dataset_name}_epoch{epoch}.pt'
            else:
                ckpt_name = f'{model_type}_rank{rank}_{inject_layer}_{conditioning_type}_{dataset_name}_epoch{epoch}.pt'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'config': {
                    'model_type': model_type,
                    'rank': rank,
                    'inject_layer': inject_layer,
                    'conditioning_type': conditioning_type,
                    'mask_type': mask_type,
                    'mask_percentage': mask_percentage,
                    'timesteps': timesteps,
                    'dataset_name': dataset_name,
                    'image_size': image_size,
                    'in_channels': in_channels
                }
            }, os.path.join(checkpoint_dir, ckpt_name))
            print(f"Saved checkpoint: {ckpt_name}")
    
    # Save final model and history
    if conditioning_type == 'inpainting':
        final_name = f'{model_type}_rank{rank}_{inject_layer}_{conditioning_type}_{mask_type}{mask_percentage}_{dataset_name}_final.pt'
    else:
        final_name = f'{model_type}_rank{rank}_{inject_layer}_{conditioning_type}_{dataset_name}_final.pt'
    torch.save({
        'model': model.state_dict(),
        'history': history,
        'config': {
            'model_type': model_type,
            'rank': rank,
            'inject_layer': inject_layer,
            'conditioning_type': conditioning_type,
            'mask_type': mask_type,
            'mask_percentage': mask_percentage,
            'timesteps': timesteps,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'dataset_name': dataset_name,
            'image_size': image_size,
            'in_channels': in_channels
        }
    }, os.path.join(checkpoint_dir, final_name))
    
    return model, trainer, history


# ============================================================================
# Experiment Configurations
# ============================================================================

def run_rank_ablation(conditioning_type='edge', epochs=5, ranks=[1, 2, 4, 8, 16]):
    """Ablation study: LoRA rank comparison"""
    print("\n" + "="*60)
    print("RANK ABLATION STUDY")
    print("="*60)
    
    results = {}
    for rank in ranks:
        print(f"\n--- Rank {rank} ---")
        model, trainer, history = train(
            model_type='lora',
            rank=rank,
            inject_layer='mid',
            conditioning_type=conditioning_type,
            epochs=epochs
        )
        results[f'rank_{rank}'] = {
            'final_loss': history['loss'][-1],
            'history': history
        }
    
    return results


def run_layer_ablation(conditioning_type='edge', epochs=5, rank=4, 
                       layers=['early', 'mid', 'all']):
    """Ablation study: Injection layer comparison"""
    print("\n" + "="*60)
    print("LAYER ABLATION STUDY")
    print("="*60)
    
    results = {}
    for layer in layers:
        print(f"\n--- Layer: {layer} ---")
        model, trainer, history = train(
            model_type='lora',
            rank=rank,
            inject_layer=layer,
            conditioning_type=conditioning_type,
            epochs=epochs
        )
        results[f'layer_{layer}'] = {
            'final_loss': history['loss'][-1],
            'history': history
        }
    
    return results


def run_conditioning_ablation(epochs=5, rank=4, 
                              conditioning_types=['edge', 'sobel', 'inpainting']):
    """Ablation study: Conditioning type comparison"""
    print("\n" + "="*60)
    print("CONDITIONING TYPE ABLATION STUDY")
    print("="*60)
    
    results = {}
    for cond_type in conditioning_types:
        print(f"\n--- Conditioning: {cond_type} ---")
        model, trainer, history = train(
            model_type='lora',
            rank=rank,
            inject_layer='mid',
            conditioning_type=cond_type,
            epochs=epochs
        )
        results[f'cond_{cond_type}'] = {
            'final_loss': history['loss'][-1],
            'history': history
        }
    
    return results


def run_full_comparison(conditioning_type='edge', epochs=5, rank=4):
    """Full comparison: Baseline vs ControlNet vs LoRA"""
    print("\n" + "="*60)
    print("FULL MODEL COMPARISON")
    print("="*60)
    
    results = {}
    
    # 1. ControlNet (full)
    print("\n--- ControlNet ---")
    model, trainer, history = train(
        model_type='controlnet',
        conditioning_type=conditioning_type,
        epochs=epochs
    )
    results['controlnet'] = {'final_loss': history['loss'][-1], 'history': history}
    
    # 2. Input concatenation baseline
    print("\n--- Input Concatenation ---")
    model, trainer, history = train(
        model_type='concat',
        conditioning_type=conditioning_type,
        epochs=epochs
    )
    results['concat'] = {'final_loss': history['loss'][-1], 'history': history}
    
    # 3. LoRA (mid)
    print("\n--- LoRA (mid) ---")
    model, trainer, history = train(
        model_type='lora',
        rank=rank,
        inject_layer='mid',
        conditioning_type=conditioning_type,
        epochs=epochs
    )
    results['lora_mid'] = {'final_loss': history['loss'][-1], 'history': history}
    
    # 4. LoRA (all layers)
    print("\n--- LoRA (all layers) ---")
    model, trainer, history = train(
        model_type='lora',
        rank=rank,
        inject_layer='all',
        conditioning_type=conditioning_type,
        epochs=epochs
    )
    results['lora_all'] = {'final_loss': history['loss'][-1], 'history': history}
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Final Loss':<15}")
    print("-"*35)
    for model_name, data in results.items():
        print(f"{model_name:<20} {data['final_loss']:<15.4f}")
    
    return results


# ============================================================================
# CLI Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train diffusion models with conditioning')
    
    # Model configuration
    parser.add_argument('--model_type', type=str, default='lora',
                       choices=['baseline', 'concat', 'controlnet', 'lora'],
                       help='Model type to train')
    parser.add_argument('--rank', type=int, default=4,
                       help='LoRA rank')
    parser.add_argument('--inject_layer', type=str, default='mid',
                       choices=['early', 'mid', 'all'],
                       help='Where to inject LoRA controls')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'svhn', 'clevr'],
                       help='Dataset to use for training')
    
    # Conditioning configuration
    parser.add_argument('--conditioning', type=str, default='edge',
                       choices=['edge', 'sobel', 'skeleton', 'inpainting', 
                               'stroke_thickness', 'center_scale', 'color_histogram',
                               'bounding_box', 'segmentation', 'object_positions'],
                       help='Type of conditioning')
    
    # Mask configuration (for inpainting)
    parser.add_argument('--mask_type', type=str, default='digit_percentage',
                       choices=['top_half', 'bottom_half', 'left_half', 'right_half',
                               'random_rect', 'center', 'digit_percentage', 
                               'digit_contiguous', 'digit_top', 'digit_bottom',
                               'digit_square', 'digit_multi_square',
                               'digit_horizontal_band', 'digit_vertical_band'],
                       help='Type of mask for inpainting conditioning')
    parser.add_argument('--mask_percentage', type=int, default=50,
                       help='Percentage of digit area to mask (0-100)')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--timesteps', type=int, default=1000,
                       help='Diffusion timesteps')
    
    # Utility
    parser.add_argument('--list_datasets', action='store_true',
                       help='List available datasets and conditioning types')
    
    # Experiment mode
    parser.add_argument('--experiment', type=str, default=None,
                       choices=['rank_ablation', 'layer_ablation', 
                               'conditioning_ablation', 'full_comparison',
                               'mask_percentage_ablation'],
                       help='Run predefined experiment')
    
    return parser.parse_args()


def run_mask_percentage_ablation(epochs=5, rank=4, percentages=[20, 40, 60, 80],
                                  mask_type='digit_percentage'):
    """Ablation study: Mask percentage comparison for inpainting"""
    print("\n" + "="*60)
    print("MASK PERCENTAGE ABLATION STUDY")
    print("="*60)
    
    results = {}
    for pct in percentages:
        print(f"\n--- Mask {pct}% of digit ---")
        model, trainer, history = train(
            model_type='lora',
            rank=rank,
            inject_layer='mid',
            conditioning_type='inpainting',
            mask_type=mask_type,
            mask_percentage=pct,
            epochs=epochs
        )
        results[f'mask_{pct}pct'] = {
            'final_loss': history['loss'][-1],
            'history': history
        }
    
    # Print summary
    print("\n" + "="*60)
    print("MASK PERCENTAGE ABLATION SUMMARY")
    print("="*60)
    print(f"{'Mask %':<20} {'Final Loss':<15}")
    print("-"*35)
    for name, data in results.items():
        print(f"{name:<20} {data['final_loss']:<15.4f}")
    
    return results


if __name__ == '__main__':
    args = parse_args()
    
    # List datasets option
    if args.list_datasets:
        print_dataset_info()
        exit(0)
    
    if args.experiment == 'rank_ablation':
        run_rank_ablation(conditioning_type=args.conditioning, epochs=args.epochs)
    elif args.experiment == 'layer_ablation':
        run_layer_ablation(conditioning_type=args.conditioning, epochs=args.epochs, rank=args.rank)
    elif args.experiment == 'conditioning_ablation':
        run_conditioning_ablation(epochs=args.epochs, rank=args.rank)
    elif args.experiment == 'full_comparison':
        run_full_comparison(conditioning_type=args.conditioning, epochs=args.epochs, rank=args.rank)
    elif args.experiment == 'mask_percentage_ablation':
        run_mask_percentage_ablation(epochs=args.epochs, rank=args.rank, 
                                      mask_type=args.mask_type)
    else:
        # Single training run
        train(
            model_type=args.model_type,
            rank=args.rank,
            inject_layer=args.inject_layer,
            conditioning_type=args.conditioning,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            timesteps=args.timesteps,
            dataset_name=args.dataset,
            mask_type=args.mask_type,
            mask_percentage=args.mask_percentage
        )