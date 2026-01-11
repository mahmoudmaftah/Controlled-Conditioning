"""
Visualization script for multi-dataset diffusion model experiments
Generates samples and shows conditioning signals for MNIST, SVHN, and CLEVR
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

from dataset import get_dataloader, get_dataset_config, get_conditioning_channels, is_vector_conditioning
from model import create_model
from train import DiffusionTrainer, linear_beta_schedule


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Get dataset config
    dataset_name = config.get('dataset_name', 'mnist')
    dataset_config = get_dataset_config(dataset_name)
    
    # Create model
    model = create_model(
        model_type=config['model_type'],
        rank=config.get('rank', 4),
        inject_layer=config.get('inject_layer', 'mid'),
        conditioning_channels=get_conditioning_channels(config['conditioning_type'], dataset_name),
        conditioning_type=config['conditioning_type'],
        image_size=dataset_config['image_size'],
        in_channels=dataset_config['channels']
    )
    
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model, config


def denormalize(tensor, is_rgb=False):
    """Convert from [-1, 1] to [0, 1] for visualization"""
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    return tensor


def visualize_dataset_samples(dataset_name='clevr', conditioning='edge', num_samples=8, save_path=None):
    """Visualize samples from a dataset with their conditioning signals"""
    
    # Load data
    dataloader = get_dataloader(
        batch_size=num_samples, 
        train=True, 
        dataset_name=dataset_name,
        conditioning=conditioning
    )
    
    batch = next(iter(dataloader))
    images = batch['image']
    conditions = batch['condition']
    labels = batch.get('label', [0] * num_samples)
    
    config = get_dataset_config(dataset_name)
    is_rgb = config['channels'] == 3
    is_vector = is_vector_conditioning(conditioning)
    
    # Create figure
    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 5))
    fig.suptitle(f'{dataset_name.upper()} - {conditioning} conditioning', fontsize=14)
    
    for i in range(num_samples):
        # Original image
        img = denormalize(images[i])
        if is_rgb:
            img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = img.squeeze().numpy()
            
        axes[0, i].imshow(img_np, cmap='gray' if not is_rgb else None)
        axes[0, i].set_title(f'Image {i}')
        axes[0, i].axis('off')
        
        # Conditioning
        if is_vector:
            # For vector conditioning, show as bar chart
            cond_np = conditions[i].numpy()
            axes[1, i].bar(range(len(cond_np)), cond_np)
            axes[1, i].set_title(f'Cond (dim={len(cond_np)})')
        else:
            cond = denormalize(conditions[i])
            if cond.shape[0] > 1:  # Multi-channel (e.g., inpainting)
                cond_np = cond[0].numpy()  # Show first channel
            else:
                cond_np = cond.squeeze().numpy()
            axes[1, i].imshow(cond_np, cmap='gray')
            axes[1, i].set_title(f'Condition')
        axes[1, i].axis('off') if not is_vector else None
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def generate_samples(model, trainer, dataloader, num_samples=8, device='cuda'):
    """Generate samples using trained model"""
    
    # Get conditioning from real data
    batch = next(iter(dataloader))
    conditions = batch['condition'][:num_samples].to(device)
    real_images = batch['image'][:num_samples]
    
    # Generate samples
    model.eval()
    with torch.no_grad():
        generated = trainer.sample(condition=conditions, n_samples=num_samples, is_conditional=True)
    
    return generated.cpu(), conditions.cpu(), real_images


def visualize_generation(checkpoint_path, num_samples=8, save_path=None, device='cuda'):
    """Visualize generated samples vs real samples"""
    
    # Load model
    model, config = load_checkpoint(checkpoint_path, device)
    
    dataset_name = config.get('dataset_name', 'mnist')
    conditioning_type = config['conditioning_type']
    dataset_config = get_dataset_config(dataset_name)
    
    # Create trainer for sampling
    trainer = DiffusionTrainer(
        model, 
        device=device, 
        timesteps=config.get('timesteps', 1000),
        image_size=dataset_config['image_size'],
        in_channels=dataset_config['channels']
    )
    
    # Get dataloader
    dataloader = get_dataloader(
        batch_size=num_samples,
        train=False,
        dataset_name=dataset_name,
        conditioning=conditioning_type
    )
    
    # Generate samples
    print("Generating samples...")
    generated, conditions, real_images = generate_samples(
        model, trainer, dataloader, num_samples, device
    )
    
    is_rgb = dataset_config['channels'] == 3
    is_vector = is_vector_conditioning(conditioning_type)
    
    # Create visualization
    n_rows = 3 if not is_vector else 4
    fig, axes = plt.subplots(n_rows, num_samples, figsize=(2 * num_samples, 2.5 * n_rows))
    fig.suptitle(f'{dataset_name.upper()} | {config["model_type"]} | {conditioning_type}', fontsize=14)
    
    for i in range(num_samples):
        # Row 1: Real images
        img = denormalize(real_images[i])
        if is_rgb:
            img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = img.squeeze().numpy()
        axes[0, i].imshow(img_np, cmap='gray' if not is_rgb else None)
        axes[0, i].set_title('Real' if i == 0 else '')
        axes[0, i].axis('off')
        
        # Row 2: Conditioning
        if is_vector:
            cond_np = conditions[i].numpy()
            axes[1, i].bar(range(len(cond_np)), cond_np, color='steelblue')
            axes[1, i].set_ylim(-1.5, 1.5)
            axes[1, i].set_title('Condition' if i == 0 else '')
        else:
            cond = denormalize(conditions[i])
            if cond.shape[0] > 1:
                cond_np = cond[0].numpy()
            else:
                cond_np = cond.squeeze().numpy()
            axes[1, i].imshow(cond_np, cmap='gray')
            axes[1, i].set_title('Condition' if i == 0 else '')
            axes[1, i].axis('off')
        
        # Row 3: Generated images
        gen = denormalize(generated[i])
        if is_rgb:
            gen_np = gen.permute(1, 2, 0).numpy()
        else:
            gen_np = gen.squeeze().numpy()
        axes[2, i].imshow(gen_np, cmap='gray' if not is_rgb else None)
        axes[2, i].set_title('Generated' if i == 0 else '')
        axes[2, i].axis('off')
        
        # Row 4 (if vector): Generated edges for comparison
        if is_vector and n_rows > 3:
            axes[3, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def visualize_denoising_process(checkpoint_path, num_steps=10, save_path=None, device='cuda'):
    """Visualize the denoising process step by step"""
    
    model, config = load_checkpoint(checkpoint_path, device)
    
    dataset_name = config.get('dataset_name', 'mnist')
    conditioning_type = config['conditioning_type']
    dataset_config = get_dataset_config(dataset_name)
    timesteps = config.get('timesteps', 1000)
    
    is_rgb = dataset_config['channels'] == 3
    image_size = dataset_config['image_size']
    in_channels = dataset_config['channels']
    
    # Get a conditioning sample
    dataloader = get_dataloader(
        batch_size=1,
        train=False,
        dataset_name=dataset_name,
        conditioning=conditioning_type
    )
    batch = next(iter(dataloader))
    condition = batch['condition'].to(device)
    
    # Setup diffusion
    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_recip_alphas = torch.sqrt(1. / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)) / (1. - alphas_cumprod)
    
    # Sample with intermediate steps
    model.eval()
    x = torch.randn(1, in_channels, image_size, image_size).to(device)
    
    step_indices = np.linspace(timesteps - 1, 0, num_steps, dtype=int)
    intermediates = [x.cpu()]
    
    with torch.no_grad():
        for i in tqdm(reversed(range(timesteps)), desc='Denoising', leave=False):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            
            noise_pred = model(x, t, condition)
            
            beta_t = betas[i]
            alpha_t = alphas[i]
            alpha_cumprod_t = alphas_cumprod[i]
            
            model_mean = sqrt_recip_alphas[i] * (
                x - (beta_t / sqrt_one_minus_alphas_cumprod[i]) * noise_pred
            )
            
            if i > 0:
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(posterior_variance[i]) * noise
            else:
                x = model_mean
            
            if i in step_indices:
                intermediates.append(x.cpu())
    
    # Visualize
    fig, axes = plt.subplots(1, len(intermediates) + 1, figsize=(2 * (len(intermediates) + 1), 3))
    fig.suptitle(f'Denoising Process ({dataset_name.upper()})', fontsize=12)
    
    # Show condition first
    cond = denormalize(batch['condition'][0])
    if cond.shape[0] > 1:
        cond_np = cond[0].numpy()
    else:
        cond_np = cond.squeeze().numpy()
    axes[0].imshow(cond_np, cmap='gray')
    axes[0].set_title('Condition')
    axes[0].axis('off')
    
    # Show intermediates
    for idx, inter in enumerate(intermediates):
        img = denormalize(inter[0])
        if is_rgb:
            img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = img.squeeze().numpy()
        
        step_label = 'Noise' if idx == 0 else f't={step_indices[idx-1] if idx <= len(step_indices) else 0}'
        if idx == len(intermediates) - 1:
            step_label = 'Final'
            
        axes[idx + 1].imshow(img_np, cmap='gray' if not is_rgb else None)
        axes[idx + 1].set_title(step_label)
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/lora_rank4_mid_edge_clevr_epoch0.pt')
    parser.add_argument('--mode', type=str, default='all', choices=['samples', 'generation', 'denoising', 'all'])
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='visualizations')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load config to get dataset name
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    dataset_name = config.get('dataset_name', 'mnist')
    conditioning = config['conditioning_type']
    
    if args.mode in ['samples', 'all']:
        print(f"\n=== Dataset Samples: {dataset_name} ===")
        visualize_dataset_samples(
            dataset_name=dataset_name,
            conditioning=conditioning,
            num_samples=args.num_samples,
            save_path=os.path.join(args.output_dir, f'{dataset_name}_samples.png')
        )
    
    if args.mode in ['generation', 'all']:
        print(f"\n=== Generation Results ===")
        visualize_generation(
            args.checkpoint,
            num_samples=args.num_samples,
            save_path=os.path.join(args.output_dir, f'{dataset_name}_generation.png'),
            device=device
        )
    
    if args.mode in ['denoising', 'all']:
        print(f"\n=== Denoising Process ===")
        visualize_denoising_process(
            args.checkpoint,
            num_steps=8,
            save_path=os.path.join(args.output_dir, f'{dataset_name}_denoising.png'),
            device=device
        )
    
    print(f"\nVisualizations saved to: {args.output_dir}/")
