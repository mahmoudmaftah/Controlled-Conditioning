"""
Flexible visualization script for any trained model checkpoint.
Usage:
    python visualize.py --checkpoint checkpoints/lora_rank4_mid_inpainting_mnist_final.pt
    python visualize.py --checkpoint checkpoints/lora_rank4_mid_inpainting_mnist_final.pt --n_samples 16
    python visualize.py --list  # List all available checkpoints
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse
from model import create_model
from dataset import get_dataloader, get_conditioning_channels, is_vector_conditioning, get_dataset_config
from train import DiffusionTrainer

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')


def list_checkpoints(checkpoint_dir='checkpoints'):
    """List all available checkpoints"""
    patterns = ['*.pt', '*.pth']
    checkpoints = []
    for pattern in patterns:
        checkpoints.extend(glob.glob(os.path.join(checkpoint_dir, pattern)))
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}/")
        return []
    
    print(f"\n{'='*60}")
    print("AVAILABLE CHECKPOINTS")
    print(f"{'='*60}\n")
    
    for i, ckpt in enumerate(sorted(checkpoints)):
        name = os.path.basename(ckpt)
        size_mb = os.path.getsize(ckpt) / (1024 * 1024)
        print(f"  [{i+1}] {name} ({size_mb:.1f} MB)")
    
    print(f"\n{'='*60}")
    return checkpoints


def load_checkpoint(checkpoint_path):
    """Load checkpoint and extract config"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint.get('config', {})
    
    # If no config, try to parse from filename
    if not config:
        name = os.path.basename(checkpoint_path).replace('.pt', '').replace('.pth', '')
        parts = name.split('_')
        
        config = {
            'model_type': 'lora',
            'rank': 4,
            'inject_layer': 'mid',
            'conditioning_type': 'edge',
            'dataset_name': 'mnist',
            'image_size': 28,
            'in_channels': 1
        }
        
        for i, p in enumerate(parts):
            if p.startswith('rank'):
                config['rank'] = int(p.replace('rank', ''))
            elif p in ['early', 'mid', 'all']:
                config['inject_layer'] = p
            elif p in ['edge', 'sobel', 'skeleton', 'inpainting']:
                config['conditioning_type'] = p
            elif p in ['mnist', 'svhn', 'clevr']:
                config['dataset_name'] = p
            elif p in ['controlnet', 'concat', 'lora', 'baseline']:
                config['model_type'] = p
    
    return checkpoint, config, device


def visualize_checkpoint(checkpoint_path, n_samples=8, save_dir='visualizations', 
                         mask_type='digit_percentage', mask_percentage=50):
    """
    Generate visualization for any checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        n_samples: Number of samples to generate
        save_dir: Directory to save visualizations
        mask_type: Mask type for inpainting (if applicable)
        mask_percentage: Mask percentage for inpainting
    """
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint, config, device = load_checkpoint(checkpoint_path)
    
    # Extract config
    model_type = config.get('model_type', 'lora')
    rank = config.get('rank', 4)
    inject_layer = config.get('inject_layer', 'mid')
    conditioning_type = config.get('conditioning_type', 'edge')
    dataset_name = config.get('dataset_name', 'mnist')
    timesteps = config.get('timesteps', 1000)
    
    # Get dataset config
    try:
        dataset_config = get_dataset_config(dataset_name)
        image_size = dataset_config['image_size']
        in_channels = dataset_config['channels']
    except:
        image_size = config.get('image_size', 28)
        in_channels = config.get('in_channels', 1)
    
    print(f"Model type: {model_type}")
    print(f"Rank: {rank}, Inject layer: {inject_layer}")
    print(f"Conditioning: {conditioning_type}")
    print(f"Dataset: {dataset_name} ({image_size}x{image_size}, {in_channels}ch)")
    
    # Get conditioning channels
    cond_channels = get_conditioning_channels(conditioning_type, dataset_name)
    is_vector = is_vector_conditioning(conditioning_type)
    
    # Create model
    if is_vector:
        model_type_actual = 'lora_vector' if model_type == 'lora' else model_type
    else:
        model_type_actual = model_type
    
    model = create_model(
        model_type=model_type_actual,
        rank=rank,
        inject_layer=inject_layer,
        conditioning_channels=cond_channels,
        conditioning_type=conditioning_type,
        image_size=image_size,
        in_channels=in_channels
    )
    
    # Load weights
    try:
        model.load_state_dict(checkpoint['model'])
        print("[OK] Model weights loaded successfully")
    except Exception as e:
        print(f"[ERROR] Error loading weights: {e}")
        return
    
    model = model.to(device)
    model.eval()
    
    # Create trainer for sampling
    trainer = DiffusionTrainer(model, device=device, timesteps=timesteps,
                               image_size=image_size, in_channels=in_channels)
    
    # Get test data
    print(f"\n{'='*60}")
    print("GENERATING SAMPLES")
    print(f"{'='*60}")
    
    test_loader = get_dataloader(
        batch_size=n_samples,
        train=False,
        conditioning=conditioning_type,
        mask_type=mask_type,
        mask_percentage=mask_percentage,
        num_workers=0,
        dataset_name=dataset_name
    )
    
    batch = next(iter(test_loader))
    condition = batch['condition'][:n_samples]
    target = batch['image'][:n_samples]
    labels = batch['label'][:n_samples]
    
    # Generate samples
    is_conditional = model_type != 'baseline'
    
    print(f"Generating {n_samples} samples...")
    with torch.no_grad():
        if is_conditional:
            generated = trainer.sample(condition, n_samples=n_samples, is_conditional=True)
        else:
            generated = trainer.sample(n_samples=n_samples, is_conditional=False)
    
    print("[OK] Sampling complete")
    
    # Create visualization
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATION")
    print(f"{'='*60}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine colormap
    cmap = 'gray' if in_channels == 1 else None
    
    if conditioning_type == 'inpainting':
        # Special visualization for inpainting - add extra column for labels
        fig, axes = plt.subplots(4, n_samples + 1, figsize=(n_samples * 2 + 2, 8),
                                 gridspec_kw={'width_ratios': [0.5] + [1]*n_samples})
        fig.suptitle(f'Inpainting Results: {model_type.upper()} (rank={rank}, {inject_layer})\n'
                     f'Mask: {mask_type}, {mask_percentage}% hidden', 
                     fontsize=14, fontweight='bold')
        
        # Row labels in first column
        row_labels = ['ORIGINAL\n(Ground Truth)', 'MASKED INPUT\n(Condition)', 
                      'MASK\n(Red = Hidden)', 'GENERATED\n(Completed)']
        for row, label in enumerate(row_labels):
            axes[row, 0].text(0.5, 0.5, label, ha='center', va='center', 
                             fontsize=11, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
            axes[row, 0].axis('off')
        
        for i in range(n_samples):
            col = i + 1  # Offset by 1 for label column
            
            # Original
            if in_channels == 1:
                orig = (target[i, 0].cpu().numpy() + 1) / 2
                axes[0, col].imshow(orig, cmap='gray', vmin=0, vmax=1)
            else:
                orig = (target[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2
                axes[0, col].imshow(np.clip(orig, 0, 1))
            axes[0, col].axis('off')
            axes[0, col].set_title(f'Label: {labels[i].item()}', fontsize=9)
            
            # Masked input
            if in_channels == 1:
                masked = (condition[i, 0].cpu().numpy() + 1) / 2
                axes[1, col].imshow(masked, cmap='gray', vmin=0, vmax=1)
            else:
                masked = (condition[i, :3].cpu().numpy().transpose(1, 2, 0) + 1) / 2
                axes[1, col].imshow(np.clip(masked, 0, 1))
            axes[1, col].axis('off')
            
            # Mask visualization
            mask_ch = 1 if in_channels == 1 else 3
            mask_vis = (condition[i, mask_ch].cpu().numpy() + 1) / 2
            axes[2, col].imshow(mask_vis, cmap='Reds', vmin=0, vmax=1)
            axes[2, col].axis('off')
            
            # Generated
            if in_channels == 1:
                gen = (generated[i, 0].cpu().numpy() + 1) / 2
                axes[3, col].imshow(gen, cmap='gray', vmin=0, vmax=1)
            else:
                gen = (generated[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2
                axes[3, col].imshow(np.clip(gen, 0, 1))
            axes[3, col].axis('off')
        
    elif is_vector:
        # Vector conditioning visualization (2 rows) - add label column
        fig, axes = plt.subplots(2, n_samples + 1, figsize=(n_samples * 2 + 2, 4),
                                 gridspec_kw={'width_ratios': [0.5] + [1]*n_samples})
        fig.suptitle(f'{conditioning_type.upper()} Conditioning: {model_type.upper()} (rank={rank})', 
                     fontsize=14, fontweight='bold')
        
        # Row labels
        row_labels = ['GENERATED', 'TARGET\n(Ground Truth)']
        for row, label in enumerate(row_labels):
            axes[row, 0].text(0.5, 0.5, label, ha='center', va='center', 
                             fontsize=11, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
            axes[row, 0].axis('off')
        
        for i in range(n_samples):
            col = i + 1
            
            # Generated
            if in_channels == 1:
                gen = (generated[i, 0].cpu().numpy() + 1) / 2
                axes[0, col].imshow(gen, cmap='gray', vmin=0, vmax=1)
            else:
                gen = (generated[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2
                axes[0, col].imshow(np.clip(gen, 0, 1))
            axes[0, col].axis('off')
            
            # Target
            if in_channels == 1:
                tgt = (target[i, 0].cpu().numpy() + 1) / 2
                axes[1, col].imshow(tgt, cmap='gray', vmin=0, vmax=1)
            else:
                tgt = (target[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2
                axes[1, col].imshow(np.clip(tgt, 0, 1))
            axes[1, col].axis('off')
            axes[1, col].set_title(f'Label: {labels[i].item()}', fontsize=9)
        
    else:
        # Image conditioning visualization (3 rows) - add label column
        fig, axes = plt.subplots(3, n_samples + 1, figsize=(n_samples * 2 + 2, 6),
                                 gridspec_kw={'width_ratios': [0.5] + [1]*n_samples})
        fig.suptitle(f'{conditioning_type.upper()} Conditioning: {model_type.upper()} (rank={rank}, {inject_layer})', 
                     fontsize=14, fontweight='bold')
        
        # Row labels
        row_labels = [f'CONDITION\n({conditioning_type})', 'GENERATED', 'TARGET\n(Ground Truth)']
        colors = ['lightyellow', 'lightgreen', 'lightblue']
        for row, (label, color) in enumerate(zip(row_labels, colors)):
            axes[row, 0].text(0.5, 0.5, label, ha='center', va='center', 
                             fontsize=11, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
            axes[row, 0].axis('off')
        
        for i in range(n_samples):
            col = i + 1
            
            # Condition
            cond = (condition[i, 0].cpu().numpy() + 1) / 2
            axes[0, col].imshow(cond, cmap='hot', vmin=0, vmax=1)
            axes[0, col].axis('off')
            axes[0, col].set_title(f'Label: {labels[i].item()}', fontsize=9)
            
            # Generated
            if in_channels == 1:
                gen = (generated[i, 0].cpu().numpy() + 1) / 2
                axes[1, col].imshow(gen, cmap='gray', vmin=0, vmax=1)
            else:
                gen = (generated[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2
                axes[1, col].imshow(np.clip(gen, 0, 1))
            axes[1, col].axis('off')
            
            # Target
            if in_channels == 1:
                tgt = (target[i, 0].cpu().numpy() + 1) / 2
                axes[2, col].imshow(tgt, cmap='gray', vmin=0, vmax=1)
            else:
                tgt = (target[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2
                axes[2, col].imshow(np.clip(tgt, 0, 1))
            axes[2, col].axis('off')
    
    plt.tight_layout()
    
    # Save
    ckpt_name = os.path.basename(checkpoint_path).replace('.pt', '').replace('.pth', '')
    save_path = os.path.join(save_dir, f'viz_{ckpt_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[OK] Saved: {save_path}")
    
    # Also show training history if available
    if 'history' in checkpoint:
        history = checkpoint['history']
        if 'loss' in history:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(history['loss'], 'b-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'Training Loss: {ckpt_name}')
            ax.grid(True, alpha=0.3)
            
            loss_path = os.path.join(save_dir, f'loss_{ckpt_name}.png')
            plt.savefig(loss_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✓ Saved: {loss_path}")
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*60}")
    
    return save_path


def visualize_all(checkpoint_dir='checkpoints', save_dir='visualizations', n_samples=8):
    """Visualize all checkpoints in directory"""
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        return
    
    print(f"\nVisualizing {len(checkpoints)} checkpoint(s)...")
    
    for ckpt in checkpoints:
        try:
            visualize_checkpoint(ckpt, n_samples=n_samples, save_dir=save_dir)
        except Exception as e:
            print(f"Error visualizing {ckpt}: {e}")
            continue


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize trained model outputs')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file')
    parser.add_argument('--n_samples', type=int, default=8,
                        help='Number of samples to generate')
    parser.add_argument('--save_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--list', action='store_true',
                        help='List available checkpoints')
    parser.add_argument('--all', action='store_true',
                        help='Visualize all checkpoints')
    parser.add_argument('--mask_type', type=str, default='digit_percentage',
                        help='Mask type for inpainting visualization')
    parser.add_argument('--mask_percentage', type=int, default=50,
                        help='Mask percentage for inpainting')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.list:
        list_checkpoints()
    elif args.all:
        visualize_all(save_dir=args.save_dir, n_samples=args.n_samples)
    elif args.checkpoint:
        visualize_checkpoint(
            args.checkpoint, 
            n_samples=args.n_samples,
            save_dir=args.save_dir,
            mask_type=args.mask_type,
            mask_percentage=args.mask_percentage
        )
    else:
        # Default: try to find latest checkpoint
        checkpoints = glob.glob('checkpoints/*.pt')
        if checkpoints:
            # Sort by modification time, get most recent
            latest = max(checkpoints, key=os.path.getmtime)
            print(f"Using most recent checkpoint: {latest}")
            visualize_checkpoint(
                latest,
                n_samples=args.n_samples,
                save_dir=args.save_dir,
                mask_type=args.mask_type,
                mask_percentage=args.mask_percentage
            )
        else:
            print("No checkpoint specified and none found in checkpoints/")
            print("\nUsage:")
            print("  python visualize.py --checkpoint checkpoints/your_model.pt")
            print("  python visualize.py --list")
            print("  python visualize.py --all")
