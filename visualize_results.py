"""
Visualization script for comparing conditioning methods in diffusion models.
Creates meaningful plots showing:
- Original images
- Conditioning signals
- Generated outputs
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from model import create_model
from dataset import get_dataloader, get_conditioning_channels
from train import DiffusionTrainer

# Use a non-interactive backend for saving
import matplotlib
matplotlib.use('Agg')


def load_model(checkpoint_path, model_type='lora', rank=4, inject_layer='mid', 
               conditioning_type='edge', conditioning_channels=1):
    """Load a trained model from checkpoint"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_model(
        model_type=model_type,
        rank=rank,
        inject_layer=inject_layer,
        conditioning_channels=conditioning_channels,
        conditioning_type=conditioning_type
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    return model, device


def visualize_edge_conditioning(checkpoint_path, n_samples=8, save_path='edge_results.png'):
    """
    Visualize edge conditioning results.
    Shows: Original digit -> Edge map (condition) -> Generated from edges
    """
    print("Generating edge conditioning visualization...")
    
    # Load model
    model, device = load_model(
        checkpoint_path, 
        conditioning_type='edge',
        conditioning_channels=1
    )
    
    # Create trainer for sampling
    trainer = DiffusionTrainer(model, device=device, timesteps=1000)
    
    # Get test data
    test_loader = get_dataloader(
        batch_size=n_samples, 
        train=False, 
        conditioning='edge',
        num_workers=0
    )
    
    batch = next(iter(test_loader))
    original = batch['image'][:n_samples]
    condition = batch['condition'][:n_samples]
    labels = batch['label'][:n_samples]
    
    # Generate samples
    with torch.no_grad():
        generated = trainer.sample(condition, n_samples=n_samples, is_conditional=True)
    
    # Create figure
    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 2.5, 8))
    
    fig.suptitle('Edge Conditioning: Canny Edges → Generated Digits', fontsize=14, fontweight='bold')
    
    for i in range(n_samples):
        # Original image
        orig_img = (original[i, 0].numpy() + 1) / 2  # Denormalize to [0, 1]
        axes[0, i].imshow(orig_img, cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Digit: {labels[i].item()}', fontsize=10)
        
        # Edge condition
        edge_img = (condition[i, 0].numpy() + 1) / 2
        axes[1, i].imshow(edge_img, cmap='hot', vmin=0, vmax=1)
        axes[1, i].axis('off')
        
        # Generated
        gen_img = (generated[i, 0].cpu().numpy() + 1) / 2
        axes[2, i].imshow(gen_img, cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')
    
    # Row labels
    axes[0, 0].set_ylabel('Original', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Edge Map\n(Condition)', fontsize=12, fontweight='bold')
    axes[2, 0].set_ylabel('Generated\nfrom Edges', fontsize=12, fontweight='bold')
    
    for ax in axes[:, 0]:
        ax.yaxis.set_label_coords(-0.3, 0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_inpainting_conditioning(checkpoint_path, n_samples=8, save_path='inpainting_results.png'):
    """
    Visualize inpainting (partial mask) conditioning results.
    Shows: Original -> Masked region -> Mask -> Generated (completed)
    
    This demonstrates the ambiguous/generative nature of inpainting:
    the model must "imagine" what's behind the mask.
    """
    print("Generating inpainting conditioning visualization...")
    
    # Load model
    model, device = load_model(
        checkpoint_path,
        conditioning_type='inpainting', 
        conditioning_channels=2  # masked_image + mask
    )
    
    # Create trainer for sampling
    trainer = DiffusionTrainer(model, device=device, timesteps=1000)
    
    # Get test data
    test_loader = get_dataloader(
        batch_size=n_samples,
        train=False,
        conditioning='inpainting',
        mask_type='random_rect',
        num_workers=0
    )
    
    batch = next(iter(test_loader))
    original = batch['image'][:n_samples]
    condition = batch['condition'][:n_samples]  # (B, 2, 28, 28): [masked_img, mask]
    labels = batch['label'][:n_samples]
    
    # Split condition into masked image and mask
    masked_img = condition[:, 0:1, :, :]  # First channel
    mask = condition[:, 1:2, :, :]        # Second channel
    
    # Generate samples
    with torch.no_grad():
        generated = trainer.sample(condition, n_samples=n_samples, is_conditional=True)
    
    # Create figure
    fig, axes = plt.subplots(4, n_samples, figsize=(n_samples * 2.5, 10))
    
    fig.suptitle('Inpainting Conditioning: Partial Mask → Complete Digit\n(Model must "imagine" the masked region)', 
                 fontsize=14, fontweight='bold')
    
    for i in range(n_samples):
        # Original image
        orig_img = (original[i, 0].numpy() + 1) / 2
        axes[0, i].imshow(orig_img, cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Digit: {labels[i].item()}', fontsize=10)
        
        # Masked image (what the model sees)
        masked = (masked_img[i, 0].numpy() + 1) / 2
        axes[1, i].imshow(masked, cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        
        # Mask visualization (red = masked region)
        mask_vis = (mask[i, 0].numpy() + 1) / 2
        # Create RGB image: show original with red overlay for mask
        rgb_mask = np.stack([orig_img, orig_img * (1 - mask_vis), orig_img * (1 - mask_vis)], axis=-1)
        axes[2, i].imshow(rgb_mask)
        axes[2, i].axis('off')
        
        # Generated (completed)
        gen_img = (generated[i, 0].cpu().numpy() + 1) / 2
        axes[3, i].imshow(gen_img, cmap='gray', vmin=0, vmax=1)
        axes[3, i].axis('off')
    
    # Row labels
    axes[0, 0].set_ylabel('Original\n(Ground Truth)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Masked Input\n(Condition)', fontsize=11, fontweight='bold')
    axes[2, 0].set_ylabel('Mask Region\n(Red = Hidden)', fontsize=11, fontweight='bold')
    axes[3, 0].set_ylabel('Generated\n(Completed)', fontsize=11, fontweight='bold')
    
    for ax in axes[:, 0]:
        ax.yaxis.set_label_coords(-0.35, 0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_comparison(edge_checkpoint, inpainting_checkpoint, n_samples=6, 
                         save_path='conditioning_comparison.png'):
    """
    Side-by-side comparison of edge vs inpainting conditioning.
    Shows how different conditioning signals lead to different generation behaviors.
    """
    print("Generating comparison visualization...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load both models
    edge_model, _ = load_model(edge_checkpoint, conditioning_type='edge', conditioning_channels=1)
    inpaint_model, _ = load_model(inpainting_checkpoint, conditioning_type='inpainting', conditioning_channels=2)
    
    edge_trainer = DiffusionTrainer(edge_model, device=device, timesteps=1000)
    inpaint_trainer = DiffusionTrainer(inpaint_model, device=device, timesteps=1000)
    
    # Get same images with both conditioning types
    edge_loader = get_dataloader(batch_size=n_samples, train=False, conditioning='edge', num_workers=0)
    inpaint_loader = get_dataloader(batch_size=n_samples, train=False, conditioning='inpainting', num_workers=0)
    
    edge_batch = next(iter(edge_loader))
    inpaint_batch = next(iter(inpaint_loader))
    
    # Generate samples
    with torch.no_grad():
        edge_generated = edge_trainer.sample(edge_batch['condition'], n_samples=n_samples, is_conditional=True)
        inpaint_generated = inpaint_trainer.sample(inpaint_batch['condition'], n_samples=n_samples, is_conditional=True)
    
    # Create comprehensive comparison figure
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.1)
    
    # Top row: Edge conditioning
    ax_edge_title = fig.add_subplot(gs[0, :])
    ax_edge_title.axis('off')
    ax_edge_title.text(0.5, 0.5, 'EDGE CONDITIONING\nStructural guidance - preserves shape boundaries', 
                       ha='center', va='center', fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Create subplots for edge samples
    fig2, axes = plt.subplots(2, n_samples * 2, figsize=(n_samples * 4, 6))
    
    for i in range(n_samples):
        # Edge: condition and result
        edge_cond = (edge_batch['condition'][i, 0].numpy() + 1) / 2
        edge_gen = (edge_generated[i, 0].cpu().numpy() + 1) / 2
        
        axes[0, i*2].imshow(edge_cond, cmap='hot')
        axes[0, i*2].axis('off')
        axes[0, i*2].set_title('Edge Map', fontsize=9)
        
        axes[0, i*2+1].imshow(edge_gen, cmap='gray')
        axes[0, i*2+1].axis('off')
        axes[0, i*2+1].set_title('Generated', fontsize=9)
        
        # Inpainting: condition and result
        masked = (inpaint_batch['condition'][i, 0].numpy() + 1) / 2
        mask = (inpaint_batch['condition'][i, 1].numpy() + 1) / 2
        inpaint_gen = (inpaint_generated[i, 0].cpu().numpy() + 1) / 2
        
        # Show masked with red overlay
        rgb_masked = np.stack([masked + mask*0.5, masked, masked], axis=-1)
        rgb_masked = np.clip(rgb_masked, 0, 1)
        
        axes[1, i*2].imshow(rgb_masked)
        axes[1, i*2].axis('off')
        axes[1, i*2].set_title('Masked Input', fontsize=9)
        
        axes[1, i*2+1].imshow(inpaint_gen, cmap='gray')
        axes[1, i*2+1].axis('off')
        axes[1, i*2+1].set_title('Completed', fontsize=9)
    
    # Add row labels
    axes[0, 0].set_ylabel('EDGE\nConditioning', fontsize=12, fontweight='bold', color='blue')
    axes[1, 0].set_ylabel('INPAINTING\nConditioning', fontsize=12, fontweight='bold', color='red')
    
    fig2.suptitle('Conditioning Comparison: Edge (Structural) vs Inpainting (Generative)\n' +
                  'Edge: Model follows explicit structure | Inpainting: Model must imagine hidden content',
                  fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_all_visualizations():
    """Generate all visualization plots"""
    
    checkpoint_dir = 'checkpoints'
    
    # Find the latest checkpoints
    edge_ckpt = os.path.join(checkpoint_dir, 'lora_rank4_mid_edge_epoch0.pt')
    inpaint_ckpt = os.path.join(checkpoint_dir, 'lora_rank4_mid_inpainting_epoch0.pt')
    
    # Check if checkpoints exist
    if not os.path.exists(edge_ckpt):
        print(f"Edge checkpoint not found: {edge_ckpt}")
        print("Run: python train.py --model_type lora --conditioning edge --epochs 1")
        return
    
    if not os.path.exists(inpaint_ckpt):
        print(f"Inpainting checkpoint not found: {inpaint_ckpt}")
        print("Run: python train.py --model_type lora --conditioning inpainting --epochs 1")
        return
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Edge conditioning visualization
    visualize_edge_conditioning(
        edge_ckpt, 
        n_samples=8,
        save_path='results_edge_conditioning.png'
    )
    
    # 2. Inpainting conditioning visualization
    visualize_inpainting_conditioning(
        inpaint_ckpt,
        n_samples=8, 
        save_path='results_inpainting_conditioning.png'
    )
    
    # 3. Comparison plot
    visualize_comparison(
        edge_ckpt,
        inpaint_ckpt,
        n_samples=6,
        save_path='results_comparison.png'
    )
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - results_edge_conditioning.png")
    print("  - results_inpainting_conditioning.png") 
    print("  - results_comparison.png")


if __name__ == '__main__':
    create_all_visualizations()
