"""
Visualize all masking strategies for report.
Generates a single figure showing all mask types applied to sample MNIST digits.

Usage:
    python visualize_masks.py
    python visualize_masks.py --n_examples 5
    python visualize_masks.py --mask_percentage 60
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataset import MNISTConditioningDataset, ConditioningGenerator
import argparse
import os

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')


def visualize_all_masks(n_examples=4, mask_percentage=50, save_path='visualizations/mask_strategies.png'):
    """
    Create a comprehensive visualization of all masking strategies.
    
    Args:
        n_examples: Number of example digits to show
        mask_percentage: Percentage of digit to mask
        save_path: Where to save the figure
    """
    
    # All available mask types with descriptions
    mask_types = [
        ('digit_square', 'Single Square\nPatch', 
         'A single square patch placed\nrandomly on the digit'),
        ('digit_multi_square', 'Multiple Square\nPatches', 
         'Multiple (2-4) smaller squares\nscattered on the digit'),
        ('digit_horizontal_band', 'Horizontal\nBand', 
         'A horizontal stripe across\nthe digit'),
        ('digit_vertical_band', 'Vertical\nBand', 
         'A vertical stripe across\nthe digit'),
        ('digit_percentage', 'Random Digit\nPixels', 
         'Random pixels from the digit\nare masked (scattered)'),
        ('digit_contiguous', 'Contiguous\nRegion', 
         'A contiguous blob expanding\nfrom a seed point'),
        ('digit_top', 'Top Portion', 
         'Top X% of digit pixels\n(by y-coordinate)'),
        ('digit_bottom', 'Bottom Portion', 
         'Bottom X% of digit pixels\n(by y-coordinate)'),
        ('random_rect', 'Random\nRectangle', 
         'Random rectangle anywhere\n(may miss digit)'),
        ('top_half', 'Top Half\n(Image)', 
         'Top half of entire image\n(not digit-aware)'),
        ('center', 'Center\nRegion', 
         'Center 50% of image\n(not digit-aware)'),
    ]
    
    # Load MNIST dataset (no conditioning, we'll generate masks manually)
    import torchvision
    mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    
    # Get diverse digit examples (one of each digit if possible)
    digit_indices = {}
    for idx in range(len(mnist)):
        _, label = mnist[idx]
        if label not in digit_indices:
            digit_indices[label] = idx
        if len(digit_indices) >= n_examples:
            break
    
    example_indices = list(digit_indices.values())[:n_examples]
    
    # Create figure
    n_masks = len(mask_types)
    fig, axes = plt.subplots(n_masks + 1, n_examples + 2, 
                             figsize=(n_examples * 2 + 4, n_masks * 1.5 + 2),
                             gridspec_kw={'width_ratios': [1.5, 1.5] + [1]*n_examples,
                                         'height_ratios': [1.2] + [1]*n_masks})
    
    fig.suptitle(f'Masking Strategies for Digit Inpainting\n(mask_percentage = {mask_percentage}%)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Header row - show original digits
    axes[0, 0].text(0.5, 0.5, 'MASK\nTYPE', ha='center', va='center', 
                    fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    axes[0, 1].text(0.5, 0.5, 'DESCRIPTION', ha='center', va='center', 
                    fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')
    
    for col, idx in enumerate(example_indices):
        img, label = mnist[idx]
        img_np = np.array(img)
        axes[0, col + 2].imshow(img_np, cmap='gray')
        axes[0, col + 2].set_title(f'Digit {label}', fontsize=10, fontweight='bold')
        axes[0, col + 2].axis('off')
    
    # Generate masks for each type
    cond_gen = ConditioningGenerator()
    
    for row, (mask_type, short_name, description) in enumerate(mask_types, start=1):
        # Mask type label
        color = 'lightblue' if 'digit' in mask_type else 'lightyellow'
        axes[row, 0].text(0.5, 0.5, short_name, ha='center', va='center', 
                         fontsize=10, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
        axes[row, 0].axis('off')
        
        # Description
        axes[row, 1].text(0.5, 0.5, description, ha='center', va='center', 
                         fontsize=8, style='italic')
        axes[row, 1].axis('off')
        
        # Apply mask to each example
        for col, idx in enumerate(example_indices):
            img, label = mnist[idx]
            img_np = np.array(img)
            
            # Generate mask
            try:
                masked_img, mask = cond_gen.get_partial_mask(img_np, mask_type, mask_percentage)
            except Exception as e:
                # Fallback for any issues
                masked_img = img_np.copy()
                mask = np.zeros_like(img_np)
            
            # Create RGB visualization: original in gray, masked areas in red
            rgb_img = np.stack([img_np, img_np, img_np], axis=-1).astype(float) / 255.0
            
            # Overlay mask in red (semi-transparent)
            mask_normalized = mask.astype(float) / 255.0
            rgb_img[:, :, 0] = np.clip(rgb_img[:, :, 0] + mask_normalized * 0.7, 0, 1)  # Red channel
            rgb_img[:, :, 1] = rgb_img[:, :, 1] * (1 - mask_normalized * 0.5)  # Reduce green
            rgb_img[:, :, 2] = rgb_img[:, :, 2] * (1 - mask_normalized * 0.5)  # Reduce blue
            
            axes[row, col + 2].imshow(rgb_img)
            axes[row, col + 2].axis('off')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='red', alpha=0.7, label='Masked Region (hidden from model)'),
        mpatches.Patch(facecolor='lightblue', alpha=0.8, label='Digit-aware masks'),
        mpatches.Patch(facecolor='lightyellow', alpha=0.8, label='Image-based masks'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
               fontsize=10, bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {save_path}")
    return save_path


def visualize_mask_percentage_comparison(mask_type='digit_square', 
                                          percentages=[20, 40, 60, 80],
                                          n_examples=4,
                                          save_path='visualizations/mask_percentage_comparison.png'):
    """
    Show how different mask percentages affect the same digits.
    """
    import torchvision
    mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    
    # Get example digits
    example_indices = []
    seen_labels = set()
    for idx in range(len(mnist)):
        _, label = mnist[idx]
        if label not in seen_labels:
            example_indices.append(idx)
            seen_labels.add(label)
        if len(example_indices) >= n_examples:
            break
    
    # Create figure
    n_pct = len(percentages)
    fig, axes = plt.subplots(n_pct + 1, n_examples + 1, 
                             figsize=(n_examples * 2 + 2, n_pct * 2 + 2),
                             gridspec_kw={'width_ratios': [1.2] + [1]*n_examples})
    
    fig.suptitle(f'Mask Percentage Comparison ({mask_type})\nRed = masked region', 
                 fontsize=14, fontweight='bold')
    
    cond_gen = ConditioningGenerator()
    
    # Header row
    axes[0, 0].text(0.5, 0.5, 'MASK\n%', ha='center', va='center', 
                    fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    for col, idx in enumerate(example_indices):
        img, label = mnist[idx]
        img_np = np.array(img)
        axes[0, col + 1].imshow(img_np, cmap='gray')
        axes[0, col + 1].set_title(f'Digit {label}', fontsize=10, fontweight='bold')
        axes[0, col + 1].axis('off')
    
    # Each percentage row
    for row, pct in enumerate(percentages, start=1):
        # Label
        axes[row, 0].text(0.5, 0.5, f'{pct}%', ha='center', va='center', 
                         fontsize=14, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
        axes[row, 0].axis('off')
        
        for col, idx in enumerate(example_indices):
            img, label = mnist[idx]
            img_np = np.array(img)
            
            masked_img, mask = cond_gen.get_partial_mask(img_np, mask_type, pct)
            
            # RGB visualization
            rgb_img = np.stack([img_np, img_np, img_np], axis=-1).astype(float) / 255.0
            mask_normalized = mask.astype(float) / 255.0
            rgb_img[:, :, 0] = np.clip(rgb_img[:, :, 0] + mask_normalized * 0.7, 0, 1)
            rgb_img[:, :, 1] = rgb_img[:, :, 1] * (1 - mask_normalized * 0.5)
            rgb_img[:, :, 2] = rgb_img[:, :, 2] * (1 - mask_normalized * 0.5)
            
            axes[row, col + 1].imshow(rgb_img)
            axes[row, col + 1].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {save_path}")
    return save_path


def visualize_masked_vs_original(n_examples=6, mask_type='digit_square', mask_percentage=50,
                                  save_path='visualizations/masked_vs_original.png'):
    """
    Simple side-by-side: Original | Masked Input | What model must reconstruct
    """
    import torchvision
    mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    
    example_indices = list(range(n_examples))
    
    fig, axes = plt.subplots(3, n_examples, figsize=(n_examples * 2, 6))
    
    fig.suptitle(f'Inpainting Task: Model sees masked input, must reconstruct original\n'
                 f'Mask type: {mask_type}, {mask_percentage}% hidden', 
                 fontsize=12, fontweight='bold')
    
    cond_gen = ConditioningGenerator()
    
    for col, idx in enumerate(example_indices):
        img, label = mnist[idx]
        img_np = np.array(img)
        
        masked_img, mask = cond_gen.get_partial_mask(img_np, mask_type, mask_percentage)
        
        # Row 0: Original
        axes[0, col].imshow(img_np, cmap='gray')
        axes[0, col].axis('off')
        axes[0, col].set_title(f'Digit {label}', fontsize=10)
        
        # Row 1: Masked input (what model sees)
        axes[1, col].imshow(masked_img, cmap='gray')
        axes[1, col].axis('off')
        
        # Row 2: Mask visualization
        axes[2, col].imshow(mask, cmap='Reds')
        axes[2, col].axis('off')
    
    # Row labels
    axes[0, 0].set_ylabel('ORIGINAL\n(Ground Truth)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('MASKED INPUT\n(Model sees this)', fontsize=11, fontweight='bold')
    axes[2, 0].set_ylabel('MASK\n(Hidden region)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description='Visualize masking strategies for report')
    parser.add_argument('--n_examples', type=int, default=4, help='Number of example digits')
    parser.add_argument('--mask_percentage', type=int, default=50, help='Default mask percentage')
    parser.add_argument('--save_dir', type=str, default='visualizations', help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*60)
    print("GENERATING MASK STRATEGY VISUALIZATIONS FOR REPORT")
    print("="*60)
    
    # 1. All mask types comparison
    print("\n[1/3] Generating all mask types comparison...")
    visualize_all_masks(
        n_examples=args.n_examples,
        mask_percentage=args.mask_percentage,
        save_path=f'{args.save_dir}/mask_strategies.png'
    )
    
    # 2. Mask percentage comparison
    print("\n[2/3] Generating mask percentage comparison...")
    visualize_mask_percentage_comparison(
        mask_type='digit_square',
        percentages=[20, 40, 60, 80],
        n_examples=args.n_examples,
        save_path=f'{args.save_dir}/mask_percentage_comparison.png'
    )
    
    # 3. Simple masked vs original
    print("\n[3/3] Generating masked vs original visualization...")
    visualize_masked_vs_original(
        n_examples=6,
        mask_type='digit_square',
        mask_percentage=args.mask_percentage,
        save_path=f'{args.save_dir}/masked_vs_original.png'
    )
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\nGenerated files in {args.save_dir}/:")
    print("  - mask_strategies.png (all mask types)")
    print("  - mask_percentage_comparison.png (20/40/60/80%)")
    print("  - masked_vs_original.png (simple overview)")


if __name__ == '__main__':
    main()
