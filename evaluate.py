import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import cv2
import os
import glob
import argparse
from model import create_model, count_parameters
from dataset import get_dataloader, get_conditioning_channels, is_vector_conditioning
from train import DiffusionTrainer

# ============================================================================
# Evaluation Metrics
# ============================================================================

def edge_alignment_score(generated, condition):
    """
    Measure how well generated images align with edge conditioning.
    Higher is better.
    """
    gen_np = generated.cpu().numpy()
    cond_np = condition.cpu().numpy()
    
    scores = []
    for i in range(gen_np.shape[0]):
        gen_img = ((gen_np[i, 0] + 1) * 127.5).astype(np.uint8)
        cond_img = ((cond_np[i, 0] + 1) * 127.5).astype(np.uint8)
        
        # Extract edges from generated
        gen_edges = cv2.Canny(gen_img, 50, 150)
        
        # Compute overlap
        overlap = np.sum((gen_edges > 0) & (cond_img > 0))
        total_cond = np.sum(cond_img > 0)
        
        if total_cond > 0:
            scores.append(overlap / total_cond)
        else:
            scores.append(0.0)
    
    return np.mean(scores)


def compute_mse(generated, target):
    """MSE between generated and target images"""
    return mean_squared_error(
        target.cpu().numpy().flatten(),
        generated.cpu().numpy().flatten()
    )


def structural_similarity(generated, target):
    """
    Simple structural similarity measure.
    Uses normalized cross-correlation.
    """
    gen_np = generated.cpu().numpy()
    target_np = target.cpu().numpy()
    
    scores = []
    for i in range(gen_np.shape[0]):
        gen_img = gen_np[i, 0]
        target_img = target_np[i, 0]
        
        # Normalize
        gen_norm = (gen_img - gen_img.mean()) / (gen_img.std() + 1e-8)
        target_norm = (target_img - target_img.mean()) / (target_img.std() + 1e-8)
        
        # Cross-correlation
        corr = np.mean(gen_norm * target_norm)
        scores.append(corr)
    
    return np.mean(scores)

# ============================================================================
# Visualization
# ============================================================================

def visualize_samples(model, trainer, test_loader, n_samples=8, save_path='samples.png',
                      conditioning_type='edge', is_conditional=True):
    """Generate and visualize samples"""
    model.eval()
    
    # Get test batch
    batch = next(iter(test_loader))
    condition = batch['condition'][:n_samples]
    target = batch['image'][:n_samples]
    
    # Determine if condition is vector or image
    is_vector = is_vector_conditioning(conditioning_type)
    
    # Sample
    with torch.no_grad():
        if is_conditional:
            generated = trainer.sample(condition, n_samples=n_samples, is_conditional=True)
        else:
            generated = trainer.sample(n_samples=n_samples, is_conditional=False)
    
    # Plot
    if is_vector:
        # For vector conditioning, show 2 rows: Generated and Target
        fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
        
        for i in range(n_samples):
            # Generated
            axes[0, i].imshow(generated[i, 0].cpu().numpy(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Generated', fontsize=10)
            
            # Target
            axes[1, i].imshow(target[i, 0].cpu().numpy(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Target', fontsize=10)
    else:
        # For image conditioning, show 3 rows: Condition, Generated, Target
        fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 2, 6))
        
        for i in range(n_samples):
            # Condition (may be multi-channel for inpainting)
            if condition.shape[1] == 1:
                axes[0, i].imshow(condition[i, 0].cpu().numpy(), cmap='gray')
            else:
                # For inpainting: show masked image
                axes[0, i].imshow(condition[i, 0].cpu().numpy(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title(f'Condition ({conditioning_type})', fontsize=10)
            
            # Generated
            axes[1, i].imshow(generated[i, 0].cpu().numpy(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Generated', fontsize=10)
            
            # Target
            axes[2, i].imshow(target[i, 0].cpu().numpy(), cmap='gray')
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_title('Target', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


def plot_training_curves(histories, save_path='training_curves.png'):
    """Plot training loss curves for multiple models"""
    plt.figure(figsize=(10, 6))
    
    for name, history in histories.items():
        plt.plot(history['loss'], label=name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")

# ============================================================================
# Full Evaluation
# ============================================================================

def evaluate_model(checkpoint_path, model_type, rank=4, inject_layer='mid',
                   conditioning_type='edge', timesteps=1000, num_batches=1):
    """
    Evaluate a trained model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_type: Type of model
        rank: LoRA rank
        inject_layer: Injection layer
        conditioning_type: Type of conditioning used
        timesteps: Diffusion timesteps
        num_batches: Number of batches to evaluate
    
    Returns:
        Dictionary of evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found: {checkpoint_path}")
        return None
    
    # Get conditioning channels
    cond_channels = get_conditioning_channels(conditioning_type)
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
        conditioning_type=conditioning_type
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Try to load state dict, handle architecture mismatches
    try:
        model.load_state_dict(checkpoint['model'])
    except RuntimeError as e:
        print(f"Warning: Architecture mismatch for {checkpoint_path}")
        print(f"  This checkpoint was trained with a different model version.")
        print(f"  Please retrain with: python train.py --model_type {model_type} --conditioning {conditioning_type}")
        return None
    
    model = model.to(device)
    
    # Create trainer
    trainer = DiffusionTrainer(model, device=device, timesteps=timesteps)
    
    # Get test data
    test_loader = get_dataloader(
        batch_size=64, 
        train=False, 
        conditioning=conditioning_type,
        num_workers=0
    )
    
    # Determine if conditional
    is_conditional = model_type != 'baseline'
    
    # Compute metrics
    print(f"\nEvaluating {model_type} (rank={rank}, layer={inject_layer})...")
    
    all_edge_scores = []
    all_mse = []
    all_ssim = []
    
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= num_batches:
            break
            
        condition = batch['condition'].to(device)
        target = batch['image'].to(device)
        
        with torch.no_grad():
            if is_conditional:
                generated = trainer.sample(condition, n_samples=condition.shape[0], 
                                          is_conditional=True)
            else:
                generated = trainer.sample(n_samples=target.shape[0], 
                                          is_conditional=False)
        
        # Metrics
        if not is_vector and conditioning_type in ['edge', 'sobel', 'skeleton']:
            edge_score = edge_alignment_score(generated, condition)
            all_edge_scores.append(edge_score)
        
        mse = compute_mse(generated, target)
        ssim = structural_similarity(generated, target)
        
        all_mse.append(mse)
        all_ssim.append(ssim)
    
    # Print results
    results = {
        'mse': np.mean(all_mse),
        'ssim': np.mean(all_ssim)
    }
    
    if all_edge_scores:
        results['edge_score'] = np.mean(all_edge_scores)
        print(f"Edge Alignment Score: {results['edge_score']:.4f}")
    
    print(f"MSE: {results['mse']:.4f}")
    print(f"SSIM: {results['ssim']:.4f}")
    
    # Visualize
    save_name = f"samples_{model_type}_rank{rank}_{inject_layer}_{conditioning_type}.png"
    visualize_samples(
        model, trainer, test_loader, 
        conditioning_type=conditioning_type,
        is_conditional=is_conditional,
        save_path=save_name
    )
    
    return results


def find_checkpoints(checkpoint_dir='checkpoints', pattern='*.pt'):
    """Find all checkpoint files matching pattern"""
    return glob.glob(os.path.join(checkpoint_dir, pattern))

# ============================================================================
# Comparative Analysis
# ============================================================================

def compare_all_models(checkpoint_dir='checkpoints', conditioning_type='edge'):
    """
    Compare all trained models found in checkpoint directory.
    Gracefully handles missing checkpoints.
    """
    
    results = {}
    
    # Look for available checkpoints
    checkpoints = find_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}/")
        print("Run training first with: python train.py --experiment full_comparison")
        return results
    
    print(f"Found {len(checkpoints)} checkpoint(s):")
    for ckpt in checkpoints:
        print(f"  - {os.path.basename(ckpt)}")
    
    # Try to evaluate each checkpoint
    for ckpt_path in checkpoints:
        try:
            # Parse checkpoint name to extract config
            ckpt_name = os.path.basename(ckpt_path)
            
            # Load checkpoint to get config
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            config = checkpoint.get('config', {})
            
            # If no config in checkpoint, try to parse from filename
            if not config:
                # Parse old format: model_rank4_layer_epoch4.pt
                parts = ckpt_name.replace('.pt', '').split('_')
                model_type = parts[0] if parts else 'lora'
                rank = 4
                inject_layer = 'mid'
                
                for i, p in enumerate(parts):
                    if p.startswith('rank'):
                        rank = int(p.replace('rank', ''))
                    elif p in ['early', 'mid', 'all']:
                        inject_layer = p
                
                config = {
                    'model_type': model_type,
                    'rank': rank,
                    'inject_layer': inject_layer,
                    'conditioning_type': conditioning_type,
                    'timesteps': 1000
                }
            
            model_type = config.get('model_type', 'lora')
            rank = config.get('rank', 4)
            inject_layer = config.get('inject_layer', 'mid')
            cond_type = config.get('conditioning_type', conditioning_type)
            timesteps = config.get('timesteps', 1000)
            
            # Evaluate
            metrics = evaluate_model(
                ckpt_path,
                model_type=model_type,
                rank=rank,
                inject_layer=inject_layer,
                conditioning_type=cond_type,
                timesteps=timesteps
            )
            
            if metrics:
                results[ckpt_name] = metrics
                
        except Exception as e:
            print(f"Error evaluating {ckpt_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print comparison table
    if results:
        print("\n" + "="*70)
        print("RESULTS COMPARISON")
        print("="*70)
        print(f"{'Checkpoint':<40} {'Edge Score':<12} {'MSE':<12} {'SSIM':<12}")
        print("-"*70)
        for model, metrics in results.items():
            edge = metrics.get('edge_score', 'N/A')
            mse = metrics.get('mse', 'N/A')
            ssim = metrics.get('ssim', 'N/A')
            
            edge_str = f"{edge:.4f}" if isinstance(edge, float) else edge
            mse_str = f"{mse:.4f}" if isinstance(mse, float) else mse
            ssim_str = f"{ssim:.4f}" if isinstance(ssim, float) else ssim
            
            print(f"{model:<40} {edge_str:<12} {mse_str:<12} {ssim_str:<12}")
        print("="*70)
    
    return results


def evaluate_single_checkpoint(checkpoint_path):
    """Evaluate a single checkpoint file"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    # Load config from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    
    return evaluate_model(
        checkpoint_path,
        model_type=config.get('model_type', 'lora'),
        rank=config.get('rank', 4),
        inject_layer=config.get('inject_layer', 'mid'),
        conditioning_type=config.get('conditioning_type', 'edge'),
        timesteps=config.get('timesteps', 1000)
    )


# ============================================================================
# CLI Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained diffusion models')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to specific checkpoint to evaluate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory containing checkpoints')
    parser.add_argument('--conditioning', type=str, default='edge',
                       help='Conditioning type')
    parser.add_argument('--compare_all', action='store_true',
                       help='Compare all available checkpoints')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.checkpoint:
        # Evaluate single checkpoint
        evaluate_single_checkpoint(args.checkpoint)
    elif args.compare_all:
        # Compare all models
        compare_all_models(
            checkpoint_dir=args.checkpoint_dir,
            conditioning_type=args.conditioning
        )
    else:
        # Default: compare all
        compare_all_models(
            checkpoint_dir=args.checkpoint_dir,
            conditioning_type=args.conditioning
        )