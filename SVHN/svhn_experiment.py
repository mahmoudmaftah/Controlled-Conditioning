"""
SVHN (Street View House Numbers) Dataset Experiment
Dedicated script for training and evaluating diffusion models on SVHN
"""

import sys
sys.path.append('..')  # Add parent directory for imports

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

# Import model components from parent
from model import create_model, count_parameters
from train import DiffusionTrainer, linear_beta_schedule

# ============================================================================
# SVHN-Specific Conditioning Generators
# ============================================================================

class SVHNConditioningGenerator:
    """Conditioning signal generators optimized for SVHN (32x32 RGB digit images)"""
    
    @staticmethod
    def to_grayscale(img_np):
        """Convert RGB to grayscale"""
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        return img_np
    
    @staticmethod
    def get_edge_map(img_np):
        """Canny edge detection - good for digit outlines"""
        gray = SVHNConditioningGenerator.to_grayscale(img_np)
        # SVHN has more noise than MNIST, use higher thresholds
        edges = cv2.Canny(gray, 80, 200)
        return edges
    
    @staticmethod
    def get_sobel_edges(img_np):
        """Sobel edge map - captures gradients"""
        gray = SVHNConditioningGenerator.to_grayscale(img_np)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = (edges / (edges.max() + 1e-8) * 255).astype(np.uint8)
        return edges
    
    @staticmethod
    def get_color_histogram(img_np, bins=16):
        """
        Color histogram - captures color distribution
        Returns: (48,) vector for RGB
        """
        hist_r = np.histogram(img_np[:, :, 0], bins=bins, range=(0, 256))[0]
        hist_g = np.histogram(img_np[:, :, 1], bins=bins, range=(0, 256))[0]
        hist_b = np.histogram(img_np[:, :, 2], bins=bins, range=(0, 256))[0]
        hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
        hist = hist / (hist.sum() + 1e-8)
        return hist
    
    @staticmethod
    def get_dominant_color(img_np, n_colors=3):
        """
        Extract dominant colors using K-means
        Returns: (n_colors * 3,) vector of RGB values normalized to [0, 1]
        """
        pixels = img_np.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        
        # Sort by frequency
        counts = np.bincount(labels.flatten(), minlength=n_colors)
        sorted_idx = np.argsort(-counts)
        centers = centers[sorted_idx]
        
        return (centers.flatten() / 255.0).astype(np.float32)
    
    @staticmethod
    def get_inpainting_mask(img_np, mask_type='random_rect'):
        """
        Generate inpainting mask and masked image
        Returns: (masked_image, mask)
        """
        H, W, C = img_np.shape
        mask = np.zeros((H, W), dtype=np.uint8)
        
        if mask_type == 'center':
            # Mask center region (where digit usually is)
            mask[8:24, 8:24] = 255
        elif mask_type == 'random_rect':
            x1 = np.random.randint(4, 16)
            y1 = np.random.randint(4, 16)
            x2 = np.random.randint(16, 28)
            y2 = np.random.randint(16, 28)
            mask[y1:y2, x1:x2] = 255
        elif mask_type == 'top_half':
            mask[:16, :] = 255
        elif mask_type == 'bottom_half':
            mask[16:, :] = 255
        
        masked_img = img_np.copy()
        masked_img[mask > 0] = 0
        
        return masked_img, mask
    
    @staticmethod
    def get_brightness_contrast(img_np):
        """
        Compute brightness and contrast statistics
        Returns: (6,) vector [mean_r, mean_g, mean_b, std_r, std_g, std_b]
        """
        means = img_np.mean(axis=(0, 1)) / 255.0
        stds = img_np.std(axis=(0, 1)) / 255.0
        return np.concatenate([means, stds]).astype(np.float32)
    
    @staticmethod
    def get_digit_mask(img_np, threshold=50):
        """
        Simple digit segmentation based on intensity difference from background
        Returns: binary mask highlighting the digit region
        """
        gray = SVHNConditioningGenerator.to_grayscale(img_np)
        
        # Adaptive thresholding works better for SVHN
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Clean up with morphology
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary


# ============================================================================
# SVHN Dataset Class
# ============================================================================

class SVHNConditioningDataset(Dataset):
    """
    SVHN dataset with multiple conditioning types
    
    Supported conditioning types:
    - 'edge': Canny edge map (B, 1, 32, 32)
    - 'sobel': Sobel edge map (B, 1, 32, 32)
    - 'color_histogram': Color histogram vector (B, 48)
    - 'dominant_color': Dominant RGB colors (B, 9)
    - 'inpainting': Masked image + mask (B, 4, 32, 32)
    - 'brightness_contrast': Mean/std per channel (B, 6)
    - 'digit_mask': Binary digit segmentation (B, 1, 32, 32)
    """
    
    CONDITIONING_TYPES = {
        'edge': {'channels': 1, 'is_vector': False},
        'sobel': {'channels': 1, 'is_vector': False},
        'color_histogram': {'channels': 48, 'is_vector': True},
        'dominant_color': {'channels': 9, 'is_vector': True},
        'inpainting': {'channels': 4, 'is_vector': False},  # RGB + mask
        'brightness_contrast': {'channels': 6, 'is_vector': True},
        'digit_mask': {'channels': 1, 'is_vector': False},
    }
    
    def __init__(self, train=True, conditioning='edge', mask_type='random_rect', data_root='../data'):
        self.conditioning = conditioning
        self.mask_type = mask_type
        self.cond_gen = SVHNConditioningGenerator()
        
        # Validate conditioning type
        if conditioning not in self.CONDITIONING_TYPES:
            raise ValueError(f"Unknown conditioning: {conditioning}. "
                           f"Available: {list(self.CONDITIONING_TYPES.keys())}")
        
        split = 'train' if train else 'test'
        self.svhn = torchvision.datasets.SVHN(
            root=data_root,
            split=split,
            download=True
        )
        
        print(f"SVHN {split} set loaded: {len(self.svhn)} samples")
        print(f"Conditioning: {conditioning} ({self.CONDITIONING_TYPES[conditioning]})")
    
    def __len__(self):
        return len(self.svhn)
    
    def __getitem__(self, idx):
        img, label = self.svhn[idx]
        
        # SVHN returns PIL Image, convert to numpy (H, W, C)
        img_np = np.array(img)
        
        # Convert to tensor [3, 32, 32], normalize to [-1, 1]
        img_tensor = transforms.ToTensor()(img)
        img_tensor = (img_tensor - 0.5) / 0.5
        
        # Generate conditioning
        cond_tensor = self._generate_conditioning(img_np)
        
        return {
            'image': img_tensor,
            'condition': cond_tensor,
            'label': label
        }
    
    def _generate_conditioning(self, img_np):
        """Generate conditioning tensor based on type"""
        
        if self.conditioning == 'edge':
            cond = self.cond_gen.get_edge_map(img_np)
            return torch.from_numpy(cond).float().unsqueeze(0) / 127.5 - 1.0
        
        elif self.conditioning == 'sobel':
            cond = self.cond_gen.get_sobel_edges(img_np)
            return torch.from_numpy(cond).float().unsqueeze(0) / 127.5 - 1.0
        
        elif self.conditioning == 'color_histogram':
            hist = self.cond_gen.get_color_histogram(img_np)
            return torch.from_numpy(hist * 2 - 1).float()
        
        elif self.conditioning == 'dominant_color':
            colors = self.cond_gen.get_dominant_color(img_np)
            return torch.from_numpy(colors * 2 - 1).float()
        
        elif self.conditioning == 'inpainting':
            masked_img, mask = self.cond_gen.get_inpainting_mask(img_np, self.mask_type)
            masked_tensor = torch.from_numpy(masked_img).float().permute(2, 0, 1) / 127.5 - 1.0
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 127.5 - 1.0
            return torch.cat([masked_tensor, mask_tensor], dim=0)
        
        elif self.conditioning == 'brightness_contrast':
            bc = self.cond_gen.get_brightness_contrast(img_np)
            return torch.from_numpy(bc * 2 - 1).float()
        
        elif self.conditioning == 'digit_mask':
            mask = self.cond_gen.get_digit_mask(img_np)
            return torch.from_numpy(mask).float().unsqueeze(0) / 127.5 - 1.0
        
        else:
            raise ValueError(f"Unknown conditioning: {self.conditioning}")
    
    @classmethod
    def get_conditioning_info(cls, conditioning_type):
        """Get info about a conditioning type"""
        return cls.CONDITIONING_TYPES.get(conditioning_type, {'channels': 1, 'is_vector': False})


def get_svhn_dataloader(batch_size=64, train=True, conditioning='edge', 
                        mask_type='random_rect', num_workers=0):
    """Get SVHN dataloader with specified conditioning"""
    dataset = SVHNConditioningDataset(
        train=train,
        conditioning=conditioning,
        mask_type=mask_type
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )


# ============================================================================
# SVHN Training Function
# ============================================================================

def train_svhn(model_type='lora', rank=4, inject_layer='mid', 
               conditioning='edge', epochs=10, batch_size=128, lr=1e-4,
               timesteps=1000, save_dir='checkpoints'):
    """
    Train diffusion model on SVHN dataset
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("SVHN TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model: {model_type}, Rank: {rank}, Inject: {inject_layer}")
    print(f"Conditioning: {conditioning}")
    
    # Get conditioning info
    cond_info = SVHNConditioningDataset.get_conditioning_info(conditioning)
    cond_channels = cond_info['channels']
    is_vector = cond_info['is_vector']
    
    print(f"Conditioning channels: {cond_channels}, Vector: {is_vector}")
    
    # Adjust model type for vector conditioning
    if is_vector and model_type == 'lora':
        model_type_actual = 'lora_vector'
    else:
        model_type_actual = model_type
    
    # Create model (32x32 RGB images)
    model = create_model(
        model_type=model_type_actual,
        rank=rank,
        inject_layer=inject_layer,
        conditioning_channels=cond_channels,
        conditioning_type=conditioning,
        image_size=32,
        in_channels=3
    )
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    
    # Data
    train_loader = get_svhn_dataloader(
        batch_size=batch_size,
        train=True,
        conditioning=conditioning
    )
    
    # Trainer
    trainer = DiffusionTrainer(
        model, 
        device=device, 
        timesteps=timesteps,
        image_size=32,
        in_channels=3
    )
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    os.makedirs(save_dir, exist_ok=True)
    history = {'loss': [], 'lr': []}
    
    is_conditional = model_type != 'baseline'
    
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
        
        # Save checkpoint
        ckpt_path = os.path.join(save_dir, f'svhn_{model_type}_{conditioning}_epoch{epoch}.pt')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'loss': avg_loss,
            'config': {
                'model_type': model_type,
                'rank': rank,
                'inject_layer': inject_layer,
                'conditioning': conditioning,
                'timesteps': timesteps,
                'dataset': 'svhn',
                'image_size': 32,
                'in_channels': 3
            }
        }, ckpt_path)
    
    # Save final
    final_path = os.path.join(save_dir, f'svhn_{model_type}_{conditioning}_final.pt')
    torch.save({
        'model': model.state_dict(),
        'history': history,
        'config': {
            'model_type': model_type,
            'rank': rank,
            'inject_layer': inject_layer,
            'conditioning': conditioning,
            'timesteps': timesteps,
            'dataset': 'svhn',
            'image_size': 32,
            'in_channels': 3,
            'total_params': total_params,
            'trainable_params': trainable_params
        }
    }, final_path)
    
    print(f"\nTraining complete! Saved to: {final_path}")
    return model, trainer, history


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_svhn_samples(conditioning='edge', num_samples=8, save_path=None):
    """Visualize SVHN samples with conditioning"""
    
    dataset = SVHNConditioningDataset(train=True, conditioning=conditioning)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 5))
    fig.suptitle(f'SVHN - {conditioning} Conditioning', fontsize=14)
    
    cond_info = SVHNConditioningDataset.get_conditioning_info(conditioning)
    is_vector = cond_info['is_vector']
    
    for i in range(num_samples):
        sample = dataset[np.random.randint(len(dataset))]
        
        # Original image
        img = (sample['image'] + 1) / 2  # Denormalize
        img_np = img.permute(1, 2, 0).numpy()
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"Label: {sample['label']}")
        axes[0, i].axis('off')
        
        # Conditioning
        if is_vector:
            cond = sample['condition'].numpy()
            axes[1, i].bar(range(len(cond)), cond, color='steelblue')
            axes[1, i].set_ylim(-1.2, 1.2)
            axes[1, i].set_title(f'dim={len(cond)}')
        else:
            cond = (sample['condition'] + 1) / 2
            if cond.shape[0] > 1:  # Multi-channel (inpainting)
                cond_np = cond[:3].permute(1, 2, 0).numpy()  # Show RGB part
            else:
                cond_np = cond.squeeze().numpy()
            axes[1, i].imshow(cond_np, cmap='gray' if cond.shape[0] == 1 else None)
            axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def visualize_svhn_generation(checkpoint_path, num_samples=8, save_path=None, device='cuda'):
    """Visualize generated SVHN samples"""
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Recreate model
    cond_info = SVHNConditioningDataset.get_conditioning_info(config['conditioning'])
    
    model_type = config['model_type']
    if cond_info['is_vector'] and model_type == 'lora':
        model_type = 'lora_vector'
    
    model = create_model(
        model_type=model_type,
        rank=config.get('rank', 4),
        inject_layer=config.get('inject_layer', 'mid'),
        conditioning_channels=cond_info['channels'],
        conditioning_type=config['conditioning'],
        image_size=32,
        in_channels=3
    )
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # Trainer for sampling
    trainer = DiffusionTrainer(
        model, device=device,
        timesteps=config.get('timesteps', 1000),
        image_size=32, in_channels=3
    )
    
    # Get test data
    test_loader = get_svhn_dataloader(
        batch_size=num_samples,
        train=False,
        conditioning=config['conditioning']
    )
    batch = next(iter(test_loader))
    
    # Generate
    print("Generating samples...")
    with torch.no_grad():
        generated = trainer.sample(
            condition=batch['condition'].to(device),
            n_samples=num_samples,
            is_conditional=True
        ).cpu()
    
    # Visualize
    is_vector = cond_info['is_vector']
    n_rows = 3
    fig, axes = plt.subplots(n_rows, num_samples, figsize=(2*num_samples, 2.5*n_rows))
    fig.suptitle(f"SVHN Generation | {config['model_type']} | {config['conditioning']}", fontsize=14)
    
    for i in range(num_samples):
        # Real
        real = (batch['image'][i] + 1) / 2
        axes[0, i].imshow(real.permute(1, 2, 0).numpy())
        axes[0, i].set_title('Real' if i == 0 else '')
        axes[0, i].axis('off')
        
        # Condition
        if is_vector:
            cond = batch['condition'][i].numpy()
            axes[1, i].bar(range(len(cond)), cond, color='steelblue')
            axes[1, i].set_ylim(-1.2, 1.2)
            axes[1, i].set_title('Cond' if i == 0 else '')
        else:
            cond = (batch['condition'][i] + 1) / 2
            if cond.shape[0] > 1:
                cond_np = cond[:3].permute(1, 2, 0).numpy()
            else:
                cond_np = cond.squeeze().numpy()
            axes[1, i].imshow(cond_np, cmap='gray' if batch['condition'].shape[1] == 1 else None)
            axes[1, i].set_title('Cond' if i == 0 else '')
            axes[1, i].axis('off')
        
        # Generated
        gen = (generated[i] + 1) / 2
        axes[2, i].imshow(gen.permute(1, 2, 0).numpy().clip(0, 1))
        axes[2, i].set_title('Gen' if i == 0 else '')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def list_svhn_conditionings():
    """Print available conditioning types for SVHN"""
    print("\n" + "="*50)
    print("SVHN CONDITIONING TYPES")
    print("="*50)
    for name, info in SVHNConditioningDataset.CONDITIONING_TYPES.items():
        type_str = "vector" if info['is_vector'] else "spatial"
        print(f"  {name:<20} {info['channels']:>3} channels ({type_str})")
    print("="*50)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SVHN Experiment')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'visualize', 'generate', 'list'])
    parser.add_argument('--conditioning', type=str, default='edge')
    parser.add_argument('--model_type', type=str, default='lora')
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=8)
    args = parser.parse_args()
    
    if args.mode == 'list':
        list_svhn_conditionings()
    
    elif args.mode == 'train':
        train_svhn(
            model_type=args.model_type,
            rank=args.rank,
            conditioning=args.conditioning,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir='checkpoints'
        )
    
    elif args.mode == 'visualize':
        os.makedirs('visualizations', exist_ok=True)
        visualize_svhn_samples(
            conditioning=args.conditioning,
            num_samples=args.num_samples,
            save_path=f'visualizations/svhn_{args.conditioning}_samples.png'
        )
    
    elif args.mode == 'generate':
        if args.checkpoint is None:
            print("Error: --checkpoint required for generate mode")
        else:
            os.makedirs('visualizations', exist_ok=True)
            visualize_svhn_generation(
                args.checkpoint,
                num_samples=args.num_samples,
                save_path=f'visualizations/svhn_generation.png'
            )
