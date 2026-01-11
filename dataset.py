import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2
import numpy as np
from scipy import ndimage
import os
from PIL import Image
import json

# ============================================================================
# Dataset Configuration
# ============================================================================

DATASET_CONFIG = {
    'mnist': {
        'image_size': 28,
        'channels': 1,
        'num_classes': 10,
        'conditioning_types': ['edge', 'sobel', 'skeleton', 'inpainting', 'stroke_thickness', 'center_scale'],
    },
    'svhn': {
        'image_size': 32,
        'channels': 3,
        'num_classes': 10,
        'conditioning_types': ['edge', 'sobel', 'color_histogram', 'inpainting', 'bounding_box', 'center_scale'],
    },
    'clevr': {
        'image_size': 64,  # Resized from original 128/240
        'channels': 3,
        'num_classes': None,  # CLEVR doesn't have discrete classes
        'conditioning_types': ['edge', 'sobel', 'segmentation', 'object_positions', 'inpainting', 'color_histogram'],
    },
}

# ============================================================================
# Conditioning Signal Generators
# ============================================================================

class ConditioningGenerator:
    """Base class for conditioning signal generation - supports grayscale and RGB images"""
    
    @staticmethod
    def to_grayscale(img_np):
        """Convert RGB to grayscale if needed"""
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        return img_np
    
    @staticmethod
    def get_edge_map(img_np):
        """Generate Canny edge map - works for both grayscale and RGB"""
        gray = ConditioningGenerator.to_grayscale(img_np)
        edges = cv2.Canny(gray, 50, 150)
        return edges
    
    @staticmethod
    def get_sobel_edges(img_np):
        """Generate Sobel edge map - works for both grayscale and RGB"""
        gray = ConditioningGenerator.to_grayscale(img_np)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = (edges / edges.max() * 255).astype(np.uint8) if edges.max() > 0 else edges.astype(np.uint8)
        return edges
    
    @staticmethod
    def get_skeleton(img_np):
        """Generate morphological skeleton using iterative erosion"""
        gray = ConditioningGenerator.to_grayscale(img_np)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        skeleton = np.zeros_like(binary)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        while True:
            eroded = cv2.erode(binary, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(binary, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            binary = eroded.copy()
            if cv2.countNonZero(binary) == 0:
                break
        
        return skeleton
    
    @staticmethod
    def get_color_histogram(img_np, bins=16):
        """
        Generate color histogram as conditioning (for RGB images)
        Returns: vector of shape (bins * 3,) for RGB or (bins,) for grayscale
        """
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            # RGB histogram
            hist_r = np.histogram(img_np[:, :, 0], bins=bins, range=(0, 256))[0]
            hist_g = np.histogram(img_np[:, :, 1], bins=bins, range=(0, 256))[0]
            hist_b = np.histogram(img_np[:, :, 2], bins=bins, range=(0, 256))[0]
            hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
        else:
            # Grayscale histogram
            hist = np.histogram(img_np, bins=bins, range=(0, 256))[0].astype(np.float32)
        
        # Normalize to [0, 1]
        hist = hist / (hist.sum() + 1e-8)
        return hist
    
    @staticmethod
    def get_partial_mask(img_np, mask_type='top_half', mask_percentage=50):
        """
        Generate partial mask for inpainting conditioning
        Returns: (masked_image, mask) - works for any image size and channels
        mask_type: 'top_half', 'bottom_half', 'left_half', 'right_half', 'random_rect', 'center',
                   'digit_percentage' (masks X% of actual digit pixels)
        mask_percentage: For 'digit_percentage' mode, how much of the digit to hide (0-100)
        """
        if len(img_np.shape) == 3:
            H, W, C = img_np.shape
        else:
            H, W = img_np.shape
            C = 1
        
        mask = np.zeros((H, W), dtype=np.uint8)
        
        if mask_type == 'top_half':
            mask[:H//2, :] = 255
        elif mask_type == 'bottom_half':
            mask[H//2:, :] = 255
        elif mask_type == 'left_half':
            mask[:, :W//2] = 255
        elif mask_type == 'right_half':
            mask[:, W//2:] = 255
        elif mask_type == 'center':
            mask[H//4:3*H//4, W//4:3*W//4] = 255
        elif mask_type == 'random_rect':
            # Random rectangle mask
            x1 = np.random.randint(0, W//2)
            y1 = np.random.randint(0, H//2)
            x2 = np.random.randint(W//2, W)
            y2 = np.random.randint(H//2, H)
            mask[y1:y2, x1:x2] = 255
        elif mask_type == 'digit_percentage':
            # Smart masking: only mask a percentage of the digit pixels (non-black areas)
            gray = ConditioningGenerator.to_grayscale(img_np)
            
            # Find digit pixels (foreground) - threshold to find non-black pixels
            digit_pixels = gray > 30  # Threshold to separate digit from background
            digit_coords = np.argwhere(digit_pixels)
            
            if len(digit_coords) > 0:
                # Calculate how many pixels to mask
                num_digit_pixels = len(digit_coords)
                num_to_mask = int(num_digit_pixels * mask_percentage / 100)
                
                # Randomly select which digit pixels to mask
                if num_to_mask > 0:
                    indices_to_mask = np.random.choice(num_digit_pixels, num_to_mask, replace=False)
                    mask_coords = digit_coords[indices_to_mask]
                    mask[mask_coords[:, 0], mask_coords[:, 1]] = 255
        elif mask_type == 'digit_contiguous':
            # Mask a contiguous region of the digit (more realistic occlusion)
            gray = ConditioningGenerator.to_grayscale(img_np)
            digit_pixels = gray > 30
            digit_coords = np.argwhere(digit_pixels)
            
            if len(digit_coords) > 0:
                # Pick a random seed point from the digit
                seed_idx = np.random.randint(len(digit_coords))
                seed_y, seed_x = digit_coords[seed_idx]
                
                # Grow mask from seed using flood-fill-like expansion
                num_digit_pixels = len(digit_coords)
                num_to_mask = int(num_digit_pixels * mask_percentage / 100)
                
                # Use distance from seed to select nearby pixels
                distances = np.sqrt((digit_coords[:, 0] - seed_y)**2 + 
                                   (digit_coords[:, 1] - seed_x)**2)
                sorted_indices = np.argsort(distances)
                mask_coords = digit_coords[sorted_indices[:num_to_mask]]
                if len(mask_coords) > 0:
                    mask[mask_coords[:, 0], mask_coords[:, 1]] = 255
        elif mask_type == 'digit_top':
            # Mask top portion of the digit only
            gray = ConditioningGenerator.to_grayscale(img_np)
            digit_pixels = gray > 30
            digit_coords = np.argwhere(digit_pixels)
            
            if len(digit_coords) > 0:
                # Sort by y-coordinate (top to bottom) and mask top percentage
                sorted_by_y = digit_coords[np.argsort(digit_coords[:, 0])]
                num_to_mask = int(len(sorted_by_y) * mask_percentage / 100)
                mask_coords = sorted_by_y[:num_to_mask]
                if len(mask_coords) > 0:
                    mask[mask_coords[:, 0], mask_coords[:, 1]] = 255
        elif mask_type == 'digit_bottom':
            # Mask bottom portion of the digit only
            gray = ConditioningGenerator.to_grayscale(img_np)
            digit_pixels = gray > 30
            digit_coords = np.argwhere(digit_pixels)
            
            if len(digit_coords) > 0:
                sorted_by_y = digit_coords[np.argsort(digit_coords[:, 0])]
                num_to_mask = int(len(sorted_by_y) * mask_percentage / 100)
                mask_coords = sorted_by_y[-num_to_mask:]  # Bottom pixels
                if len(mask_coords) > 0:
                    mask[mask_coords[:, 0], mask_coords[:, 1]] = 255
        elif mask_type == 'digit_square':
            # Mask a square patch centered on the digit
            gray = ConditioningGenerator.to_grayscale(img_np)
            digit_pixels = gray > 30
            digit_coords = np.argwhere(digit_pixels)
            
            if len(digit_coords) > 0:
                # Get bounding box of the digit
                y_min, x_min = digit_coords.min(axis=0)
                y_max, x_max = digit_coords.max(axis=0)
                digit_h = y_max - y_min + 1
                digit_w = x_max - x_min + 1
                
                # Square size based on mask_percentage of digit area
                digit_area = digit_h * digit_w
                target_area = digit_area * mask_percentage / 100
                square_size = max(2, int(np.sqrt(target_area)))
                
                # Random position within digit bounding box
                max_y = max(y_min, y_max - square_size + 1)
                max_x = max(x_min, x_max - square_size + 1)
                
                if max_y > y_min and max_x > x_min:
                    sq_y = np.random.randint(y_min, max_y + 1)
                    sq_x = np.random.randint(x_min, max_x + 1)
                else:
                    sq_y, sq_x = y_min, x_min
                
                # Apply square mask
                mask[sq_y:sq_y+square_size, sq_x:sq_x+square_size] = 255
        elif mask_type == 'digit_multi_square':
            # Mask multiple small square patches on the digit
            gray = ConditioningGenerator.to_grayscale(img_np)
            digit_pixels = gray > 30
            digit_coords = np.argwhere(digit_pixels)
            
            if len(digit_coords) > 0:
                # Get bounding box
                y_min, x_min = digit_coords.min(axis=0)
                y_max, x_max = digit_coords.max(axis=0)
                digit_h = y_max - y_min + 1
                digit_w = x_max - x_min + 1
                
                # Calculate number and size of squares
                digit_area = digit_h * digit_w
                target_area = digit_area * mask_percentage / 100
                num_squares = np.random.randint(2, 5)  # 2-4 squares
                square_size = max(2, int(np.sqrt(target_area / num_squares)))
                
                for _ in range(num_squares):
                    max_y = max(y_min, y_max - square_size + 1)
                    max_x = max(x_min, x_max - square_size + 1)
                    
                    if max_y > y_min and max_x > x_min:
                        sq_y = np.random.randint(y_min, max_y + 1)
                        sq_x = np.random.randint(x_min, max_x + 1)
                        mask[sq_y:sq_y+square_size, sq_x:sq_x+square_size] = 255
        elif mask_type == 'digit_horizontal_band':
            # Mask a horizontal band across the digit
            gray = ConditioningGenerator.to_grayscale(img_np)
            digit_pixels = gray > 30
            digit_coords = np.argwhere(digit_pixels)
            
            if len(digit_coords) > 0:
                y_min, x_min = digit_coords.min(axis=0)
                y_max, x_max = digit_coords.max(axis=0)
                digit_h = y_max - y_min + 1
                
                # Band height based on percentage
                band_h = max(2, int(digit_h * mask_percentage / 100))
                
                # Random y position for band
                if y_max - band_h > y_min:
                    band_y = np.random.randint(y_min, y_max - band_h + 1)
                else:
                    band_y = y_min
                
                # Full width band
                mask[band_y:band_y+band_h, x_min:x_max+1] = 255
        elif mask_type == 'digit_vertical_band':
            # Mask a vertical band across the digit
            gray = ConditioningGenerator.to_grayscale(img_np)
            digit_pixels = gray > 30
            digit_coords = np.argwhere(digit_pixels)
            
            if len(digit_coords) > 0:
                y_min, x_min = digit_coords.min(axis=0)
                y_max, x_max = digit_coords.max(axis=0)
                digit_w = x_max - x_min + 1
                
                # Band width based on percentage
                band_w = max(2, int(digit_w * mask_percentage / 100))
                
                # Random x position for band
                if x_max - band_w > x_min:
                    band_x = np.random.randint(x_min, x_max - band_w + 1)
                else:
                    band_x = x_min
                
                # Full height band
                mask[y_min:y_max+1, band_x:band_x+band_w] = 255
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
        
        # Masked image: keep unmasked regions, zero out masked regions
        masked_img = img_np.copy()
        if len(img_np.shape) == 3:
            masked_img[mask > 0] = 0
        else:
            masked_img[mask > 0] = 0
        
        return masked_img, mask
    
    @staticmethod
    def get_stroke_thickness(img_np):
        """
        Estimate average stroke thickness using distance transform
        Returns: scalar value representing average stroke width
        """
        gray = ConditioningGenerator.to_grayscale(img_np)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        if np.sum(binary) == 0:
            return 0.0
        
        # Distance transform gives distance to nearest background pixel
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Average distance at foreground pixels = approximate stroke half-width
        foreground_pixels = binary > 0
        if np.sum(foreground_pixels) > 0:
            avg_dist = np.mean(dist[foreground_pixels])
            stroke_thickness = avg_dist * 2  # Full width
        else:
            stroke_thickness = 0.0
        
        return stroke_thickness
    
    @staticmethod
    def get_center_scale(img_np):
        """
        Compute center of mass and scale (bounding box area)
        Returns: vector [center_x, center_y, scale] normalized to [0, 1]
        Works for any image size
        """
        gray = ConditioningGenerator.to_grayscale(img_np)
        H, W = gray.shape
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Center of mass
        moments = cv2.moments(binary)
        if moments['m00'] > 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
        else:
            cx, cy = W / 2.0, H / 2.0  # Center of image
        
        # Normalize center to [0, 1]
        cx_norm = cx / float(W)
        cy_norm = cy / float(H)
        
        # Bounding box for scale
        coords = np.argwhere(binary > 0)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bbox_area = (x_max - x_min + 1) * (y_max - y_min + 1)
            scale = bbox_area / float(W * H)  # Normalize by total area
        else:
            scale = 0.0
        
        return np.array([cx_norm, cy_norm, scale], dtype=np.float32)
    
    @staticmethod
    def get_bounding_box(img_np, threshold=30):
        """
        Get bounding box of the main object (for SVHN digit detection)
        Returns: vector [x_min, y_min, x_max, y_max] normalized to [0, 1]
        """
        gray = ConditioningGenerator.to_grayscale(img_np)
        H, W = gray.shape
        
        # Threshold to get foreground
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        coords = np.argwhere(binary > 0)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
        else:
            x_min, y_min, x_max, y_max = 0, 0, W, H
        
        # Normalize to [0, 1]
        bbox = np.array([
            x_min / float(W),
            y_min / float(H),
            x_max / float(W),
            y_max / float(H)
        ], dtype=np.float32)
        
        return bbox
    
    @staticmethod
    def get_segmentation_mask(img_np, num_clusters=5):
        """
        Generate color-based segmentation mask (for CLEVR-like images)
        Uses K-means clustering on colors
        Returns: segmentation map with values 0 to num_clusters-1
        """
        if len(img_np.shape) == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        
        H, W, C = img_np.shape
        pixels = img_np.reshape(-1, C).astype(np.float32)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(pixels, num_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        
        segmentation = labels.reshape(H, W).astype(np.uint8)
        
        # Normalize to [0, 255]
        segmentation = (segmentation * (255 // (num_clusters - 1))).astype(np.uint8)
        
        return segmentation
    
    @staticmethod
    def get_object_positions(img_np, max_objects=10):
        """
        Detect object positions using contours (for CLEVR-like images)
        Returns: vector of shape (max_objects * 2,) with (x, y) positions
        Positions normalized to [0, 1], padded with -1 for missing objects
        """
        gray = ConditioningGenerator.to_grayscale(img_np)
        H, W = gray.shape
        
        # Edge detection and contour finding
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get centroids of largest contours
        positions = []
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_objects]
        
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00'] / float(W)
                cy = M['m01'] / M['m00'] / float(H)
                positions.extend([cx, cy])
        
        # Pad to fixed size
        while len(positions) < max_objects * 2:
            positions.extend([-1.0, -1.0])  # Padding value
        
        return np.array(positions[:max_objects * 2], dtype=np.float32)


# ============================================================================
# Dataset Classes
# ============================================================================

class BaseConditioningDataset(Dataset):
    """
    Base class for conditioning datasets
    Handles common conditioning generation logic
    """
    
    def __init__(self, conditioning='edge', mask_type='random_rect', mask_percentage=50,
                 image_size=28, in_channels=1):
        self.conditioning = conditioning
        self.mask_type = mask_type
        self.mask_percentage = mask_percentage
        self.cond_gen = ConditioningGenerator()
        self.image_size = image_size
        self.in_channels = in_channels
    
    def _generate_conditioning(self, img_np):
        """Generate conditioning signal based on type"""
        
        if self.conditioning == 'edge':
            cond = self.cond_gen.get_edge_map(img_np)
            cond_tensor = torch.from_numpy(cond).float().unsqueeze(0) / 127.5 - 1.0
            
        elif self.conditioning == 'sobel':
            cond = self.cond_gen.get_sobel_edges(img_np)
            cond_tensor = torch.from_numpy(cond).float().unsqueeze(0) / 127.5 - 1.0
            
        elif self.conditioning == 'skeleton':
            cond = self.cond_gen.get_skeleton(img_np)
            cond_tensor = torch.from_numpy(cond).float().unsqueeze(0) / 127.5 - 1.0
            
        elif self.conditioning == 'inpainting':
            masked_img, mask = self.cond_gen.get_partial_mask(img_np, self.mask_type, self.mask_percentage)
            if len(masked_img.shape) == 3:  # RGB
                # Convert masked image to tensor (C, H, W)
                masked_tensor = torch.from_numpy(masked_img).float().permute(2, 0, 1) / 127.5 - 1.0
                mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 127.5 - 1.0
                cond_tensor = torch.cat([masked_tensor, mask_tensor], dim=0)  # (C+1, H, W)
            else:  # Grayscale
                masked_tensor = torch.from_numpy(masked_img).float().unsqueeze(0) / 127.5 - 1.0
                mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 127.5 - 1.0
                cond_tensor = torch.cat([masked_tensor, mask_tensor], dim=0)  # (2, H, W)
            
        elif self.conditioning == 'stroke_thickness':
            thickness = self.cond_gen.get_stroke_thickness(img_np)
            thickness_norm = (thickness - 3.0) / 3.0  # Center around 0
            cond_tensor = torch.tensor([thickness_norm], dtype=torch.float32)
            
        elif self.conditioning == 'center_scale':
            center_scale = self.cond_gen.get_center_scale(img_np)
            cond_tensor = torch.from_numpy(center_scale * 2 - 1).float()  # (3,)
            
        elif self.conditioning == 'color_histogram':
            hist = self.cond_gen.get_color_histogram(img_np, bins=16)
            cond_tensor = torch.from_numpy(hist * 2 - 1).float()  # Normalize to [-1, 1]
            
        elif self.conditioning == 'bounding_box':
            bbox = self.cond_gen.get_bounding_box(img_np)
            cond_tensor = torch.from_numpy(bbox * 2 - 1).float()  # (4,)
            
        elif self.conditioning == 'segmentation':
            seg = self.cond_gen.get_segmentation_mask(img_np)
            cond_tensor = torch.from_numpy(seg).float().unsqueeze(0) / 127.5 - 1.0
            
        elif self.conditioning == 'object_positions':
            positions = self.cond_gen.get_object_positions(img_np)
            cond_tensor = torch.from_numpy(positions).float()  # Already normalized
            
        else:
            raise ValueError(f"Unknown conditioning type: {self.conditioning}")
        
        return cond_tensor


class MNISTConditioningDataset(BaseConditioningDataset):
    """
    MNIST dataset with multiple conditioning types
    
    Supported conditioning types:
    - 'edge': Canny edge map (B, 1, 28, 28)
    - 'sobel': Sobel edge map (B, 1, 28, 28)
    - 'skeleton': Morphological skeleton (B, 1, 28, 28)
    - 'inpainting': Masked image + mask (B, 2, 28, 28)
    - 'stroke_thickness': Scalar stroke width (B, 1)
    - 'center_scale': Center of mass + scale (B, 3)
    """
    
    def __init__(self, train=True, conditioning='edge', mask_type='random_rect', mask_percentage=50):
        super().__init__(conditioning=conditioning, mask_type=mask_type, mask_percentage=mask_percentage, image_size=28, in_channels=1)
        self.mnist = torchvision.datasets.MNIST(
            root='./data', 
            train=train, 
            download=True
        )
        
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        img_np = np.array(img)
        
        # Convert to tensor [1, 28, 28], normalize to [-1, 1]
        img_tensor = transforms.ToTensor()(img)
        img_tensor = (img_tensor - 0.5) / 0.5
        
        # Generate conditioning
        cond_tensor = self._generate_conditioning(img_np)
        
        return {
            'image': img_tensor,
            'condition': cond_tensor,
            'label': label
        }


class SVHNConditioningDataset(BaseConditioningDataset):
    """
    SVHN (Street View House Numbers) dataset with multiple conditioning types
    
    SVHN is RGB 32x32, contains real-world digit images from house numbers
    
    Supported conditioning types:
    - 'edge': Canny edge map (B, 1, 32, 32)
    - 'sobel': Sobel edge map (B, 1, 32, 32)
    - 'inpainting': Masked image + mask (B, 4, 32, 32) - RGB + mask
    - 'color_histogram': Color histogram vector (B, 48) - 16 bins * 3 channels
    - 'bounding_box': Bounding box coords (B, 4)
    - 'center_scale': Center of mass + scale (B, 3)
    """
    
    def __init__(self, train=True, conditioning='edge', mask_type='random_rect', mask_percentage=50):
        super().__init__(conditioning=conditioning, mask_type=mask_type, mask_percentage=mask_percentage, image_size=32, in_channels=3)
        split = 'train' if train else 'test'
        self.svhn = torchvision.datasets.SVHN(
            root='./data',
            split=split,
            download=True
        )
        
    def __len__(self):
        return len(self.svhn)
    
    def __getitem__(self, idx):
        img, label = self.svhn[idx]
        
        # SVHN images are PIL images (32x32 RGB)
        img_np = np.array(img)  # (32, 32, 3)
        
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


class CLEVRConditioningDataset(BaseConditioningDataset):
    """
    CLEVR (Compositional Language and Elementary Visual Reasoning) dataset
    
    CLEVR contains synthetic 3D rendered scenes with simple objects
    Great for testing compositional generation and scene understanding
    
    Supported conditioning types:
    - 'edge': Canny edge map (B, 1, 64, 64)
    - 'sobel': Sobel edge map (B, 1, 64, 64)
    - 'segmentation': Color-based segmentation (B, 1, 64, 64)
    - 'object_positions': Object centroid positions (B, 20) - max 10 objects * 2 coords
    - 'inpainting': Masked image + mask (B, 4, 64, 64)
    - 'color_histogram': Color histogram (B, 48)
    
    Note: Requires manual download from https://cs.stanford.edu/people/jcjohns/clevr/
    Or use the mini version from HuggingFace datasets
    """
    
    def __init__(self, train=True, conditioning='edge', mask_type='random_rect', 
                 data_root='./data/clevr', image_size=64, use_mini=True):
        super().__init__(conditioning=conditioning, mask_type=mask_type, 
                        image_size=image_size, in_channels=3)
        self.data_root = data_root
        self.target_size = image_size
        self.use_mini = use_mini
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        # Try to load from HuggingFace datasets (CLEVR mini) or local files
        self.images = []
        self.labels = []
        self._load_data(train)
        
    def _load_data(self, train):
        """Load CLEVR data from various sources"""
        
        # Option 1: Try HuggingFace datasets
        try:
            from datasets import load_dataset
            split = 'train' if train else 'validation'
            dataset = load_dataset('vaishaal/clevr-math', split=split)
            self.hf_dataset = dataset
            self.use_hf = True
            print(f"Loaded CLEVR from HuggingFace ({len(dataset)} samples)")
            return
        except Exception as e:
            self.use_hf = False
            print(f"HuggingFace CLEVR not available: {e}")
        
        # Option 2: Load from local directory
        if os.path.exists(self.data_root):
            split_dir = 'train' if train else 'val'
            image_dir = os.path.join(self.data_root, 'images', split_dir)
            
            if os.path.exists(image_dir):
                self.image_files = sorted([
                    os.path.join(image_dir, f) 
                    for f in os.listdir(image_dir) 
                    if f.endswith(('.png', '.jpg'))
                ])
                print(f"Loaded CLEVR from local ({len(self.image_files)} images)")
                return
        
        # Option 3: Generate synthetic CLEVR-like images
        print("No CLEVR data found. Generating synthetic CLEVR-like images...")
        self._generate_synthetic_data(train)
    
    def _generate_synthetic_data(self, train, num_samples=10000):
        """Generate simple CLEVR-like synthetic images for testing"""
        num_samples = num_samples if train else 2000
        self.synthetic_mode = True
        self.num_samples = num_samples
        
    def _generate_single_image(self):
        """Generate a single synthetic CLEVR-like image"""
        img = np.ones((self.target_size, self.target_size, 3), dtype=np.uint8) * 200  # Gray background
        
        num_objects = np.random.randint(1, 6)
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        
        for _ in range(num_objects):
            color = colors[np.random.randint(0, len(colors))]
            shape = np.random.choice(['circle', 'rectangle'])
            size = np.random.randint(5, 15)
            cx = np.random.randint(size, self.target_size - size)
            cy = np.random.randint(size, self.target_size - size)
            
            if shape == 'circle':
                cv2.circle(img, (cx, cy), size, color, -1)
            else:
                cv2.rectangle(img, (cx - size, cy - size), (cx + size, cy + size), color, -1)
        
        return img
        
    def __len__(self):
        if hasattr(self, 'use_hf') and self.use_hf:
            return len(self.hf_dataset)
        elif hasattr(self, 'image_files'):
            return len(self.image_files)
        elif hasattr(self, 'synthetic_mode'):
            return self.num_samples
        return 0
    
    def __getitem__(self, idx):
        # Load image based on data source
        if hasattr(self, 'use_hf') and self.use_hf:
            item = self.hf_dataset[idx]
            img = item['image']
            if hasattr(img, 'convert'):
                img = img.convert('RGB')
            img = img.resize((self.target_size, self.target_size))
            img_np = np.array(img)
            label = 0  # CLEVR doesn't have single labels
            
        elif hasattr(self, 'image_files'):
            img = Image.open(self.image_files[idx]).convert('RGB')
            img = img.resize((self.target_size, self.target_size))
            img_np = np.array(img)
            label = 0
            
        elif hasattr(self, 'synthetic_mode'):
            img_np = self._generate_single_image()
            label = 0
        else:
            raise RuntimeError("No data source available")
        
        # Convert to tensor [3, H, W], normalize to [-1, 1]
        img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 127.5 - 1.0
        
        # Generate conditioning
        cond_tensor = self._generate_conditioning(img_np)
        
        return {
            'image': img_tensor,
            'condition': cond_tensor,
            'label': label
        }


# Backwards compatibility alias
MNISTEdgeDataset = MNISTConditioningDataset


def get_dataset_class(dataset_name):
    """Get dataset class by name"""
    datasets = {
        'mnist': MNISTConditioningDataset,
        'svhn': SVHNConditioningDataset,
        'clevr': CLEVRConditioningDataset,
    }
    if dataset_name.lower() not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(datasets.keys())}")
    return datasets[dataset_name.lower()]


def get_dataloader(batch_size=64, train=True, conditioning='edge', mask_type='random_rect', 
                   mask_percentage=50, num_workers=0, dataset_name='mnist', **kwargs):
    """
    Get dataloader for specified dataset with conditioning
    
    Args:
        batch_size: Batch size
        train: Whether to use training set
        conditioning: Type of conditioning (dataset-specific)
        mask_type: For inpainting, type of mask. Options:
                   - 'top_half', 'bottom_half', 'left_half', 'right_half': Simple splits
                   - 'random_rect': Random rectangle mask
                   - 'center': Mask center region
                   - 'digit_percentage': Mask X% of digit pixels randomly
                   - 'digit_contiguous': Mask X% of digit pixels in contiguous region
                   - 'digit_top': Mask top X% of digit pixels
                   - 'digit_bottom': Mask bottom X% of digit pixels
        mask_percentage: For digit-based masks, what percentage to hide (0-100)
        num_workers: Number of data loading workers (0 for Windows compatibility)
        dataset_name: 'mnist', 'svhn', or 'clevr'
        **kwargs: Additional arguments for specific datasets (e.g., image_size for CLEVR)
    """
    DatasetClass = get_dataset_class(dataset_name)
    
    # Create dataset with appropriate arguments
    if dataset_name.lower() == 'clevr':
        dataset = DatasetClass(
            train=train, 
            conditioning=conditioning,
            mask_type=mask_type,
            mask_percentage=mask_percentage,
            **kwargs
        )
    else:
        dataset = DatasetClass(
            train=train, 
            conditioning=conditioning,
            mask_type=mask_type,
            mask_percentage=mask_percentage
        )
    
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )


def get_conditioning_channels(conditioning_type, dataset_name='mnist'):
    """
    Return the number of channels for each conditioning type
    
    Args:
        conditioning_type: Type of conditioning
        dataset_name: Dataset name (affects inpainting channels for RGB datasets)
    """
    # Get dataset configuration
    config = DATASET_CONFIG.get(dataset_name.lower(), DATASET_CONFIG['mnist'])
    is_rgb = config['channels'] == 3
    
    channels = {
        'edge': 1,
        'sobel': 1,
        'skeleton': 1,
        'inpainting': 4 if is_rgb else 2,  # RGB + mask or grayscale + mask
        'stroke_thickness': 1,  # scalar (needs projection)
        'center_scale': 3,  # cx, cy, scale (needs projection)
        'color_histogram': 48 if is_rgb else 16,  # 16 bins * 3 channels for RGB
        'bounding_box': 4,  # x_min, y_min, x_max, y_max
        'segmentation': 1,  # single channel segmentation map
        'object_positions': 20,  # max 10 objects * 2 coords
    }
    return channels.get(conditioning_type, 1)


def is_vector_conditioning(conditioning_type):
    """Check if conditioning type produces vector (not image) output"""
    return conditioning_type in [
        'stroke_thickness', 
        'center_scale', 
        'color_histogram',
        'bounding_box',
        'object_positions'
    ]


def get_dataset_config(dataset_name):
    """Get configuration for a dataset"""
    if dataset_name.lower() not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIG.keys())}")
    return DATASET_CONFIG[dataset_name.lower()]


def list_available_conditionings(dataset_name='mnist'):
    """List available conditioning types for a dataset"""
    config = get_dataset_config(dataset_name)
    return config['conditioning_types']


def print_dataset_info():
    """Print information about all available datasets and conditioning types"""
    print("\n" + "=" * 70)
    print("AVAILABLE DATASETS AND CONDITIONING TYPES")
    print("=" * 70)
    
    for name, config in DATASET_CONFIG.items():
        print(f"\n📊 {name.upper()}")
        print(f"   Image Size: {config['image_size']}x{config['image_size']}")
        print(f"   Channels: {config['channels']} ({'RGB' if config['channels'] == 3 else 'Grayscale'})")
        print(f"   Classes: {config['num_classes'] if config['num_classes'] else 'N/A'}")
        print(f"   Conditioning types:")
        for cond in config['conditioning_types']:
            ch = get_conditioning_channels(cond, name)
            is_vec = is_vector_conditioning(cond)
            type_str = "vector" if is_vec else "spatial"
            print(f"      - {cond}: {ch} channels ({type_str})")
    
    print("\n" + "=" * 70)