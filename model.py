import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# LoRA Layers
# ============================================================================

class LoRALinear(nn.Module):
    """Low-Rank Adaptation for Linear layers"""
    def __init__(self, in_features, out_features, rank=4, alpha=None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scaling = self.alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
    def forward(self, x):
        # x: [B, in_features] or [B, ..., in_features]
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRAConv2d(nn.Module):
    """Low-Rank Adaptation for Conv2d layers"""
    def __init__(self, in_channels, out_channels, kernel_size=1, rank=4, alpha=None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scaling = self.alpha / rank
        
        # Implement LoRA for convolutions using 1x1 convs (channel mixing)
        self.lora_A = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        # x: [B, C, H, W]
        return self.lora_B(self.lora_A(x)) * self.scaling


class LoRALayer(nn.Module):
    """Generic Low-Rank Adaptation layer (backwards compatible)"""
    def __init__(self, in_features, out_features, rank=4, alpha=None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scaling = self.alpha / rank
        
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
    def forward(self, x):
        # x: [B, in_features] or [B, C, H, W]
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
            delta = (x_flat @ self.lora_A @ self.lora_B) * self.scaling
            delta = delta.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        else:
            delta = (x @ self.lora_A @ self.lora_B) * self.scaling
        return delta

# ============================================================================
# Basic UNet Blocks
# ============================================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class VectorConditioningEncoder(nn.Module):
    """
    Encoder for vector conditioning (stroke thickness, center/scale, histograms)
    Projects low-dim vector to spatial feature maps
    """
    def __init__(self, input_dim, output_channels, target_size=28):
        super().__init__()
        self.target_size = target_size
        self.output_channels = output_channels
        
        # MLP to project vector to spatial features
        # Scale hidden dims based on target size
        hidden_dim = min(256, target_size * target_size // 4)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_channels * target_size * target_size)
        )
        
    def forward(self, x):
        # x: (B, input_dim)
        B = x.shape[0]
        features = self.mlp(x)  # (B, C * H * W)
        return features.reshape(B, self.output_channels, self.target_size, self.target_size)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.conv1(F.silu(x))
        h = self.norm1(h)
        
        # Add time embedding
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        
        h = self.conv2(F.silu(h))
        h = self.norm2(h)
        
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        k = k.reshape(B, C, H * W)  # [B, C, HW]
        v = v.reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        
        attn = torch.softmax(q @ k / (C ** 0.5), dim=-1)  # [B, HW, HW]
        out = attn @ v  # [B, HW, C]
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        
        return x + self.proj(out)

# ============================================================================
# Simple UNet (Base Denoiser)
# ============================================================================

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, model_channels=64, time_dim=256, 
                 out_channels=None, image_size=28):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.model_channels = model_channels
        self.time_dim = time_dim
        self.image_size = image_size
        
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        # Encoder
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        self.down1 = nn.ModuleList([
            ResBlock(model_channels, model_channels, time_dim),
            ResBlock(model_channels, model_channels, time_dim)
        ])
        self.downsample1 = nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)
        
        self.down2 = nn.ModuleList([
            ResBlock(model_channels, model_channels * 2, time_dim),
            ResBlock(model_channels * 2, model_channels * 2, time_dim)
        ])
        self.downsample2 = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)
        
        # Middle
        self.mid = nn.ModuleList([
            ResBlock(model_channels * 2, model_channels * 2, time_dim),
            AttentionBlock(model_channels * 2),
            ResBlock(model_channels * 2, model_channels * 2, time_dim)
        ])
        
        # Decoder
        self.upsample2 = nn.ConvTranspose2d(model_channels * 2, model_channels * 2, 4, stride=2, padding=1)
        self.up2 = nn.ModuleList([
            ResBlock(model_channels * 4, model_channels * 2, time_dim),
            ResBlock(model_channels * 2, model_channels, time_dim)
        ])
        
        self.upsample1 = nn.ConvTranspose2d(model_channels, model_channels, 4, stride=2, padding=1)
        self.up1 = nn.ModuleList([
            ResBlock(model_channels * 2, model_channels, time_dim),
            ResBlock(model_channels, model_channels, time_dim)
        ])
        
        self.conv_out = nn.Conv2d(model_channels, self.out_channels, 3, padding=1)
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Initial conv
        h = self.conv_in(x)
        
        # Encoder
        h1 = h
        for block in self.down1:
            h1 = block(h1, t_emb)
        h1_skip = h1
        h1 = self.downsample1(h1)
        
        h2 = h1
        for block in self.down2:
            h2 = block(h2, t_emb)
        h2_skip = h2
        h2 = self.downsample2(h2)
        
        # Middle
        h = h2
        for block in self.mid:
            if isinstance(block, AttentionBlock):
                h = block(h)
            else:
                h = block(h, t_emb)
        
        # Decoder
        h = self.upsample2(h)
        h = torch.cat([h, h2_skip], dim=1)
        for block in self.up2:
            h = block(h, t_emb)
        
        h = self.upsample1(h)
        h = torch.cat([h, h1_skip], dim=1)
        for block in self.up1:
            h = block(h, t_emb)
        
        return self.conv_out(h)


# ============================================================================
# Input Concatenation Baseline
# ============================================================================

class ConcatConditionUNet(nn.Module):
    """
    Baseline: Simple input concatenation conditioning.
    Concatenates conditioning to input channels.
    """
    def __init__(self, conditioning_channels=1, model_channels=64, time_dim=256,
                 in_channels=1, image_size=28):
        super().__init__()
        # UNet with extra input channels for condition
        self.unet = SimpleUNet(
            in_channels=in_channels + conditioning_channels,  # Image + condition
            model_channels=model_channels,
            time_dim=time_dim,
            out_channels=in_channels,  # Output same as input image channels
            image_size=image_size
        )
        self.conditioning_channels = conditioning_channels
        self.in_channels = in_channels
        
    def forward(self, x, t, condition):
        # Concatenate condition to input
        x_cond = torch.cat([x, condition], dim=1)
        return self.unet(x_cond, t)

# ============================================================================
# ControlNet (Full Copy)
# ============================================================================

class ControlNet(nn.Module):
    """
    Full ControlNet: copy of UNet encoder + zero convs for control injection.
    This is the parameter-heavy baseline.
    """
    def __init__(self, base_unet, conditioning_channels=1):
        super().__init__()
        
        model_channels = base_unet.model_channels
        time_dim = base_unet.time_dim
        
        # Own time embedding (shared structure, independent weights)
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        # Condition input conv
        self.conv_in = nn.Conv2d(conditioning_channels, model_channels, 3, padding=1)
        
        # Copy encoder blocks (independent weights)
        self.down1 = nn.ModuleList([
            ResBlock(model_channels, model_channels, time_dim),
            ResBlock(model_channels, model_channels, time_dim)
        ])
        self.downsample1 = nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)
        
        self.down2 = nn.ModuleList([
            ResBlock(model_channels, model_channels * 2, time_dim),
            ResBlock(model_channels * 2, model_channels * 2, time_dim)
        ])
        self.downsample2 = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)
        
        self.mid = nn.ModuleList([
            ResBlock(model_channels * 2, model_channels * 2, time_dim),
            AttentionBlock(model_channels * 2),
            ResBlock(model_channels * 2, model_channels * 2, time_dim)
        ])
        
        # Zero convolutions for control injection
        self.zero_convs = nn.ModuleList([
            nn.Conv2d(model_channels, model_channels, 1),      # After down1
            nn.Conv2d(model_channels, model_channels, 1),      # After downsample1
            nn.Conv2d(model_channels * 2, model_channels * 2, 1),  # After down2
            nn.Conv2d(model_channels * 2, model_channels * 2, 1),  # After downsample2
            nn.Conv2d(model_channels * 2, model_channels * 2, 1),  # After mid
        ])
        
        # Initialize zero convs to zero
        for zc in self.zero_convs:
            nn.init.zeros_(zc.weight)
            nn.init.zeros_(zc.bias)
    
    def forward(self, condition, t):
        t_emb = self.time_embed(t)
        
        controls = []
        h = self.conv_in(condition)
        
        # Down1
        for block in self.down1:
            h = block(h, t_emb)
        controls.append(self.zero_convs[0](h))
        h = self.downsample1(h)
        controls.append(self.zero_convs[1](h))
        
        # Down2
        for block in self.down2:
            h = block(h, t_emb)
        controls.append(self.zero_convs[2](h))
        h = self.downsample2(h)
        controls.append(self.zero_convs[3](h))
        
        # Mid
        for block in self.mid:
            if isinstance(block, AttentionBlock):
                h = block(h)
            else:
                h = block(h, t_emb)
        controls.append(self.zero_convs[4](h))
        
        return controls

# ============================================================================
# LoRA-ControlNet (Parameter-Efficient Conditioning)
# ============================================================================

class LoRAControlNet(nn.Module):
    """
    ControlNet using LoRA instead of full encoder copy.
    
    This tests the hypothesis that conditioning signals are low-rank
    relative to the denoiser, so LoRA adapters are sufficient.
    
    Args:
        base_unet: The base UNet model
        conditioning_channels: Number of input channels for condition
        rank: LoRA rank (configurable for ablation: 1, 2, 4, 8, 16)
        inject_layer: Where to inject controls ('early', 'mid', 'all')
        alpha: LoRA scaling factor (default: rank)
    """
    def __init__(self, base_unet, conditioning_channels=1, rank=4, inject_layer='mid', alpha=None):
        super().__init__()
        
        self.inject_layer = inject_layer
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        
        model_channels = base_unet.model_channels
        
        # Lightweight condition encoder (much smaller than ControlNet)
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(conditioning_channels, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 28 -> 14
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 14 -> 7
            nn.SiLU(),
        )
        
        # LoRA injections based on strategy
        if inject_layer == 'early':
            # Only inject in early layers
            self.proj_early = nn.Conv2d(64, model_channels, 1)
            self.lora_early = LoRAConv2d(model_channels, model_channels, rank=rank, alpha=self.alpha)
            
        elif inject_layer == 'mid':
            # Only inject in middle block
            self.lora_mid = LoRAConv2d(128, model_channels * 2, rank=rank, alpha=self.alpha)
            
        elif inject_layer == 'all':
            # Inject at multiple scales
            self.proj_early = nn.Conv2d(64, model_channels, 1)
            self.lora_early = LoRAConv2d(model_channels, model_channels, rank=rank, alpha=self.alpha)
            self.lora_down = LoRAConv2d(128, model_channels * 2, rank=rank, alpha=self.alpha)
            self.lora_mid = LoRAConv2d(128, model_channels * 2, rank=rank, alpha=self.alpha)
            
        else:
            raise ValueError(f"Unknown inject_layer: {inject_layer}")
    
    def forward(self, condition, t):
        """
        Returns list of control signals to add to UNet features.
        Format: [down1_ctrl, downsample1_ctrl, down2_ctrl, downsample2_ctrl, mid_ctrl]
        None values indicate no control at that position.
        """
        # Encode condition through lightweight network
        cond_feat = self.condition_encoder(condition)  # [B, 128, 7, 7]
        
        controls = [None, None, None, None, None]
        
        if self.inject_layer == 'early':
            # Upsample condition features to early resolution
            cond_early = F.interpolate(cond_feat, size=(28, 28), mode='bilinear', align_corners=False)
            cond_early = self.proj_early(cond_early[:, :64])  # Take first 64 channels
            delta = self.lora_early(cond_early)
            controls[0] = delta  # Inject after down1
            
        elif self.inject_layer == 'mid':
            # Apply LoRA modulation in middle
            delta = self.lora_mid(cond_feat)
            controls[4] = delta  # Inject after mid block
            
        elif self.inject_layer == 'all':
            # Multi-scale injection
            # Early: upsample to 28x28
            cond_28 = F.interpolate(cond_feat, size=(28, 28), mode='bilinear', align_corners=False)
            cond_28 = self.proj_early(cond_28[:, :64])
            controls[0] = self.lora_early(cond_28)
            
            # Down: at 14x14
            cond_14 = F.interpolate(cond_feat, size=(14, 14), mode='bilinear', align_corners=False)
            controls[2] = self.lora_down(cond_14)
            
            # Mid: at 7x7
            controls[4] = self.lora_mid(cond_feat)
        
        return controls


# ============================================================================
# Vector Conditioning LoRA (for stroke thickness, center/scale)
# ============================================================================

class VectorLoRAControlNet(nn.Module):
    """
    LoRA-based conditioning for vector inputs (stroke thickness, center/scale, histograms).
    Projects low-dim vector conditioning to modulate UNet features via LoRA.
    """
    def __init__(self, base_unet, input_dim=1, rank=4, inject_layer='mid', alpha=None, target_size=28):
        super().__init__()
        
        self.inject_layer = inject_layer
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.target_size = target_size
        
        model_channels = base_unet.model_channels
        
        # Compute spatial sizes at different layers
        # After 2 downsamples: size // 4
        mid_size = max(target_size // 4, 4)  # Ensure at least 4x4
        
        # MLP to process vector conditioning - scale capacity based on input dim
        hidden_dim = min(256, max(64, input_dim * 8))
        self.vector_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
        )
        
        # Spatial broadcast layers
        if inject_layer in ['mid', 'all']:
            self.proj_mid = nn.Linear(hidden_dim * 2, model_channels * 2 * mid_size * mid_size)
            self.mid_size = mid_size
            self.lora_mid = LoRAConv2d(model_channels * 2, model_channels * 2, rank=rank, alpha=self.alpha)
            
        if inject_layer == 'all':
            self.proj_early = nn.Linear(hidden_dim * 2, model_channels * target_size * target_size)
            self.lora_early = LoRAConv2d(model_channels, model_channels, rank=rank, alpha=self.alpha)
    
    def forward(self, condition, t):
        """
        condition: (B, input_dim) vector conditioning
        """
        B = condition.shape[0]
        
        # Encode vector
        vec_feat = self.vector_encoder(condition)  # (B, hidden_dim * 2)
        
        controls = [None, None, None, None, None]
        
        if self.inject_layer == 'mid' or self.inject_layer == 'all':
            # Project to spatial and apply LoRA
            mid_feat = self.proj_mid(vec_feat).reshape(B, -1, self.mid_size, self.mid_size)
            controls[4] = self.lora_mid(mid_feat)
            
        if self.inject_layer == 'all':
            early_feat = self.proj_early(vec_feat).reshape(B, -1, self.target_size, self.target_size)
            controls[0] = self.lora_early(early_feat)
        
        return controls

# ============================================================================
# Conditional UNet (combines base UNet + ControlNet/LoRA)
# ============================================================================

class ConditionalUNet(nn.Module):
    """
    Combines base UNet with a control module (ControlNet or LoRA).
    The control module injects residuals at multiple layers.
    """
    def __init__(self, base_unet, control_net):
        super().__init__()
        self.base_unet = base_unet
        self.control_net = control_net
    
    def forward(self, x, t, condition):
        # Get control signals
        controls = self.control_net(condition, t)
        
        # Modified UNet forward with control injection
        t_emb = self.base_unet.time_embed(t)
        h = self.base_unet.conv_in(x)
        
        # Encoder with control injection
        h1 = h
        for block in self.base_unet.down1:
            h1 = block(h1, t_emb)
        if controls[0] is not None:
            h1 = h1 + controls[0]
        h1_skip = h1
        h1 = self.base_unet.downsample1(h1)
        if controls[1] is not None:
            h1 = h1 + controls[1]
        
        h2 = h1
        for block in self.base_unet.down2:
            h2 = block(h2, t_emb)
        if controls[2] is not None:
            h2 = h2 + controls[2]
        h2_skip = h2
        h2 = self.base_unet.downsample2(h2)
        if controls[3] is not None:
            h2 = h2 + controls[3]
        
        # Middle with control
        h = h2
        for block in self.base_unet.mid:
            if isinstance(block, AttentionBlock):
                h = block(h)
            else:
                h = block(h, t_emb)
        if controls[4] is not None:
            h = h + controls[4]
        
        # Decoder (no control injection)
        h = self.base_unet.upsample2(h)
        h = torch.cat([h, h2_skip], dim=1)
        for block in self.base_unet.up2:
            h = block(h, t_emb)
        
        h = self.base_unet.upsample1(h)
        h = torch.cat([h, h1_skip], dim=1)
        for block in self.base_unet.up1:
            h = block(h, t_emb)
        
        return self.base_unet.conv_out(h)
    
    def freeze_base(self):
        """Freeze base UNet weights (for LoRA training)"""
        for param in self.base_unet.parameters():
            param.requires_grad = False
    
    def unfreeze_base(self):
        """Unfreeze base UNet weights"""
        for param in self.base_unet.parameters():
            param.requires_grad = True


# ============================================================================
# Model Factory
# ============================================================================

def create_model(model_type='controlnet', rank=4, inject_layer='mid', 
                 conditioning_channels=1, conditioning_type='edge',
                 freeze_base=False, model_channels=64,
                 image_size=28, in_channels=1):
    """
    Create a diffusion model with specified conditioning mechanism.
    
    Args:
        model_type: 
            - 'baseline': Unconditional UNet
            - 'concat': Input concatenation baseline
            - 'controlnet': Full ControlNet (parameter-heavy)
            - 'lora': LoRA-based conditioning (parameter-efficient)
            - 'lora_vector': LoRA for vector conditioning
        rank: LoRA rank for ablation (1, 2, 4, 8, 16)
        inject_layer: Where to inject controls ('early', 'mid', 'all')
        conditioning_channels: Number of condition input channels
        conditioning_type: Type of conditioning ('edge', 'inpainting', etc.)
        freeze_base: Whether to freeze base UNet (for LoRA)
        model_channels: Base channels for UNet
        image_size: Size of input images (28 for MNIST, 32 for SVHN, 64 for CLEVR)
        in_channels: Number of input image channels (1 for grayscale, 3 for RGB)
    
    Returns:
        Configured model ready for training
    """
    # Scale model channels based on image size for better capacity
    if image_size >= 64:
        model_channels = max(model_channels, 128)
    
    base_unet = SimpleUNet(in_channels=in_channels, model_channels=model_channels, 
                          out_channels=in_channels, image_size=image_size)
    
    if model_type == 'baseline':
        return base_unet
    
    elif model_type == 'concat':
        return ConcatConditionUNet(
            conditioning_channels=conditioning_channels,
            model_channels=model_channels,
            in_channels=in_channels,
            image_size=image_size
        )
    
    elif model_type == 'controlnet':
        control_net = ControlNet(base_unet, conditioning_channels=conditioning_channels)
        model = ConditionalUNet(base_unet, control_net)
        return model
    
    elif model_type == 'lora':
        control_net = LoRAControlNet(
            base_unet, 
            conditioning_channels=conditioning_channels,
            rank=rank, 
            inject_layer=inject_layer
        )
        model = ConditionalUNet(base_unet, control_net)
        if freeze_base:
            model.freeze_base()
        return model
    
    elif model_type == 'lora_vector':
        # For vector conditioning (stroke thickness, center/scale, histograms, etc.)
        vector_dims = {
            'stroke_thickness': 1,
            'center_scale': 3,
            'color_histogram': conditioning_channels,  # 48 for RGB, 16 for grayscale
            'bounding_box': 4,
            'object_positions': conditioning_channels,  # 20 for max 10 objects
        }
        vector_dim = vector_dims.get(conditioning_type, conditioning_channels)
        control_net = VectorLoRAControlNet(
            base_unet,
            input_dim=vector_dim,
            rank=rank,
            inject_layer=inject_layer,
            target_size=image_size
        )
        model = ConditionalUNet(base_unet, control_net)
        if freeze_base:
            model.freeze_base()
        return model
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_summary(model):
    """Get a summary of model parameters by component"""
    summary = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                prefix = name.split('.')[0] if '.' in name else name
                summary[prefix] = summary.get(prefix, 0) + params
    return summary
    