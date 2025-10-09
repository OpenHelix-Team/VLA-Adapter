
"""
pc_encoder.py

Implementations of pointcloud encoder in iDP3, which also supports 2D point cloud maps. 

# reference: https://github.com/YanjieZe/Improved-3D-Diffusion-Policy/blob/main/Improved-3D-Diffusion-Policy/diffusion_policy_3d/model/vision_3d
"""
from typing import Tuple, List, Optional, Dict, Union, Type
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prismatic.overwatch import initialize_overwatch
overwatch = initialize_overwatch(__name__)
# ==================== Utility Functions ====================

def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out

def maxpool(x, dim=-1, keepdim=False):
    out = x.max(dim=dim, keepdim=keepdim).values
    return out

def shuffle_point_torch(point_cloud):
    B, N, C = point_cloud.shape
    indices = torch.randperm(N)
    return point_cloud[:, indices]

def pad_point_torch(point_cloud, num_points):
    B, N, C = point_cloud.shape
    device = point_cloud.device
    if num_points > N:
        num_pad = num_points - N
        pad_points = torch.zeros(B, num_pad, C).to(device)
        point_cloud = torch.cat([point_cloud, pad_points], dim=1)
        point_cloud = shuffle_point_torch(point_cloud)
    return point_cloud

def uniform_sampling_torch(point_cloud, num_points):
    B, N, C = point_cloud.shape
    if num_points == N:
        return point_cloud
    if num_points > N:
        return pad_point_torch(point_cloud, num_points)
    
    # random sampling
    indices = torch.randperm(N)[:num_points]
    sampled_points = point_cloud[:, indices]
    return sampled_points

# ==================== 2D Point Cloud utility ====================

def shuffle_map_torch(map_data):
    B, H, W, C = map_data.shape
    # Flatten spatial dimensions
    map_flat = map_data.view(B, H * W, C)
    indices = torch.randperm(H * W)
    map_shuffled = map_flat[:, indices, :]
    # Reshape back
    return map_shuffled.view(B, H, W, C)


def pad_map_torch(map_data, target_size):
    B, H, W, C = map_data.shape
    device = map_data.device
    
    if target_size > H or target_size > W:
        # Create zero-padded map
        padded_map = torch.zeros(B, target_size, target_size, C, device=device)
        # Copy original data to top-left corner
        padded_map[:, :H, :W, :] = map_data
        # Optionally shuffle to distribute zeros randomly
        return shuffle_map_torch(padded_map)
    
    return map_data


def resize_map_torch(map_data, target_size):
    B, H, W, C = map_data.shape
    
    # Convert to [B, C, H, W] for F.interpolate
    map_permuted = map_data.permute(0, 3, 1, 2)
    
    # Resize
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    map_resized = F.interpolate(
        map_permuted, 
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    )
    
    # Convert back to [B, H, W, C]
    return map_resized.permute(0, 2, 3, 1)

def crop_map_torch(map_data, target_size):
    B, H, W, C = map_data.shape
    
    if isinstance(target_size, int):
        target_h = target_w = target_size
    else:
        target_h, target_w = target_size
    
    if H < target_h or W < target_w:
        # If smaller than target, pad first
        return pad_map_torch(map_data, max(target_h, target_w))
    
    # Random crop
    top = torch.randint(0, H - target_h + 1, (1,)).item()
    left = torch.randint(0, W - target_w + 1, (1,)).item()
    
    return map_data[:, top:top+target_h, left:left+target_w, :]


def uniform_sampling_map_torch(map_data, target_size, method='resize'):
    """
    Unified sampling function for 2D maps
    Args:
        map_data: [B, H, W, 3]
        target_size: int or tuple (target_H, target_W)
        method: 'resize', 'crop', or 'pad'
    Returns:
        sampled map: [B, target_H, target_W, 3]
    """
    B, H, W, C = map_data.shape
    
    if isinstance(target_size, int):
        target_h = target_w = target_size
    else:
        target_h, target_w = target_size
    
    # If already at target size, return as is
    if H == target_h and W == target_w:
        return map_data
    
    if method == 'resize':
        return resize_map_torch(map_data, target_size)
    elif method == 'crop':
        return crop_map_torch(map_data, target_size)
    elif method == 'pad':
        return pad_map_torch(map_data, target_size)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'resize', 'crop', or 'pad'.")

# ==================== 1D Point Cloud Encoder ====================

class MultiStagePointNetEncoder(nn.Module):
    """1D Point Cloud Encoder using 1D convolutions"""
    def __init__(self, h_dim=128, out_channels=128, num_layers=4, **kwargs):
        super().__init__()

        self.h_dim = h_dim
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.act = nn.LeakyReLU()

        self.conv_in = nn.Conv1d(3, h_dim, kernel_size=1)
        self.layers, self.global_layers = nn.ModuleList(), nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(nn.Conv1d(h_dim, h_dim, kernel_size=1))
            self.global_layers.append(nn.Conv1d(h_dim * 2, h_dim, kernel_size=1))
        self.conv_out = nn.Conv1d(h_dim * self.num_layers, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [B, L, 3] --> [B, 3, L]
        assert x.shape[-1] == 3, f"Input shape must have 3 channels at the last dim, got{x.shape}"
        x = x.transpose(1, 2)
        y = self.act(self.conv_in(x))
        feat_list = []
        for i in range(self.num_layers):
            y = self.act(self.layers[i](y))
            y_global = y.max(-1, keepdim=True).values
            y = torch.cat([y, y_global.expand_as(y)], dim=1)
            y = self.act(self.global_layers[i](y))
            feat_list.append(y)
        x = torch.cat(feat_list, dim=1)
        x = self.conv_out(x)

        x_global = x.max(-1).values  # [B, out_channels]

        return x_global


# ==================== 2D Point Cloud Map Encoder ====================

class MultiStageMapNetEncoder(nn.Module):
    """2D Point Cloud Map Encoder using 2D convolutions"""
    def __init__(self, h_dim=128, out_channels=128, num_layers=4, **kwargs):
        super().__init__()

        self.h_dim = h_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.act = nn.LeakyReLU()
        self.conv_in = nn.Conv2d(3, h_dim, kernel_size=3, padding=1)
        
        self.layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        
        for _ in range(self.num_layers):
            self.layers.append(nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1))
            self.global_layers.append(nn.Conv2d(h_dim * 2, h_dim, kernel_size=1, stride=1, padding=0))
        
        self.conv_out = nn.Conv2d(h_dim * self.num_layers, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x: [B, 3, H, W]
        y = self.act(self.conv_in(x))
        feat_list = []
        
        for i in range(self.num_layers):
            y = self.act(self.layers[i](y))
            y_global = F.adaptive_max_pool2d(y, 1)  # [B, h_dim, 1, 1]
            y_global_expanded = y_global.expand_as(y)
            y = torch.cat([y, y_global_expanded], dim=1)
            y = self.act(self.global_layers[i](y))
            feat_list.append(y)
        
        x = torch.cat(feat_list, dim=1)
        x = self.conv_out(x)

        x_global = F.adaptive_max_pool2d(x, 1)  # [B, out_channels, 1, 1]
        x_global = x_global.squeeze(-1).squeeze(-1)  # [B, out_channels]

        return x_global


# ==================== Unified iDP3 Encoder ====================

class iDP3Encoder(nn.Module):
    """
    Unified Point Cloud Encoder
    
    Supports 4 input formats:
    1. [B, L, 3]       - Single 1D point cloud per batch
    2. [B, N, L, 3]    - Multiple 1D point clouds (N views)
    3. [B, H, W, 3]    - Single 2D point cloud map per batch
    4. [B, N, H, W, 3] - Multiple 2D point cloud maps (N views)
    
    Output:
    - For single input: [B, out_channels]
    - For multi-view input: [B, N, out_channels] (encode each view separately)
    """
    def __init__(self, 
                 out_channels=128,
                 num_points=4096,  # Only used for 1D point cloud downsampling
                 target_map_size=224, # Target size for 2D map processing
                 h_dim=128,
                 num_layers=4,
                 point_downsample=True,  # Only for 1D point clouds
                 map_sampling_method='resize',  # 'resize', 'crop', or 'pad' for 2D maps
                 ):
        super().__init__()
        self.n_output_channels = out_channels
        self.num_points = num_points
        self.downsample = point_downsample
        self.target_map_size = target_map_size
        self.map_sampling_method = map_sampling_method
        
        # 1D Point Cloud Encoder
        self.pointnet_encoder = MultiStagePointNetEncoder(
            h_dim=h_dim,
            out_channels=out_channels,
            num_layers=num_layers
        )
        
        # 2D Map Encoder
        self.mapnet_encoder = MultiStageMapNetEncoder(
            h_dim=h_dim,
            out_channels=out_channels,
            num_layers=num_layers
        )
        
        overwatch.info(f"iDP3 Encoder has num layers: {num_layers}, h_dim: {h_dim}, output dim: {self.n_output_channels}")

    def _encode_1d_pointcloud(self, pc):
        if self.downsample:
            pc = uniform_sampling_torch(pc, self.num_points)
        return self.pointnet_encoder(pc)
    
    def _encode_2d_map(self, map_data):
        # [B, H, W, 3] -> [B, 3, H, W]
        if H != self.target_map_size or W != self.target_map_size:
            map_data = uniform_sampling_map_torch(
                map_data, 
                self.target_map_size, 
                method=self.map_sampling_method
            )
        map_data = map_data.permute(0, 3, 1, 2)
        return self.mapnet_encoder(map_data)

    def forward(self, x: torch.Tensor, multi_scene: bool = False) -> torch.Tensor:
        """
        Args:
            x: Point cloud tensor in one of these formats:
               - [B, L, 3]: Single 1D point cloud
               - [B, N, L, 3]: Multiple 1D point clouds
               - [B, H, W, 3]: Single 2D point cloud map
               - [B, N, H, W, 3]: Multiple 2D point cloud maps
        
        Returns:
            features: 
               - [B, out_channels] for single input
               - [B, N, out_channels] for multi-view input
        """
        assert x.shape[-1] == 3, f"Last dimension must be 3 (XYZ), got {x.shape[-1]}"
        
        ndim = len(x.shape)
        
        if ndim == 3:
            # Case 1: [B, L, 3] - Single 1D point cloud
            return self._encode_1d_pointcloud(x)
        
        elif ndim == 4:
            B, dim1, dim2, C = x.shape    
            # Distinguish between [B, N, L, 3] and [B, H, W, 3]
            if multi_scene: 
                # Case 2: [B, N, L, 3] - Multiple 1D point clouds
                B, N, L, C = x.shape
                x_reshaped = x.view(B * N, L, C) # Reshape to [B*N, L, 3]
                features = self._encode_1d_pointcloud(x_reshaped)  # [B*N, out_channels]
                features = features.view(B, N, -1) # Reshape back to [B, N, out_channels]
                return features
            
            else:
                # Case 3: [B, H, W, 3] - Single 2D point cloud map
                return self._encode_2d_map(x)
        
        elif ndim == 5:
            # Case 4: [B, N, H, W, 3] - Multiple 2D point cloud maps
            B, N, H, W, C = x.shape
            x_reshaped = x.view(B * N, H, W, C)
            features = self._encode_2d_map(x_reshaped)  # [B*N, out_channels]
            features = features.view(B, N, -1) # Reshape back to [B, N, out_channels]
            return features
        
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}. Expected 3, 4, or 5 dimensions.")
    @property
    def output_shape(self) -> int:
        return self.n_output_channels


# ==================== Main Test ====================

if __name__ == "__main__":
    print("="*70)
    print("Testing Unified iDP3Encoder")
    print("="*70)
    
    # Initialize encoder
    encoder = iDP3Encoder(
        out_channels=256,
        num_points=4096,
        h_dim=128,
        num_layers=4,
        point_downsample=True
    )
    
    print(f"\nTotal parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    encoder.eval()
    
    # ========== Case 1: [B, L, 3] - Single 1D point cloud ==========
    print("\n" + "="*70)
    print("Case 1: [B, L, 3] - Single 1D point cloud")
    print("="*70)
    
    B, L = 4, 8192
    pc_1d = torch.randn(B, L, 3)
    print(f"Input shape: {pc_1d.shape}")
    
    with torch.no_grad():
        out_1d = encoder(pc_1d)
    
    print(f"Output shape: {out_1d.shape}")
    print(f"Expected: [{B}, {encoder.output_shape}]")
    assert out_1d.shape == (B, 256), f"Shape mismatch! Got {out_1d.shape}"
    print("✓ Test passed!")
    
    # ========== Case 2: [B, N, L, 3] - Multiple 1D point clouds ==========
    print("\n" + "="*70)
    print("Case 2: [B, N, L, 3] - Multiple 1D point clouds")
    print("="*70)
    
    B, N, L = 4, 3, 8192
    pc_multi_1d = torch.randn(B, N, L, 3)
    print(f"Input shape: {pc_multi_1d.shape}")
    print(f"N={N} views, each with {L} points")
    
    with torch.no_grad():
        out_multi_1d = encoder(pc_multi_1d, multi_scene=True)
    
    print(f"Output shape: {out_multi_1d.shape}")
    print(f"Expected: [{B}, {N}, {encoder.output_shape}]")
    assert out_multi_1d.shape == (B, N, 256), f"Shape mismatch! Got {out_multi_1d.shape}"
    print("✓ Test passed! Each view encoded separately.")
    
    # ========== Case 3: [B, H, W, 3] - Single 2D point cloud map ==========
    print("\n" + "="*70)
    print("Case 3: [B, H, W, 3] - Single 2D point cloud map")
    print("="*70)
    
    B, H, W = 4, 224, 224
    map_2d = torch.randn(B, H, W, 3)
    print(f"Input shape: {map_2d.shape}")
    
    with torch.no_grad():
        out_2d = encoder(map_2d, multi_scene=False)
    
    print(f"Output shape: {out_2d.shape}")
    print(f"Expected: [{B}, {encoder.output_shape}]")
    assert out_2d.shape == (B, 256), f"Shape mismatch! Got {out_2d.shape}"
    print("✓ Test passed!")
    
    # ========== Case 4: [B, N, H, W, 3] - Multiple 2D point cloud maps ==========
    print("\n" + "="*70)
    print("Case 4: [B, N, H, W, 3] - Multiple 2D point cloud maps")
    print("="*70)
    
    B, N, H, W = 4, 5, 224, 224
    map_multi_2d = torch.randn(B, N, H, W, 3)
    print(f"Input shape: {map_multi_2d.shape}")
    print(f"N={N} views, each with {H}x{W} resolution")
    
    with torch.no_grad():
        out_multi_2d = encoder(map_multi_2d)
    
    print(f"Output shape: {out_multi_2d.shape}")
    print(f"Expected: [{B}, {N}, {encoder.output_shape}]")
    assert out_multi_2d.shape == (B, N, 256), f"Shape mismatch! Got {out_multi_2d.shape}"
    print("✓ Test passed! Each map encoded separately.")
    
    # ========== Summary ==========
    print("\n" + "="*70)
    print("Summary of All Test Cases")
    print("="*70)
    print(f"Case 1: [B, L, 3]       → [{B}, {encoder.output_shape}]")
    print(f"Case 2: [B, N, L, 3]    → [{B}, {N}, {encoder.output_shape}] ({N} views)")
    print(f"Case 3: [B, H, W, 3]    → [{B}, {encoder.output_shape}]")
    print(f"Case 4: [B, N, H, W, 3] → [{B}, {N}, {encoder.output_shape}] ({N} maps)")
    print("\n✨ All tests passed! Unified encoder works correctly for all cases.")
    print("="*70)