"""
pi3_loader.py

Implementations of pi3_loader, loading pi3 model which predicts pointclouds and camera extrinsics from images. 
"""
from typing import Tuple, List, Optional, Dict, Union, Type
from pathlib import Path
from termcolor import cprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from prismatic.models.pi3.models.pi3 import Pi3
from prismatic.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)

def load_pc_model(pi3_path: Union[str, Path]) -> Pi3:
    overwatch.info(f"Loading PC model from {pi3_path}")
    if pi3_path is not None:
        pc_model = Pi3.from_pretrained(Path(pi3_path) if isinstance(pi3_path, str) else pi3_path)
        overwatch.info(f"PC model Loaded Successfully from loacal dir: {pi3_path}")
    else:
        raise ValueError("Please provide a valid path or repo id to a PC model")
    
    return pc_model

# Pointcloud Encoder 
def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out

def maxpool(x, dim=-1, keepdim=False):
    out = x.max(dim=dim, keepdim=keepdim).values
    return out

class MultiStagePointNetEncoder(nn.Module):
    def __init__(self, h_dim=128, out_channels=128, num_layers=4, **kwargs):
        super().__init__()

        self.h_dim = h_dim
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=False)

        self.conv_in = nn.Conv1d(3, h_dim, kernel_size=1)
        self.layers, self.global_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Conv1d(h_dim, h_dim, kernel_size=1))
            self.global_layers.append(nn.Conv1d(h_dim * 2, h_dim, kernel_size=1))
        self.conv_out = nn.Conv1d(h_dim * self.num_layers, out_channels, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, N, 3] --> [B, 3, N]
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

        x_global = x.max(-1).values

        return x_global

def shuffle_point_numpy(point_cloud):
    B, N, C = point_cloud.shape
    indices = np.random.permutation(N)
    return point_cloud[:, indices]

def pad_point_numpy(point_cloud, num_points):
    B, N, C = point_cloud.shape
    if num_points > N:
        num_pad = num_points - N
        pad_points = np.zeros((B, num_pad, C))
        point_cloud = np.concatenate([point_cloud, pad_points], axis=1)
        point_cloud = shuffle_point_numpy(point_cloud)
    return point_cloud

def uniform_sampling_numpy(point_cloud, num_points):
    B, N, C = point_cloud.shape
    # padd if num_points > N
    if num_points > N:
        return pad_point_numpy(point_cloud, num_points)
    
    # random sampling
    indices = np.random.permutation(N)[:num_points]
    sampled_points = point_cloud[:, indices]
    return sampled_points

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
    device = point_cloud.device
    # padd if num_points > N
    if num_points == N:
        return point_cloud
    if num_points > N:
        return pad_point_torch(point_cloud, num_points)
    
    # random sampling
    indices = torch.randperm(N)[:num_points]
    sampled_points = point_cloud[:, indices]
    return sampled_points

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules

    
class iDP3Encoder(nn.Module):
    """
    修改后的 iDP3Encoder，只处理点云数据，删除了所有 state 相关的部分
    """
    def __init__(self, 
                 observation_space: Dict, 
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='multi_stage_pointnet',
                 point_downsample=True,
                 ):
        super().__init__()
        self.point_cloud_key = 'point_cloud'
        self.n_output_channels = pointcloud_encoder_cfg.out_channels
        
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.num_points = pointcloud_encoder_cfg.num_points  # 4096
        
        print(f"[iDP3Encoder] point cloud shape: {self.point_cloud_shape}")

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        
        self.downsample = point_downsample
        if self.downsample:
            self.point_preprocess = uniform_sampling_torch
        else:
            self.point_preprocess = nn.Identity()
        
        if pointnet_type == "multi_stage_pointnet":
            self.extractor = MultiStagePointNetEncoder(
                out_channels=pointcloud_encoder_cfg.out_channels
            )
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        print(f"[iDP3Encoder] output dim: {self.n_output_channels}")

    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, f"point cloud shape: {points.shape}, length should be 3"

        # 下采样点云
        if self.downsample:
            points = self.point_preprocess(points, self.num_points)
           
        # 提取点云特征
        pn_feat = self.extractor(points)  # B * out_channels
        
        return pn_feat

    def output_shape(self):
        return self.n_output_channels
    

class PointCloudEncoderConfig:
    def __init__(self, out_channels=128, num_points=4096):
        self.out_channels = out_channels
        self.num_points = num_points

if __name__ == "__main__":
    
    pc_model = load_pc_model("/home/ruihengwang/vla/VLA-Adapter/pretrained_models/pi3_checkpoint")
    batch_size = 2
    out_channels = 128
    num_points = 4096
    observation_space = {
            'point_cloud': (num_points, 3),
        }
    pointcloud_encoder_cfg = PointCloudEncoderConfig(
        out_channels=out_channels,
        num_points=num_points
    )
    encoder = iDP3Encoder(
        observation_space=observation_space,
        pointcloud_encoder_cfg=pointcloud_encoder_cfg,
        point_downsample=True
    )
    encoder.eval()
    point_cloud = torch.randn(batch_size, num_points, 3)
    
    observations = {
        'point_cloud': point_cloud
    }
    with torch.no_grad():
        output = encoder(observations)
    print(f"Input shape: {observations['point_cloud'].shape}")
    print(f"\n输出特征形状: {output.shape}")
    print(f"输出特征范围: [{output.min():.3f}, {output.max():.3f}]")
    print(f"输出维度: {encoder.output_shape()}")
