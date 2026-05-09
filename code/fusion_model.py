"""Model definitions used by the fusion inference pipeline.

The module keeps the original inference architecture and parameter names so
checkpoints saved by the training script can be loaded directly.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_channels(features: torch.Tensor) -> torch.Tensor:
    assert features.dim() == 4
    spatial_sum = features.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (features.size(2) * features.size(3))


def stdv_channels(features: torch.Tensor) -> torch.Tensor:
    assert features.dim() == 4
    feature_mean = mean_channels(features)
    feature_variance = (
        (features - feature_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True)
        / (features.size(2) * features.size(3))
    )
    return feature_variance.pow(0.5)


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=1, padding=0),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.Conv2d(
                base_channels,
                base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.Relu = nn.ReLU(inplace=True)

        self.res_block = nn.Sequential(
            ResidualLayer(in_channels, base_channels),
            ResidualLayer(base_channels, base_channels * 2),
            ResidualLayer(base_channels * 2, base_channels * 3),
        )
        self.attention_net = nn.Sequential(
            nn.Conv2d(
                base_channels * 4,
                base_channels // 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.1),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shallow_features = self.init_conv(x)
        residual_features = self.res_block(x)
        features = torch.cat([shallow_features, residual_features], dim=1)
        attention = self.attention_net(features)
        return features, shallow_features, attention


class ResidualLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv_layers(x)
        out += residual
        return self.relu(out)


class EnhancedSCGatedConv(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, in_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, attention_map: torch.Tensor) -> torch.Tensor:
        channel_weight = self.channel_gate(x)
        spatial_input = torch.cat([x, attention_map], dim=1)
        spatial_weight = self.spatial_gate(spatial_input)
        combined_weight = channel_weight * spatial_weight * (1 - attention_map)
        gated_x = x * combined_weight
        return self.conv(gated_x) + x


class MultiScaleDecoder(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, in_channels * 2),
            nn.ReLU(),
        )

        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, in_channels * 2),
        )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.final_conv = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_down = self.downsample(x)
        x_res = F.relu(self.res_block(x_down) + x_down)
        x_up = self.upsample(x_res)
        return torch.sigmoid(self.final_conv(x_up))


class FusionDecoderWithDegradation(nn.Module):
    def __init__(self, vis_channels: int, ir_channels: int) -> None:
        super().__init__()

        self.vis_align = nn.Conv2d(vis_channels, 64, kernel_size=1)
        self.ir_align = nn.Conv2d(ir_channels, 64, kernel_size=1)
        self.sc_gated_conv = EnhancedSCGatedConv(128)
        self.decoder = MultiScaleDecoder(64)

    def forward(
        self,
        features_vis: torch.Tensor,
        features_ir: torch.Tensor,
        attention_vis: torch.Tensor,
        attention_ir: torch.Tensor,
    ) -> torch.Tensor:
        features_vis = self.vis_align(features_vis)
        features_ir = self.ir_align(features_ir)
        features_fused = torch.cat([features_vis, features_ir], dim=1)
        attention_combined = (attention_vis + attention_ir) / 2

        features_gated = self.sc_gated_conv(features_fused, attention_combined)
        features_gated = features_gated[:, :64, :, :]
        return self.decoder(features_gated)
