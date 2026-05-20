"""ResNet + CBAM model for ILS spectrogram classification.

Adds Convolutional Block Attention Module (CBAM) after ResNet backbone
to help the model focus on local burst textures (e.g. Arc transients).
"""
import torch
import torch.nn as nn
import timm


class ChannelAttention(nn.Module):
    """Channel attention: learns which feature channels are important."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention: learns which locations are important."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(attn)


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Applies channel attention followed by spatial attention.
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x


class ResNetArcAux(nn.Module):
    """ResNet with Arc auxiliary branch.

    Main branch: 5-class multi-label output
    Arc aux branch: binary Arc prediction with shared backbone features

    Loss = BCE_multi-label + lambda * BCE_Arc
    """

    def __init__(self, backbone: str = 'resnet18', num_classes: int = 5,
                 pretrained: bool = True, arc_lambda: float = 0.5):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.arc_lambda = arc_lambda

        # Shared backbone (no global pool - we handle it in forward)
        base_model = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0,
            global_pool='',
        )
        self.backbone = base_model

        # Determine feature dimension
        self.feat_dim = self._get_feature_dim(backbone)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Main multi-label head
        self.main_head = nn.Linear(self.feat_dim, num_classes)

        # Arc auxiliary head (binary)
        self.arc_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feat_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    @staticmethod
    def _get_feature_dim(backbone_name: str) -> int:
        dims = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048,
                'sigresnet18': 512, 'sigresnet34': 512}
        return dims.get(backbone_name, 512)

    def forward(self, x):
        # Shared features [B, C, H, W]
        features = self.backbone.forward_features(x)
        # Pool and flatten
        features = self.pool(features).flatten(1)  # [B, C]
        # Main multi-label output
        main_out = self.main_head(features)
        # Arc auxiliary output
        arc_out = self.arc_head(features)
        return main_out, arc_out


class ResNetCBAM(nn.Module):
    """ResNet backbone + CBAM + classifier for multi-label classification.

    Args:
        backbone: timm model name (e.g. 'resnet18', 'resnet34')
        num_classes: Number of output classes (default 5)
        pretrained: Use ImageNet pretrained weights
    """

    def __init__(self, backbone: str = 'resnet18', num_classes: int = 5,
                 pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        # Build the backbone model to get its feature dimension
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0,  # remove classifier
            global_pool='',  # keep spatial feature map
        )

        # Determine feature channels (depends on backbone)
        feature_info = self._get_feature_channels(backbone)
        self.feat_channels = feature_info

        # CBAM attention
        self.cbam = CBAM(self.feat_channels)

        # Classifier head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.feat_channels, num_classes)

    @staticmethod
    def _get_feature_channels(backbone_name: str) -> int:
        channels = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'sigresnet18': 512,
            'sigresnet34': 512,
        }
        return channels.get(backbone_name, 512)

    def forward(self, x):
        # Extract features from backbone (keep spatial dims)
        x = self.backbone.forward_features(x)
        # CBAM attention
        x = self.cbam(x)
        # Global pooling + classifier
        x = self.global_pool(x).flatten(1)
        x = self.classifier(x)
        return x
