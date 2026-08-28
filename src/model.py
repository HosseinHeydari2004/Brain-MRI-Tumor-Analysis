import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet34,
    ResNet34_Weights
)

# ===================================
# SEGMENTATION MODEL
# ===================================

class ResNet34UNet3Plus(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        decoder_channels=64
    ):

        super().__init__()

        self.encoder = ResNet34Encoder(
            in_channels=in_channels
        )

        self.dec4 = UNet3PlusDecoderBlock(
            skip_channels=[
                64,
                64,
                128,
                256,
                512
            ],
            out_channels=decoder_channels
        )

        self.dec3 = UNet3PlusDecoderBlock(
            skip_channels=[
                64,
                64,
                128,
                256,
                decoder_channels
            ],
            out_channels=decoder_channels
        )

        self.dec2 = UNet3PlusDecoderBlock(
            skip_channels=[
                64,
                64,
                128,
                decoder_channels,
                decoder_channels
            ],
            out_channels=decoder_channels
        )

        self.dec1 = UNet3PlusDecoderBlock(
            skip_channels=[
                64,
                64,
                decoder_channels,
                decoder_channels,
                decoder_channels
            ],
            out_channels=decoder_channels
        )

        self.final_refinement = nn.Sequential(

            nn.Conv2d(
                decoder_channels,
                decoder_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                decoder_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                decoder_channels,
                decoder_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                decoder_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

        self.final_conv = nn.Conv2d(
            decoder_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        e1, e2, e3, e4, e5 = self.encoder(x)

        d4 = self.dec4(
            [
                e1,
                e2,
                e3,
                e4,
                e5
            ],
            target_size=e4.shape[-2:]
        )

        d3 = self.dec3(
            [
                e1,
                e2,
                e3,
                e4,
                d4
            ],
            target_size=e3.shape[-2:]
        )

        d2 = self.dec2(
            [
                e1,
                e2,
                e3,
                d3,
                d4
            ],
            target_size=e2.shape[-2:]
        )

        d1 = self.dec1(
            [
                e1,
                e2,
                d2,
                d3,
                d4
            ],
            target_size=e1.shape[-2:]
        )

        out = F.interpolate(
            d1,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        out = self.final_refinement(
            out
        )

        out = self.final_conv(
            out
        )

        return out


class ResNet34Encoder(nn.Module):

    def __init__(self, in_channels=1):

        super().__init__()

        backbone = resnet34(
            weights=ResNet34_Weights.DEFAULT
        )

        original_conv = backbone.conv1

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        with torch.no_grad():

            new_conv.weight.copy_(
                original_conv.weight.mean(
                    dim=1,
                    keepdim=True
                )
            )

        backbone.conv1 = new_conv

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        e1 = x

        x = self.maxpool(x)
        x = self.layer1(x)

        e2 = x

        x = self.layer2(x)

        e3 = x

        x = self.layer3(x)

        e4 = x

        x = self.layer4(x)

        e5 = x

        return e1, e2, e3, e4, e5


class UNet3PlusDecoderBlock(nn.Module):

    def __init__(
        self,
        skip_channels,
        out_channels=64
    ):

        super().__init__()

        self.out_channels = out_channels

        self.projections = nn.ModuleList()

        for channels in skip_channels:

            self.projections.append(
                nn.Sequential(

                    nn.Conv2d(
                        channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False
                    ),

                    nn.BatchNorm2d(
                        out_channels
                    ),

                    nn.ReLU(
                        inplace=True
                    )
                )
            )

        self.fusion = nn.Sequential(

            nn.Conv2d(
                out_channels * len(skip_channels),
                out_channels * len(skip_channels),
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels * len(skip_channels)
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels * len(skip_channels),
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(
        self,
        features,
        target_size
    ):

        fused_features = []

        for feature, projection in zip(
            features,
            self.projections
        ):

            feature = F.interpolate(
                feature,
                size=target_size,
                mode="bilinear",
                align_corners=False
            )

            feature = projection(
                feature
            )

            fused_features.append(
                feature
            )

        x = torch.cat(
            fused_features,
            dim=1
        )

        x = self.fusion(x)

        return x

# ====================================
# CLASSIFICATION MODEL
# ====================================

class ResNet34Classifier(nn.Module):

    def __init__(
        self,
        num_classes,
        in_channels=1
    ):

        super().__init__()

        backbone = resnet34(
            weights=ResNet34_Weights.DEFAULT
        )

        original_conv = backbone.conv1

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        with torch.no_grad():

            new_conv.weight.copy_(
                original_conv.weight.mean(
                    dim=1,
                    keepdim=True
                )
            )

        backbone.conv1 = new_conv

        num_features = (
            backbone.fc.in_features
        )

        backbone.fc = nn.Linear(
            num_features,
            num_classes
        )

        self.backbone = backbone

    def forward(self, x):

        return self.backbone(x)
