# Contains the code for the 3D Improved U-Net and benchmarrk 2D U-Net architectures

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 2D U-Net baseline inspired by Ronneberger et al. (2015) reference in task sheet
# ---------------------------------------------------------------------------
class DoubleConv2D(nn.Module):
    """Two 3x3 convolutions, each followed by batch normalisation and ReLU.

    padding=1 keeps the spatial size unchanged, so encoder features can be concatenated in the decoder without cropping
    (the original paper used unpadded convolutions and cropped instead).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    """2D U-Net for slice-wise multi-class segmentation.

    Parameters:
        in_channels:   number of input image channels (1 for MRI)
        num_classes:   number of output classes (one logit map per class)
        base_channels: channels at the first level
        depth:         number of down-sampling steps (input sides must be divisible by 2**depth)

    Return: raw logits of shape (N, num_classes, H, W); softmax is applied in the loss.
    """

    def __init__(self, in_channels=1, num_classes=6, base_channels=64, depth=4):
        super().__init__()
        self.depth = depth
        channels = [base_channels * 2 ** level for level in range(depth + 1)]  # e.g. 64..1024

        # Encoder: one DoubleConv per level; the last one is bottleneck
        self.encoders = nn.ModuleList()
        previous = in_channels
        for ch in channels:
            self.encoders.append(DoubleConv2D(previous, ch))
            previous = ch
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Decoder: upsample, concatenate the matching skip connection, DoubleConv
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for ch in reversed(channels[:-1]):
            self.upsamples.append(nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv2D(ch * 2, ch))  # ch (upsampled) + ch (skip)

        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        factor = 2 ** self.depth
        if x.shape[-2] % factor or x.shape[-1] % factor:
            raise ValueError(f'Input size {tuple(x.shape[-2:])} must be divisible by {factor}')

        skips = []
        for level, encoder in enumerate(self.encoders):
            x = encoder(x)
            if level < self.depth:  # no pooling after the bottleneck
                skips.append(x)
                x = self.pool(x)

        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = upsample(x)
            x = decoder(torch.cat([skip, x], dim=1))

        return self.head(x)