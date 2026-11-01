# Contains the code for the 3D Improved U-Net and benchmarrk 2D U-Net architectures

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 2D U-Net baseline inspired by Ronneberger et al. (2015) reference in task sheet
# Commit Note: Passed smoke test for correct number of parameters across channels, and input/output shapes match correctly
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



# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
class DiceCELoss(nn.Module):
    """Cross-entropy plus soft Dice loss for multi-class segmentation.

    - Cross-entropy gives stable per-pixel gradients
    - Soft Dice measures overlap per class
    - Dice is computed over the whole batch

    Args:
        num_classes:        number of classes, inc. background (class 0)
        include_background: include class 0 in the Dice average
        smooth:             small constant that avoids division by zero
    """

    def __init__(self, num_classes, include_background=False, smooth=1e-5,
                 ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.smooth = smooth
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, labels):
        """logits: (N, C, *spatial) raw scores; labels: (N, *spatial) class indices."""
        logits = logits.float() 
        ce = F.cross_entropy(logits, labels)

        probs = logits.softmax(dim=1)
        target = F.one_hot(labels, self.num_classes).movedim(-1, 1).float()
        sum_dims = (0,) + tuple(range(2, logits.dim()))  # sum over batch and all spatial dims
        intersection = (probs * target).sum(sum_dims)
        denominator = probs.sum(sum_dims) + target.sum(sum_dims)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)  # one score per class

        if not self.include_background:
            dice = dice[1:]
        return self.ce_weight * ce + self.dice_weight * (1 - dice.mean())





# ---------------------------------------------------------------------------
# 3D Improved U-Net (Isensee et al., 2018)
# ---------------------------------------------------------------------------
def _norm_act(channels):
    """Instance normalisation + leaky ReLU, the normalisation/activation pair used throughout."""
    return [nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)]


class ConvNormAct3D(nn.Sequential):
    """3D convolution followed by instance normalisation and leaky ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2),
            *_norm_act(out_channels))


class ContextModule(nn.Module):
    """Pre-activation residual block of the context (encoder) pathway.

    (norm, LReLU, 3x3x3 conv) -> dropout -> (norm, LReLU, 3x3x3 conv), with the block's
    input added to its output (identity shortcut).
    """

    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            *_norm_act(channels),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.Dropout3d(dropout),
            *_norm_act(channels),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class UpsamplingModule(nn.Sequential):
    """Nearest-neighbour 2x upscale followed by a 3x3x3 convolution that halves the channels."""

    def __init__(self, in_channels, out_channels):
        super().__init__(nn.Upsample(scale_factor=2, mode='nearest'),
                         ConvNormAct3D(in_channels, out_channels))


class LocalizationModule(nn.Sequential):
    """Recombines upsampled and skip features: 3x3x3 conv, then a 1x1x1 conv halving the channels."""

    def __init__(self, in_channels, out_channels):
        super().__init__(ConvNormAct3D(in_channels, in_channels),
                         ConvNormAct3D(in_channels, out_channels, kernel_size=1))


class ImprovedUNet3D(nn.Module):
    """3D Improved U-Net (Isensee et al., 2018) for volumetric multi-class segmentation.

    Context pathway: per level, a 3x3x3 convolution (stride 2 below the first level) followed
    by a residual context module. Localization pathway: per level, an upsampling module, a
    skip-connection concatenation and a localization module. Deep supervision: segmentation
    layers at the three highest-resolution decoder levels are upscaled and summed element-wise
    into the final output.

    Args:
        in_channels:   number of input channels (1 for MRI)
        num_classes:   number of output classes
        base_channels: channels at the first level; doubled at each level down
        depth:         number of down-sampling steps (input sides must be divisible by 2**depth)
        dropout:       dropout probability inside the context modules

    Returns raw logits of shape (N, num_classes, D, H, W); softmax is applied in the loss.
    """

    def __init__(self, in_channels=1, num_classes=6, base_channels=16, depth=4, dropout=0.3):
        super().__init__()
        if depth < 3:
            raise ValueError('depth must be at least 3 for three deep-supervision levels')
        self.depth = depth
        channels = [base_channels * 2 ** level for level in range(depth + 1)]  # e.g. 16..256

        # Context (encoder) pathway
        self.down_convs = nn.ModuleList()
        self.contexts = nn.ModuleList()
        previous = in_channels
        for level, ch in enumerate(channels):
            stride = 1 if level == 0 else 2
            self.down_convs.append(nn.Conv3d(previous, ch, kernel_size=3, stride=stride, padding=1))
            self.contexts.append(ContextModule(ch, dropout))
            previous = ch

        # Localization (decoder) pathway, from the level just above the bottleneck to the top
        self.upsamples = nn.ModuleList()
        self.localizations = nn.ModuleList()
        for level in reversed(range(depth)):
            self.upsamples.append(UpsamplingModule(channels[level + 1], channels[level]))
            if level > 0:
                self.localizations.append(LocalizationModule(2 * channels[level], channels[level]))
            else:  # the top level uses a single 3x3x3 convolution, as in the paper
                self.localizations.append(ConvNormAct3D(2 * channels[0], 2 * channels[0]))

        # Deep supervision: segmentation layers at the three highest-resolution decoder levels
        self.seg_levels = (2, 1, 0)
        self.seg_layers = nn.ModuleDict({
            str(level): nn.Conv3d(channels[level] if level > 0 else 2 * channels[0],
                                  num_classes, kernel_size=1)
            for level in self.seg_levels})
        self.upscale = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x):
        factor = 2 ** self.depth
        if any(size % factor for size in x.shape[2:]):
            raise ValueError(f'Input size {tuple(x.shape[2:])} must be divisible by {factor}')

        skips = []
        for level, (down_conv, context) in enumerate(zip(self.down_convs, self.contexts)):
            x = context(down_conv(x))
            if level < self.depth:  # the bottleneck has no skip connection
                skips.append(x)

        seg_sum = None
        for upsample, localization, level in zip(self.upsamples, self.localizations,
                                                 reversed(range(self.depth))):
            x = localization(torch.cat([skips[level], upsample(x)], dim=1))
            if level in self.seg_levels:
                seg = self.seg_layers[str(level)](x)
                # Upscale the running sum to this level's resolution and add this level's output
                seg_sum = seg if seg_sum is None else self.upscale(seg_sum) + seg
        return seg_sum