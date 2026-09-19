# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from einops.layers.torch import Rearrange

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock

__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3', 'ConvNormLayer', 'BasicBlock', 
           'BottleNeck', 'Blocks', 'MSLCB', 'CSSA', 'RepC3_DE')


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

################################### RT-DETR PResnet ###################################
def get_activation(act: str, inpace: bool=True):
    '''get activation
    '''
    act = act.lower()
    
    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()
        
    elif act is None:
        m = nn.Identity()
    
    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 


    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        out = out + short
        out = self.act(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out 

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class Blocks(nn.Module):
    def __init__(self, ch_in, ch_out, block, count, stage_num, act='relu', input_resolution=None, sr_ratio=None, kernel_size=None, kan_name=None, variant='d'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            if input_resolution is not None and sr_ratio is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        input_resolution=input_resolution,
                        sr_ratio=sr_ratio)
                )
            elif kernel_size is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        kernel_size=kernel_size)
                )
            elif kan_name is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        kan_name=kan_name)
                )
            else:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act)
                )
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out



def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# -----------------------------------------------------------------------------
# Multi-Scale Local Contrast Block (MSLCB)
# -----------------------------------------------------------------------------

class LocalContrastEnhance(nn.Module):
    """Enhance local contrast by subtracting a 5x5 local mean."""

    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.padding = kernel_size // 2
        self.local_avg = nn.AvgPool2d(
            kernel_size, stride=1, padding=self.padding
        )
        self.contrast_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        local_mean = self.local_avg(x)
        contrast = x - local_mean
        return self.contrast_conv(torch.cat([x, contrast], dim=1))


class MultiScalePooling(nn.Module):
    """Multi-scale context branch using 8x8, 4x4, and 2x2 adaptive pooling."""

    def __init__(self, channels):
        super().__init__()
        self.c1 = channels // 3
        self.c2 = channels // 3
        self.c3 = channels - self.c1 - self.c2
        self.pool_scales = (8, 4, 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.c1, self.c1, 1, bias=False),
            nn.BatchNorm2d(self.c1),
            nn.SiLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.c2, self.c2, 1, bias=False),
            nn.BatchNorm2d(self.c2),
            nn.SiLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.c3, self.c3, 1, bias=False),
            nn.BatchNorm2d(self.c3),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2:]

        x1 = x[:, :self.c1, :, :]
        x2 = x[:, self.c1:self.c1 + self.c2, :, :]
        x3 = x[:, self.c1 + self.c2:, :, :]

        p1 = F.adaptive_avg_pool2d(x1, self.pool_scales[0])
        p2 = F.adaptive_avg_pool2d(x2, self.pool_scales[1])
        p3 = F.adaptive_max_pool2d(x3, self.pool_scales[2])

        p1 = self.conv1(p1)
        p2 = self.conv2(p2)
        p3 = self.conv3(p3)

        p1 = F.interpolate(p1, size=(h, w), mode="bilinear", align_corners=False)
        p2 = F.interpolate(p2, size=(h, w), mode="bilinear", align_corners=False)
        p3 = F.interpolate(p3, size=(h, w), mode="bilinear", align_corners=False)

        return torch.cat([p1, p2, p3], dim=1)


class SpatialAttention(nn.Module):
    """Spatial attention based on channel-wise max and mean descriptors."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial = torch.cat([max_out, avg_out], dim=1)
        spatial = self.sigmoid(self.conv(spatial))
        return x * spatial


class MSLCB(nn.Module):
    """Multi-Scale Local Contrast Block used in the CSD-DETR backbone."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()

        self.contrast = LocalContrastEnhance(c1, kernel_size=5)
        self.multi_pool = MultiScalePooling(c1)

        self.fusion = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

        self.spatial_att = SpatialAttention(kernel_size=7)
        self.conv_out = nn.Conv2d(
            c1, c2, k, s, autopad(k, p), groups=g, bias=False
        )
        self.bn_out = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        contrast_feat = self.contrast(x)
        pool_feat = self.multi_pool(x)
        fused = self.fusion(torch.cat([contrast_feat, pool_feat], dim=1))
        fused = self.spatial_att(fused)
        fused = fused + x
        return self.act(self.bn_out(self.conv_out(fused)))


# -----------------------------------------------------------------------------
# Coordinate State-Space Attention (CSSA)
# -----------------------------------------------------------------------------

class ConvLayer2D(nn.Module):
    """Lightweight 2D convolution wrapper used by HSM-SSD."""

    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        norm=nn.BatchNorm2d,
        act_layer=nn.ReLU,
        bn_weight_init=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=False,
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None

        if self.norm and hasattr(self.norm, "weight") and self.norm.weight is not None:
            nn.init.constant_(self.norm.weight, bn_weight_init)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvLayer1D(nn.Module):
    """Lightweight 1D convolution wrapper used by HSM-SSD."""

    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        norm=nn.BatchNorm1d,
        act_layer=nn.ReLU,
        bn_weight_init=1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None

        if self.norm and hasattr(self.norm, "weight") and self.norm.weight is not None:
            nn.init.constant_(self.norm.weight, bn_weight_init)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class HSMSSD(nn.Module):
    """Hidden State Mixer-based State Space Duality (HSM-SSD)."""

    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim=64):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(ssd_expand * d_model)
        self.state_dim = state_dim

        self.BCdt_proj = ConvLayer1D(
            d_model, 3 * state_dim, 1, norm=None, act_layer=None
        )
        conv_dim = state_dim * 3
        self.dw = ConvLayer2D(
            conv_dim,
            conv_dim,
            3,
            1,
            1,
            groups=conv_dim,
            norm=None,
            act_layer=None,
            bn_weight_init=0,
        )
        self.hz_proj = ConvLayer1D(
            d_model, 2 * self.d_inner, 1, norm=None, act_layer=None
        )
        self.out_proj = ConvLayer1D(
            self.d_inner, d_model, 1, norm=None, act_layer=None, bn_weight_init=0
        )

        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = nn.Parameter(A)
        self.act = nn.SiLU(inplace=False)
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

    def forward(self, x, height=None, width=None):
        """
        Args:
            x: Tensor of shape [B, C, L], where L = height * width.
            height: Spatial height before flattening.
            width: Spatial width before flattening.

        Passing height and width explicitly removes the previous assumption that
        the feature map must be square. A square fallback is retained for
        backward compatibility.
        """
        batch, _, length = x.shape

        if height is None or width is None:
            side = int(math.sqrt(length))
            if side * side != length:
                raise ValueError(
                    "HSMSSD requires height and width for non-square feature maps."
                )
            height = width = side

        if height * width != length:
            raise ValueError(
                f"Inconsistent spatial size: height*width={height * width}, L={length}."
            )

        BCdt = self.BCdt_proj(x).view(batch, -1, height, width)
        BCdt = self.dw(BCdt).flatten(2)
        B, C, dt = torch.split(
            BCdt,
            [self.state_dim, self.state_dim, self.state_dim],
            dim=1,
        )

        B = B.contiguous()
        C = C.contiguous()
        dt = dt.contiguous()

        A = (dt + self.A.view(1, -1, 1)).softmax(-1)
        AB = A * B
        h = x @ AB.transpose(-2, -1)

        hz = self.hz_proj(h)
        h, z = torch.split(hz, [self.d_inner, self.d_inner], dim=1)
        h = h.contiguous()
        z = z.contiguous()

        h = self.out_proj(h * self.act(z) + h * self.D)
        y = h @ C
        y = y.view(batch, -1, height, width).contiguous()
        return y, h


class CSSA(nn.Module):
    """Coordinate State-Space Attention used to replace AIFI in CSD-DETR."""

    def __init__(self, in_channels, reduction=4, state_dim=32):
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError("CSSA requires an even number of input channels.")

        self.in_channels = in_channels
        self.half_channels = in_channels // 2
        self.reduced_channels = max(in_channels // reduction, 8)

        self.x_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.y_pool = nn.AdaptiveAvgPool2d((1, None))

        self.conv_gn = nn.Sequential(
            nn.Conv2d(
                self.half_channels,
                self.reduced_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=min(8, self.reduced_channels),
                num_channels=self.reduced_channels,
            ),
            nn.SiLU(inplace=False),
        )

        self.dw_conv5x5 = nn.Sequential(
            nn.Conv2d(
                self.reduced_channels,
                self.reduced_channels,
                kernel_size=5,
                padding=2,
                groups=self.reduced_channels,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=min(8, self.reduced_channels),
                num_channels=self.reduced_channels,
            ),
            nn.SiLU(inplace=False),
        )

        self.attn_conv = nn.Conv2d(
            self.reduced_channels, self.reduced_channels, kernel_size=1, bias=False
        )
        self.conv_out = nn.Conv2d(
            self.reduced_channels, self.half_channels, kernel_size=1, bias=False
        )

        self.hsmssd = HSMSSD(
            d_model=self.half_channels,
            ssd_expand=1,
            state_dim=state_dim,
        )

        self.gate_conv = nn.Conv2d(
            self.half_channels, self.half_channels, kernel_size=1, bias=False
        )
        self.sigmoid = nn.Sigmoid()
        self.final_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, bias=False
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        _, _, height, width = x.shape
        identity = x

        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = x1.contiguous()
        x2 = x2.contiguous()

        # Coordinate-attention branch.
        x_pool = self.x_pool(x1)
        y_pool = self.y_pool(x1)
        y_pool_t = y_pool.permute(0, 1, 3, 2).contiguous()

        coord_feat = torch.cat([x_pool, y_pool_t], dim=2)
        coord_feat = self.conv_gn(coord_feat)
        coord_feat = self.dw_conv5x5(coord_feat)

        attn_weights = F.softmax(self.attn_conv(coord_feat), dim=2)
        coord_feat = self.conv_out(coord_feat * attn_weights)

        x_attn, y_attn = torch.split(coord_feat, [height, width], dim=2)
        x_attn = self.sigmoid(x_attn.contiguous())
        y_attn = self.sigmoid(
            y_attn.permute(0, 1, 3, 2).contiguous()
        )
        right_branch = x1 * x_attn * y_attn

        # State-space branch.
        x2_flat = x2.flatten(2).contiguous()
        left_branch, _ = self.hsmssd(x2_flat, height, width)

        gate_input = left_branch.mean(dim=(2, 3), keepdim=True)
        gate = self.sigmoid(
            self.gate_conv(gate_input).expand_as(left_branch)
        )
        left_branch = left_branch * gate

        # Gated fusion and residual connection.
        out = torch.cat([left_branch, right_branch], dim=1)
        out = out * self.sigmoid(self.final_conv(out))
        return out + identity


# -----------------------------------------------------------------------------
# Detail-Enhanced RepC3 (RepC3_DE)
# -----------------------------------------------------------------------------

class Conv2d_cd(nn.Module):
    """Central-difference convolution kernel generator."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        theta=1.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange(
            "c_in c_out k1 k2 -> c_in c_out (k1 k2)"
        )(conv_weight)

        conv_weight_cd = torch.zeros(
            conv_shape[0],
            conv_shape[1],
            9,
            device=conv_weight.device,
            dtype=conv_weight.dtype,
        )
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = (
            conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        )
        conv_weight_cd = Rearrange(
            "c_in c_out (k1 k2) -> c_in c_out k1 k2",
            k1=conv_shape[2],
            k2=conv_shape[3],
        )(conv_weight_cd)
        return conv_weight_cd, self.conv.bias


class Conv2d_ad(nn.Module):
    """Angular-difference convolution kernel generator."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        theta=1.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange(
            "c_in c_out k1 k2 -> c_in c_out (k1 k2)"
        )(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[
            :, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]
        ]
        conv_weight_ad = Rearrange(
            "c_in c_out (k1 k2) -> c_in c_out k1 k2",
            k1=conv_shape[2],
            k2=conv_shape[3],
        )(conv_weight_ad)
        return conv_weight_ad, self.conv.bias


class Conv2d_hd(nn.Module):
    """Horizontal-difference convolution kernel generator."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        theta=1.0,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_hd = torch.zeros(
            conv_shape[0],
            conv_shape[1],
            9,
            device=conv_weight.device,
            dtype=conv_weight.dtype,
        )
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange(
            "c_in c_out (k1 k2) -> c_in c_out k1 k2",
            k1=conv_shape[2],
            k2=conv_shape[2],
        )(conv_weight_hd)
        return conv_weight_hd, self.conv.bias


class Conv2d_vd(nn.Module):
    """Vertical-difference convolution kernel generator."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = torch.zeros(
            conv_shape[0],
            conv_shape[1],
            9,
            device=conv_weight.device,
            dtype=conv_weight.dtype,
        )
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange(
            "c_in c_out (k1 k2) -> c_in c_out k1 k2",
            k1=conv_shape[2],
            k2=conv_shape[2],
        )(conv_weight_vd)
        return conv_weight_vd, self.conv.bias


class DEConv(nn.Module):


    def __init__(self, dim):
        super().__init__()
        # Keep the original module names for checkpoint compatibility.
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        weight = w1 + w2 + w3 + w4 + w5
        bias = b1 + b2 + b3 + b4 + b5

        return F.conv2d(
            x,
            weight=weight,
            bias=bias,
            stride=1,
            padding=1,
            groups=1,
        )


class RepC3_DE(nn.Module):
    """RepC3 enhanced with multi-directional difference convolution."""

    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1_proj = (
            nn.Conv2d(c1, c_, 1, 1, bias=False)
            if c1 != c_
            else nn.Identity()
        )
        self.cv1_bn = nn.BatchNorm2d(c_)
        self.cv1_deconv = DEConv(c_)

        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        x1 = self.cv1_proj(x)
        x1 = self.cv1_deconv(x1)
        x1 = F.silu(self.cv1_bn(x1))
        return self.cv3(self.m(x1) + self.cv2(x))
