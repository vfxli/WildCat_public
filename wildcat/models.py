# -*- coding: utf-8 -*-
"""WildCat multi-scale coarse-to-fine EncoderDecoder neural architecture (Small, Medium, Large)."""
import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cat_format

MODEL_SPECS = {
    "Small": {"k_size": 3, "n_resblocks": 1, "params": 1_033_955},
    "Medium": {"k_size": 5, "n_resblocks": 1, "params": 2_578_659},
    "Large": {"k_size": 5, "n_resblocks": 3, "params": 6_881_251},
    "WildCatLargeV1": {"k_size": 5, "n_resblocks": 2, "params": 5_078_665},
}


def get_same_padding(x: int, k: int, s: int, d: int = 1) -> int:
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def pad_same(
    x: torch.Tensor,
    k: Tuple[int, int],
    s: Tuple[int, int],
    d: Tuple[int, int] = (1, 1),
    value: float = 0.0,
) -> torch.Tensor:
    ih, iw = x.shape[-2], x.shape[-1]
    pad_h = get_same_padding(ih, k[0], s[0], d[0])
    pad_w = get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(
            x,
            [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
            mode="constant",
            value=value,
        )
    return x


class Conv2dSame(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        cudnn_pad: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (int(kernel_size), int(kernel_size))
        self.stride = (int(stride), int(stride))
        self.padding = (0, 0)
        self.dilation = (int(dilation), int(dilation))
        self.groups = int(groups)
        self.cudnn_pad = bool(cudnn_pad)
        self._can_fuse_pad = (
            self.stride == (1, 1)
            and self.dilation == (1, 1)
            and (self.kernel_size[0] % 2 == 1)
            and (self.kernel_size[1] % 2 == 1)
        )
        self._sym_pad = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.weight = nn.Parameter(
            torch.empty(
                self.out_channels,
                self.in_channels // self.groups,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cudnn_pad and self._can_fuse_pad:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self._sym_pad,
                self.dilation,
                self.groups,
            )
        x = pad_same(x, self.kernel_size, self.stride, self.dilation, 0.0)
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            (0, 0),
            self.dilation,
            self.groups,
        )


class Conv2dTransposeSame(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (int(kernel_size), int(kernel_size))
        self.stride = (int(stride), int(stride))
        self.padding = (0, 0)
        self.dilation = (int(dilation), int(dilation))
        self.groups = int(groups)
        self.weight = nn.Parameter(
            torch.empty(
                self.in_channels,
                self.out_channels // self.groups,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.shape[-2], x.shape[-1]
        pad_h = get_same_padding(ih, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_w = get_same_padding(iw, self.kernel_size[1], self.stride[1], self.dilation[1])
        return F.conv_transpose2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            (pad_h // 2, pad_w // 2),
            (0, 0),
            self.groups,
            self.dilation,
        )


class ResNetBlock(nn.Module):
    def __init__(self, channels: int, k_size: int):
        super().__init__()
        self.conv2d_1 = Conv2dSame(channels, channels, k_size, stride=1)
        self.relu = nn.ReLU()
        self.conv2d_2 = Conv2dSame(channels, channels, k_size, stride=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv2d_1(inputs)
        x = self.relu(x)
        x = self.conv2d_2(x)
        return x + inputs


class EncoderDecoder(nn.Module):
    """Exact multi-scale coarse-to-fine EncoderDecoder topology used in Small, Medium, and Large tiers."""

    def __init__(
        self,
        channels_in: int = 3,
        channels_out: int = 3,
        k_size: int = 5,
        n_resblocks: int = 1,
        n_levels: int = 3,
        scale: float = 0.5,
    ):
        super().__init__()
        self.n_levels = int(n_levels)
        self.scale = float(scale)
        self.channels_in = int(channels_in)
        self.channels_out = int(channels_out)
        self.k_size = int(k_size)
        self.n_resblocks = int(n_resblocks)
        self.cudnn_pad = False
        self.relu = nn.ReLU()

        c_cat = self.channels_in + self.channels_out

        if self.n_resblocks == 1:
            self.conv1_1 = Conv2dSame(c_cat, 32, k_size, stride=1)
            self.block1_2 = ResNetBlock(32, k_size)
            self.block1 = nn.Sequential(self.conv1_1, self.relu, self.block1_2)

            self.conv2_1 = Conv2dSame(32, 64, k_size, stride=2)
            self.block2_2 = ResNetBlock(64, k_size)
            self.block2 = nn.Sequential(self.conv2_1, self.relu, self.block2_2)

            self.conv3_1 = Conv2dSame(64, 128, k_size, stride=2)
            self.block3_2 = ResNetBlock(128, k_size)
            self.block3 = nn.Sequential(self.conv3_1, self.relu, self.block3_2)

            self.deblock3_1 = ResNetBlock(128, k_size)
            self.deconv1_1 = Conv2dTransposeSame(128, 64, 4, stride=2)
            self.block4 = nn.Sequential(self.deblock3_1, self.deconv1_1, self.relu)

            self.deblock2_1 = ResNetBlock(64, k_size)
            self.deconv2_1 = Conv2dTransposeSame(64, 32, 4, stride=2)
            self.block5 = nn.Sequential(self.deblock2_1, self.deconv2_1, self.relu)

            self.deblock1_1 = ResNetBlock(32, k_size)
            self.deconv3_1 = Conv2dTransposeSame(32, self.channels_out, k_size, stride=1)
            self.block6 = nn.Sequential(self.deblock1_1, self.deconv3_1)
        else:
            self.conv1_1 = Conv2dSame(c_cat, 32, k_size, stride=1)
            self.block1_2 = ResNetBlock(32, k_size)
            self.block1_3 = ResNetBlock(32, k_size)
            self.block1_4 = ResNetBlock(32, k_size)
            self.block1 = nn.Sequential(
                self.conv1_1, self.relu, self.block1_2, self.block1_3, self.block1_4
            )

            self.conv2_1 = Conv2dSame(32, 64, k_size, stride=2)
            self.block2_2 = ResNetBlock(64, k_size)
            self.block2_3 = ResNetBlock(64, k_size)
            self.block2_4 = ResNetBlock(64, k_size)
            self.block2 = nn.Sequential(
                self.conv2_1, self.relu, self.block2_2, self.block2_3, self.block2_4
            )

            self.conv3_1 = Conv2dSame(64, 128, k_size, stride=2)
            self.block3_2 = ResNetBlock(128, k_size)
            self.block3_3 = ResNetBlock(128, k_size)
            self.block3_4 = ResNetBlock(128, k_size)
            self.block3 = nn.Sequential(
                self.conv3_1, self.relu, self.block3_2, self.block3_3, self.block3_4
            )

            self.deblock3_3 = ResNetBlock(128, k_size)
            self.deblock3_2 = ResNetBlock(128, k_size)
            self.deblock3_1 = ResNetBlock(128, k_size)
            self.deconv1_1 = Conv2dTransposeSame(128, 64, 4, stride=2)
            self.block4 = nn.Sequential(
                self.deblock3_3,
                self.deblock3_2,
                self.deblock3_1,
                self.deconv1_1,
                self.relu,
            )

            self.deblock2_3 = ResNetBlock(64, k_size)
            self.deblock2_2 = ResNetBlock(64, k_size)
            self.deblock2_1 = ResNetBlock(64, k_size)
            self.deconv2_1 = Conv2dTransposeSame(64, 32, 4, stride=2)
            self.block5 = nn.Sequential(
                self.deblock2_3,
                self.deblock2_2,
                self.deblock2_1,
                self.deconv2_1,
                self.relu,
            )

            self.deblock1_3 = ResNetBlock(32, k_size)
            self.deblock1_2 = ResNetBlock(32, k_size)
            self.deblock1_1 = ResNetBlock(32, k_size)
            self.deconv3_1 = Conv2dTransposeSame(32, self.channels_out, k_size, stride=1)
            self.block6 = nn.Sequential(
                self.deblock1_3,
                self.deblock1_2,
                self.deblock1_1,
                self.deconv3_1,
            )

    def set_cudnn_pad(self, enabled: bool = True):
        self.cudnn_pad = bool(enabled)
        for m in self.modules():
            if isinstance(m, Conv2dSame):
                m.cudnn_pad = bool(enabled)
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_all(inputs)[-1]

    def forward_all(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """All pyramid-level predictions, coarse -> fine (CopyCat supervises every level)."""
        h = inputs.shape[2]
        w = inputs.shape[3]
        if self.channels_out == self.channels_in:
            input_pred = inputs
        else:
            input_pred = torch.zeros(
                (inputs.size(0), self.channels_out, h, w),
                dtype=inputs.dtype,
                device=inputs.device,
            )
        n_outputs: List[torch.Tensor] = []
        for i in range(self.n_levels):
            scale = self.scale ** (self.n_levels - i - 1)
            hi = int(h * scale)
            wi = int(w * scale)
            if self.cudnn_pad and hi == h and wi == w:
                input_init = inputs
            else:
                input_init = F.interpolate(
                    inputs, size=(hi, wi), mode="bilinear", align_corners=True
                )
            if self.cudnn_pad and i == 0 and self.channels_out == self.channels_in:
                input_pred = input_init
            else:
                input_pred = F.interpolate(
                    input_pred, size=(hi, wi), mode="bilinear", align_corners=True
                )
            input_all = torch.cat((input_init, input_pred), dim=1)
            conv1_4 = self.block1(input_all)
            conv2_4 = self.block2(conv1_4)
            encoded = self.block3(conv2_4)
            deconv1_4 = self.block4(encoded)
            cat2 = deconv1_4 + conv2_4
            deconv2_4 = self.block5(cat2)
            cat1 = deconv2_4 + conv1_4
            input_pred = self.block6(cat1)
            n_outputs.append(input_pred)
        return n_outputs


def _init_hom_conv(m: nn.Conv2d, gain: float = 1.0) -> None:
    nn.init.xavier_uniform_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class _HomResBlock(nn.Module):
    """Scale-invariant 1-homogeneous residual block (LeakyReLU 0.05, no normalization)."""

    def __init__(self, dim: int, k_size: int = 3):
        super().__init__()
        p = k_size // 2
        self.c1 = nn.Conv2d(dim, dim, kernel_size=k_size, padding=p, bias=True)
        self.c2 = nn.Conv2d(dim, dim, kernel_size=k_size, padding=p, bias=True)
        _init_hom_conv(self.c1, 1.0)
        _init_hom_conv(self.c2, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.c2(F.leaky_relu(self.c1(x), 0.05))


class _HomFFNBlock(nn.Module):
    """1-Homogeneous pointwise 1x1 inverted-bottleneck FFN (dim -> expand*dim -> LeakyReLU -> dim)."""

    def __init__(self, dim: int, expand: int = 3):
        super().__init__()
        hidden = dim * expand
        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=True)
        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)
        _init_hom_conv(self.pw1, 1.0)
        _init_hom_conv(self.pw2, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw2(F.leaky_relu(self.pw1(x), 0.05))


class _UpShuffle(nn.Module):
    """Sub-pixel upsampling: 1x1 projection + PixelShuffle(2)."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out * 4, kernel_size=1, bias=True)
        self.shuffle = nn.PixelShuffle(2)
        _init_hom_conv(self.proj, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.proj(x))


class WildCatLargeV1(nn.Module):
    """WildCat v3 single-pass 4-scale [64, 96, 128, 160] asymmetric 1-homogeneous U-Net + 1x1 FFN tail."""

    def __init__(self, channels_in: int = 3, channels_out: int = 3):
        super().__init__()
        self.channels_in = int(channels_in)
        self.channels_out = int(channels_out)
        c1, c2, c3, c4 = 64, 96, 128, 160

        self.conv1 = nn.Conv2d(self.channels_in, c1, kernel_size=5, padding=2, bias=True)
        self.enc1 = nn.Sequential(_HomResBlock(c1, 3), _HomResBlock(c1, 3))

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=5, stride=2, padding=2, bias=True)
        self.enc2 = nn.Sequential(_HomResBlock(c2, 3), _HomResBlock(c2, 3))

        self.conv3 = nn.Conv2d(c2, c3, kernel_size=5, stride=2, padding=2, bias=True)
        self.enc3 = nn.Sequential(_HomResBlock(c3, 3), _HomResBlock(c3, 5))

        self.conv4 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=True)
        self.mid = nn.Sequential(_HomResBlock(c4, 5), _HomResBlock(c4, 3))

        self.up3 = _UpShuffle(c4, c3)
        self.dec3 = nn.Sequential(_HomResBlock(c3, 3), _HomFFNBlock(c3, expand=2))

        self.up2 = _UpShuffle(c3, c2)
        self.dec2 = nn.Sequential(_HomResBlock(c2, 3), _HomResBlock(c2, 3), _HomFFNBlock(c2, expand=2))

        self.up1 = _UpShuffle(c2, c1)
        self.dec1 = nn.Sequential(
            _HomResBlock(c1, 3),
            _HomResBlock(c1, 3),
            _HomFFNBlock(c1, expand=3),
            _HomFFNBlock(c1, expand=3),
        )

        for m in (self.conv1, self.conv2, self.conv3, self.conv4):
            _init_hom_conv(m, 1.0)

        self.head_q = nn.Conv2d(c3, self.channels_out, kernel_size=3, padding=1, bias=True)
        self.head_h = nn.Conv2d(c2, self.channels_out, kernel_size=3, padding=1, bias=True)
        self.head_f = nn.Conv2d(c1, self.channels_out, kernel_size=3, padding=1, bias=True)
        for h in (self.head_q, self.head_h, self.head_f):
            _init_hom_conv(h, 0.05)

    def _encode_decode(self, x: torch.Tensor):
        e1 = self.enc1(F.leaky_relu(self.conv1(x), 0.05))
        e2 = self.enc2(F.leaky_relu(self.conv2(e1), 0.05))
        e3 = self.enc3(F.leaky_relu(self.conv3(e2), 0.05))
        m = self.mid(F.leaky_relu(self.conv4(e3), 0.05))
        d3 = self.dec3(F.leaky_relu(self.up3(m), 0.05) + e3)
        d2 = self.dec2(F.leaky_relu(self.up2(d3), 0.05) + e2)
        d1 = self.dec1(F.leaky_relu(self.up1(d2), 0.05) + e1)
        return d3, d2, d1

    @torch.jit.ignore
    def forward_all(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        h, w = inputs.shape[2], inputs.shape[3]
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        x = F.pad(inputs, (0, pw, 0, ph), mode="replicate") if (ph > 0 or pw > 0) else inputs
        d3, d2, d1 = self._encode_decode(x)
        hp, wp = x.shape[2], x.shape[3]
        same_c = self.channels_out == self.channels_in

        pq = self.head_q(d3)
        if same_c:
            xq = F.interpolate(x, size=(hp // 4, wp // 4), mode="bilinear", align_corners=True)
            pq = xq + pq

        ph_out = self.head_h(d2)
        if same_c:
            xh = F.interpolate(x, size=(hp // 2, wp // 2), mode="bilinear", align_corners=True)
            ph_out = xh + ph_out

        pf = self.head_f(d1)[:, :, :h, :w]
        if same_c:
            pf = inputs + pf
        return [pq, ph_out, pf]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.contiguous()
        h, w = inputs.shape[2], inputs.shape[3]
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        x = torch.ops.aten.replication_pad2d(inputs, [0, pw, 0, ph]) if (ph > 0 or pw > 0) else inputs
        _, _, d1 = self._encode_decode(x)
        pf = self.head_f(d1)[:, :, :h, :w]
        out = (inputs + pf) if (self.channels_out == self.channels_in) else pf
        return out.contiguous()


def _log_compress(x: torch.Tensor) -> torch.Tensor:
    """Smooth signed log1p compression for HDR-safe attention query/key matching."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _coord_features_from_box(
    b: int,
    h: int,
    w: int,
    box: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate 6-ch normalized 2D coordinate & 1st-harmonic Fourier map [B, 6, H, W]."""
    ix = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / float(max(w, 1))
    iy = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / float(max(h, 1))
    if box is None:
        cx = (-1.0 + 2.0 * ix).view(1, 1, 1, w).expand(b, 1, h, w)
        cy = (-1.0 + 2.0 * iy).view(1, 1, h, 1).expand(b, 1, h, w)
    else:
        bx = box.to(device=device, dtype=torch.float32)
        x0 = bx[:, 0].view(b, 1, 1, 1)
        y0 = bx[:, 1].view(b, 1, 1, 1)
        x1 = bx[:, 2].view(b, 1, 1, 1)
        y1 = bx[:, 3].view(b, 1, 1, 1)
        cx = (x0 + (x1 - x0) * ix.view(1, 1, 1, w)).expand(b, 1, h, w)
        cy = (y0 + (y1 - y0) * iy.view(1, 1, h, 1)).expand(b, 1, h, w)
    pi = 3.141592653589793
    feats = torch.cat(
        [
            cx,
            cy,
            torch.sin(pi * cx),
            torch.cos(pi * cx),
            torch.sin(pi * cy),
            torch.cos(pi * cy),
        ],
        dim=1,
    )
    return feats.to(dtype=dtype)


class GlobalRadianceStem(nn.Module):
    """Sector radiance encoder ([B, C_in, 128, 128] -> [B, dim, G, G]) with zero spatial 3x3 convs."""

    def __init__(self, channels_in: int = 3, dim: int = 160, grid_size: int = 16):
        super().__init__()
        self.channels_in = int(channels_in)
        self.dim = int(dim)
        self.grid_size = int(grid_size)
        self.k = 128 // self.grid_size

        in_feats = self.channels_in * 5
        self.pw1 = nn.Conv2d(in_feats, 48, kernel_size=1, bias=True)
        self.pw2 = nn.Conv2d(48, self.dim, kernel_size=1, bias=False)
        _init_hom_conv(self.pw1, 1.0)
        _init_hom_conv(self.pw2, 1.0)

    def forward(self, thumb_128: torch.Tensor) -> torch.Tensor:
        pad = self.k // 2
        sm = F.avg_pool2d(
            torch.ops.aten.replication_pad2d(thumb_128, [pad, pad, pad, pad]),
            kernel_size=2 * pad + 1,
            stride=1,
        )
        e_avg = F.avg_pool2d(sm, kernel_size=self.k, stride=self.k)
        e_rms = torch.sqrt(F.avg_pool2d(sm * sm, kernel_size=self.k, stride=self.k) + 1e-6)
        e_hi = F.avg_pool2d(F.relu(sm - 0.35), kernel_size=self.k, stride=self.k)
        feats = torch.cat([e_avg, e_rms, e_hi, _log_compress(e_avg), _log_compress(e_rms)], dim=1)
        return self.pw2(F.leaky_relu(self.pw1(feats), 0.05))


class GlobalLightAttnFFNBlock(nn.Module):
    """1-Layer Global Cosine Cross-Attention + Bilinear-HomFFN at the 1/8 bottleneck."""

    def __init__(
        self,
        dim: int = 160,
        grid_size: int = 16,
        heads: int = 4,
        head_dim: int = 32,
        hidden: int = 320,
        max_log_tau: float = 1.8,
    ):
        super().__init__()
        self.dim = int(dim)
        self.grid_size = int(grid_size)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.heads * self.head_dim
        self.max_log_tau = float(max_log_tau)

        self.coord_q = nn.Conv2d(6, dim, kernel_size=1, bias=True)
        self.coord_k = nn.Conv2d(6, dim, kernel_size=1, bias=True)

        self.q_proj = nn.Conv2d(dim, self.inner_dim, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(dim, self.inner_dim, kernel_size=1, bias=True)
        self.v_proj = nn.Conv2d(dim, self.inner_dim, kernel_size=1, bias=False)
        init_tau = min(math.log(5.0), self.max_log_tau)
        self.log_tau = nn.Parameter(torch.full((1, self.heads, 1, 1), init_tau))

        self.pw_face = nn.Conv2d(dim, hidden, kernel_size=1, bias=True)
        self.pw_light = nn.Conv2d(self.inner_dim, hidden, kernel_size=1, bias=False)
        self.pw_cat = nn.Conv2d(dim + self.inner_dim, hidden, kernel_size=1, bias=True)
        self.pw_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), 0.2))

        for m in (
            self.coord_q,
            self.coord_k,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.pw_face,
            self.pw_light,
            self.pw_cat,
        ):
            _init_hom_conv(m, 1.0)
        _init_hom_conv(self.pw_out, 0.25)

    def forward(
        self,
        m: torch.Tensor,
        g_ctx: torch.Tensor,
        crop_box: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, _, h8, w8 = m.shape
        g = self.grid_size
        n_q = h8 * w8
        n_k = g * g

        q_coords = _coord_features_from_box(b, h8, w8, crop_box, m.device, m.dtype)
        k_coords = _coord_features_from_box(b, g, g, None, g_ctx.device, g_ctx.dtype)

        q_in = _log_compress(m) + self.coord_q(q_coords)
        k_in = _log_compress(g_ctx) + self.coord_k(k_coords)

        q = self.q_proj(q_in).reshape(b, self.heads, self.head_dim, n_q).permute(0, 1, 3, 2)
        k = self.k_proj(k_in).reshape(b, self.heads, self.head_dim, n_k)
        v = self.v_proj(g_ctx).reshape(b, self.heads, self.head_dim, n_k).permute(0, 1, 3, 2)

        qf = F.normalize(q.float(), p=2.0, dim=-1, eps=1e-6)
        kf = F.normalize(k.float(), p=2.0, dim=-2, eps=1e-6)
        scale = torch.exp(torch.clamp(self.log_tau.float(), min=0.0, max=self.max_log_tau))
        sim = torch.matmul(qf, kf) * scale
        attn = torch.softmax(sim, dim=-1)

        out_v = torch.matmul(attn, v.float()).to(dtype=m.dtype)
        retrieved = out_v.permute(0, 1, 3, 2).reshape(b, self.inner_dim, h8, w8)

        bilin = torch.tanh(self.pw_face(m)) * self.pw_light(retrieved)
        add_ffn = F.leaky_relu(self.pw_cat(torch.cat([m, retrieved], dim=1)), 0.05)
        delta = self.pw_out(bilin + add_ffn)
        return m + self.gamma.to(dtype=m.dtype) * delta


class WildCatLargeV1Global(nn.Module):
    """WildCatLargeV1 + optional 16x16 Global Radiance Cross-Attention + Bilinear-HomFFN."""

    def __init__(self, channels_in: int = 3, channels_out: int = 3, grid_size: int = 16):
        super().__init__()
        self.channels_in = int(channels_in)
        self.channels_out = int(channels_out)
        self.grid_size = int(grid_size)
        self.uses_global_thumb = True
        c1, c2, c3, c4 = 64, 96, 128, 160

        self.conv1 = nn.Conv2d(self.channels_in, c1, kernel_size=5, padding=2, bias=True)
        self.enc1 = nn.Sequential(_HomResBlock(c1, 3), _HomResBlock(c1, 3))

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=5, stride=2, padding=2, bias=True)
        self.enc2 = nn.Sequential(_HomResBlock(c2, 3), _HomResBlock(c2, 3))

        self.conv3 = nn.Conv2d(c2, c3, kernel_size=5, stride=2, padding=2, bias=True)
        self.enc3 = nn.Sequential(_HomResBlock(c3, 3), _HomResBlock(c3, 5))

        self.conv4 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=True)
        self.mid = nn.Sequential(_HomResBlock(c4, 5), _HomResBlock(c4, 3))

        self.light_stem = GlobalRadianceStem(channels_in=self.channels_in, dim=c4, grid_size=self.grid_size)
        self.global_block = GlobalLightAttnFFNBlock(
            dim=c4,
            grid_size=self.grid_size,
            heads=4,
            head_dim=32,
            hidden=320,
            max_log_tau=1.8,
        )

        self.up3 = _UpShuffle(c4, c3)
        self.dec3 = nn.Sequential(_HomResBlock(c3, 3), _HomFFNBlock(c3, expand=2))

        self.up2 = _UpShuffle(c3, c2)
        self.dec2 = nn.Sequential(_HomResBlock(c2, 3), _HomResBlock(c2, 3), _HomFFNBlock(c2, expand=2))

        self.up1 = _UpShuffle(c2, c1)
        self.dec1 = nn.Sequential(
            _HomResBlock(c1, 3),
            _HomResBlock(c1, 3),
            _HomFFNBlock(c1, expand=3),
            _HomFFNBlock(c1, expand=3),
        )

        for m in (self.conv1, self.conv2, self.conv3, self.conv4):
            _init_hom_conv(m, 1.0)

        self.head_q = nn.Conv2d(c3, self.channels_out, kernel_size=3, padding=1, bias=True)
        self.head_h = nn.Conv2d(c2, self.channels_out, kernel_size=3, padding=1, bias=True)
        self.head_f = nn.Conv2d(c1, self.channels_out, kernel_size=3, padding=1, bias=True)
        for h in (self.head_q, self.head_h, self.head_f):
            _init_hom_conv(h, 0.05)

    def _encode_decode(
        self,
        x_pad: torch.Tensor,
        orig_inputs: torch.Tensor,
        full_thumb: Optional[torch.Tensor] = None,
        crop_box: Optional[torch.Tensor] = None,
    ):
        e1 = self.enc1(F.leaky_relu(self.conv1(x_pad), 0.05))
        e2 = self.enc2(F.leaky_relu(self.conv2(e1), 0.05))
        e3 = self.enc3(F.leaky_relu(self.conv3(e2), 0.05))
        m = self.mid(F.leaky_relu(self.conv4(e3), 0.05))

        thumb = (
            full_thumb
            if full_thumb is not None
            else F.interpolate(orig_inputs, size=(128, 128), mode="bilinear", align_corners=False)
        )
        g_ctx = self.light_stem(thumb)
        m = self.global_block(m, g_ctx, crop_box)

        d3 = self.dec3(F.leaky_relu(self.up3(m), 0.05) + e3)
        d2 = self.dec2(F.leaky_relu(self.up2(d3), 0.05) + e2)
        d1 = self.dec1(F.leaky_relu(self.up1(d2), 0.05) + e1)
        return d3, d2, d1

    @torch.jit.ignore
    def forward_all(
        self,
        inputs: torch.Tensor,
        full_thumb: Optional[torch.Tensor] = None,
        crop_box: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        h, w = inputs.shape[2], inputs.shape[3]
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        x = F.pad(inputs, (0, pw, 0, ph), mode="replicate") if (ph > 0 or pw > 0) else inputs
        adj_box = crop_box
        if crop_box is not None and (ph > 0 or pw > 0):
            bx = crop_box.float()
            dx = (bx[:, 2] - bx[:, 0]) * (float(w + pw) / float(max(w, 1)))
            dy = (bx[:, 3] - bx[:, 1]) * (float(h + ph) / float(max(h, 1)))
            adj_box = torch.stack([bx[:, 0], bx[:, 1], bx[:, 0] + dx, bx[:, 1] + dy], dim=1)

        d3, d2, d1 = self._encode_decode(x, inputs, full_thumb=full_thumb, crop_box=adj_box)
        hp, wp = x.shape[2], x.shape[3]
        same_c = self.channels_out == self.channels_in

        pq = self.head_q(d3)
        if same_c:
            xq = F.interpolate(x, size=(hp // 4, wp // 4), mode="bilinear", align_corners=True)
            pq = xq + pq

        ph_out = self.head_h(d2)
        if same_c:
            xh = F.interpolate(x, size=(hp // 2, wp // 2), mode="bilinear", align_corners=True)
            ph_out = xh + ph_out

        pf = self.head_f(d1)[:, :, :h, :w]
        if same_c:
            pf = inputs + pf
        return [pq, ph_out, pf]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.contiguous()
        h, w = inputs.shape[2], inputs.shape[3]
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        x = torch.ops.aten.replication_pad2d(inputs, [0, pw, 0, ph]) if (ph > 0 or pw > 0) else inputs
        b = inputs.shape[0]
        base_box = torch.tensor([[-1.0, -1.0, 1.0, 1.0]], device=inputs.device, dtype=torch.float32).expand(b, 4)
        if ph > 0 or pw > 0:
            dx = (base_box[:, 2] - base_box[:, 0]) * (float(w + pw) / float(max(w, 1)))
            dy = (base_box[:, 3] - base_box[:, 1]) * (float(h + ph) / float(max(h, 1)))
            box: Optional[torch.Tensor] = torch.stack(
                [base_box[:, 0], base_box[:, 1], base_box[:, 0] + dx, base_box[:, 1] + dy],
                dim=1,
            )
        else:
            box = base_box
        _, _, d1 = self._encode_decode(x, inputs, full_thumb=None, crop_box=box)
        pf = self.head_f(d1)[:, :, :h, :w]
        out = (inputs + pf) if (self.channels_out == self.channels_in) else pf
        return out.contiguous()


def build_generator(mo: dict, cudnn_pad: bool = False):
    model_size = mo.get("model_size", "Medium")
    if model_size not in MODEL_SPECS:
        model_size = "Medium"
    spec = MODEL_SPECS[model_size]
    c_in = int(mo.get("in_channels", 3))
    c_out = int(mo.get("out_channels", c_in))
    if model_size == "WildCatLargeV1":
        if bool(mo.get("global_attention", False)):
            return WildCatLargeV1Global(channels_in=c_in, channels_out=c_out), spec["k_size"]
        return WildCatLargeV1(channels_in=c_in, channels_out=c_out), spec["k_size"]
    net = EncoderDecoder(
        channels_in=c_in,
        channels_out=c_out,
        k_size=spec["k_size"],
        n_resblocks=spec["n_resblocks"],
        n_levels=int(mo.get("n_levels", 3)),
        scale=float(mo.get("scale", 0.5)),
    )
    if cudnn_pad or mo.get("cudnn_pad", False):
        net.set_cudnn_pad(True)
    return net, spec["k_size"]


def _resolve_weight_path(weight_source: str) -> str:
    if not weight_source or os.path.isabs(weight_source) or os.path.isfile(weight_source):
        return weight_source
    norm = str(weight_source).replace("\\", "/")
    run_env = os.environ.get("WC_RUN_ROOT", "").strip()
    if run_env and norm.startswith("run/"):
        cand = os.path.join(run_env, norm[4:])
        if os.path.isfile(cand):
            return cand
    return weight_source


def inspect_weight_source(
    weight_source: str,
    fallback_model_size: str = "Medium",
    fallback_in: int = 3,
    fallback_out: int = 3,
) -> dict:
    """Inspect an initial_weights source ('None' or a .pt file path) and return its architecture and metadata."""
    import io
    import tempfile
    import zipfile

    fb_size = fallback_model_size if fallback_model_size in MODEL_SPECS else "Medium"
    if not weight_source or weight_source == "None":
        return {
            "model_size": fb_size,
            "global_attention": False,
            "in_channels": int(fallback_in),
            "out_channels": int(fallback_out),
            "lock_model_size": False,
            "lock_channels": False,
            "step": 0,
            "total_steps": 0,
            "epochs": 0,
            "source": "",
            "target": "",
        }

    pt_path = _resolve_weight_path(weight_source)
    tmp_pt = None
    try:
        target_pt = pt_path

        if os.path.isfile(target_pt) and zipfile.is_zipfile(target_pt):
            with zipfile.ZipFile(target_pt, "r") as z:
                dp_list = [fn for fn in z.namelist() if fn.endswith("data.pkl")]
                if dp_list:
                    obj = cat_format._TorchScriptUnpickler(io.BytesIO(z.read(dp_list[0]))).load()
                    if isinstance(obj, dict):
                        sd = obj.get("G", obj)
                        cfg = obj.get("config") or {}
                        glob_attn = False
                        if "conv4.weight" in sd and "head_f.weight" in sd:
                            in_ch = int(sd["conv1.weight"][2][1])
                            out_ch = int(sd["head_f.weight"][2][0])
                            ms = "WildCatLargeV1"
                            glob_attn = ("global_block.q_proj.weight" in sd) or bool(
                                ((cfg.get("model") or {}).get("global_attention", False))
                            )
                        else:
                            c1_shape = sd["conv1_1.weight"][2]
                            d3_shape = sd["deconv3_1.weight"][2]
                            k_size = int(c1_shape[2])
                            out_ch = int(d3_shape[1])
                            in_ch = int(c1_shape[1]) - out_ch
                            is_large = "block1_3.conv2d_1.weight" in sd
                            ms = "Large" if is_large else ("Small" if k_size == 3 else "Medium")
                        return {
                            "model_size": ms,
                            "global_attention": glob_attn,
                            "in_channels": in_ch,
                            "out_channels": out_ch,
                            "lock_model_size": True,
                            "lock_channels": True,
                            "step": int(obj.get("step", 0)),
                            "total_steps": int((cfg.get("train") or {}).get("total_steps", 0)),
                            "epochs": int((cfg.get("train") or {}).get("epochs", 0)),
                            "source": (cfg.get("paths") or {}).get("source", ""),
                            "target": (cfg.get("paths") or {}).get("target", ""),
                        }
                    else:
                        st = getattr(obj, "state", {})
                        tensors = dict((k, shp) for k, _, shp in cat_format._walk_tensors(obj))
                        glob_attn = False
                        if "conv4.weight" in tensors and "head_f.weight" in tensors:
                            in_ch = int(st.get("channels_in", tensors["conv1.weight"][1]))
                            out_ch = int(st.get("channels_out", tensors["head_f.weight"][0]))
                            ms = "WildCatLargeV1"
                            glob_attn = "global_block.q_proj.weight" in tensors
                        else:
                            c1_shape = tensors.get("conv1_1.weight", (32, 6, 5, 5))
                            d3_shape = tensors.get("deconv3_1.weight", (32, 3, 5, 5))
                            k_size = int(st.get("k_size", c1_shape[2]))
                            out_ch = int(st.get("channels_out", d3_shape[1]))
                            in_ch = int(st.get("channels_in", c1_shape[1] - out_ch))
                            is_large = "block1_3.conv2d_1.weight" in tensors
                            ms = "Large" if is_large else ("Small" if k_size == 3 else "Medium")
                        return {
                            "model_size": ms,
                            "global_attention": glob_attn,
                            "in_channels": in_ch,
                            "out_channels": out_ch,
                            "lock_model_size": True,
                            "lock_channels": True,
                            "step": 0,
                            "total_steps": 0,
                            "epochs": 0,
                            "source": "",
                            "target": "",
                        }
    except Exception:
        pass
    finally:
        if tmp_pt and os.path.exists(tmp_pt):
            os.remove(tmp_pt)

    return {
        "model_size": fb_size,
        "global_attention": False,
        "in_channels": int(fallback_in),
        "out_channels": int(fallback_out),
        "lock_model_size": False,
        "lock_channels": False,
        "step": 0,
        "total_steps": 0,
        "epochs": 0,
        "source": "",
        "target": "",
    }


def load_weights_into_model(
    net: nn.Module, weight_source: str, model_size: str = "Medium", device="cpu"
):
    """Load initial weights from a WildCat checkpoint (.pt) or TorchScript (.pt) file."""
    if not weight_source or weight_source == "None":
        return False
    weight_source = _resolve_weight_path(weight_source)

    pt_path = None
    if os.path.isfile(weight_source):
        pt_path = weight_source

    if not pt_path or not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Initial weights not found: {weight_source}")

    import zipfile

    is_ts_archive = False
    if zipfile.is_zipfile(pt_path):
        with zipfile.ZipFile(pt_path, "r") as zf:
            is_ts_archive = any(n.endswith("constants.pkl") for n in zf.namelist())

    if not is_ts_archive:
        try:
            ck = torch.load(pt_path, map_location=device, weights_only=False)
            if isinstance(ck, dict):
                sd = ck["G"] if "G" in ck else ck
                net.load_state_dict(sd, strict=True)
                return ck
        except Exception:
            pass

    sd_np, _ = cat_format.load_pt_state_dict_numpy(pt_path)
    _apply_numpy_state_dict(net, sd_np, device)
    return True


def _apply_numpy_state_dict(net: nn.Module, sd_np: dict, device="cpu"):
    cur_sd = net.state_dict()
    new_sd = {}
    for k, target_tensor in cur_sd.items():
        if k in sd_np:
            src_arr = sd_np[k]
            src_t = torch.from_numpy(src_arr).to(
                device=device, dtype=target_tensor.dtype
            )
            if src_t.shape == target_tensor.shape:
                new_sd[k] = src_t
            else:
                adapted = target_tensor.clone()
                c0 = min(adapted.shape[0], src_t.shape[0])
                c1 = min(adapted.shape[1], src_t.shape[1]) if adapted.ndim > 1 else 0
                if adapted.ndim == 4:
                    adapted[:c0, :c1, : src_t.shape[2], : src_t.shape[3]] = src_t[
                        :c0, :c1
                    ]
                elif adapted.ndim == 1:
                    adapted[:c0] = src_t[:c0]
                new_sd[k] = adapted
        else:
            new_sd[k] = target_tensor
    net.load_state_dict(new_sd, strict=True)
