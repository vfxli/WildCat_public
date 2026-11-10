# -*- coding: utf-8 -*-
"""WildCat multi-scale coarse-to-fine EncoderDecoder neural architecture (Small, Medium, Large)."""
import math
import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cat_format

MODEL_SPECS = {
    "Small": {"k_size": 3, "n_resblocks": 1, "params": 1_033_955},
    "Medium": {"k_size": 5, "n_resblocks": 1, "params": 2_578_659},
    "Large": {"k_size": 5, "n_resblocks": 3, "params": 6_881_251},
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


# Alias for backward compatibility
WildCatNet = EncoderDecoder


def build_generator(mo: dict, cudnn_pad: bool = False):
    model_size = mo.get("model_size", "Medium")
    if model_size not in MODEL_SPECS:
        model_size = "Medium"
    spec = MODEL_SPECS[model_size]
    c_in = int(mo.get("in_channels", 3))
    c_out = int(mo.get("out_channels", c_in))
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

    pt_path = weight_source
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
                        c1_shape = sd["conv1_1.weight"][2]
                        d3_shape = sd["deconv3_1.weight"][2]
                        k_size = int(c1_shape[2])
                        out_ch = int(d3_shape[1])
                        in_ch = int(c1_shape[1]) - out_ch
                        is_large = "block1_3.conv2d_1.weight" in sd
                        ms = "Large" if is_large else ("Small" if k_size == 3 else "Medium")
                        cfg = obj.get("config") or {}
                        return {
                            "model_size": ms,
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
                        c1_shape = tensors.get("conv1_1.weight", (32, 6, 5, 5))
                        d3_shape = tensors.get("deconv3_1.weight", (32, 3, 5, 5))
                        k_size = int(st.get("k_size", c1_shape[2]))
                        out_ch = int(st.get("channels_out", d3_shape[1]))
                        in_ch = int(st.get("channels_in", c1_shape[1] - out_ch))
                        is_large = "block1_3.conv2d_1.weight" in tensors
                        ms = "Large" if is_large else ("Small" if k_size == 3 else "Medium")
                        return {
                            "model_size": ms,
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
