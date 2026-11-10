# -*- coding: utf-8 -*-
"""TorchScript (.pt) reading and export helpers."""
import io
import os
import pickle
import shutil
import struct
import subprocess
import tempfile
import zipfile
from collections import OrderedDict



class _DummyObj:
    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state


class _TorchScriptUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "_rebuild_tensor_v2":
            return lambda storage, offset, size, stride, req_grad, hooks: (
                "TENSOR",
                storage[2],
                size,
            )
        if name == "OrderedDict":
            return OrderedDict
        return type(f"{module}.{name}", (_DummyObj,), {})

    def persistent_load(self, pid):
        return pid


def _walk_tensors(obj, prefix=""):
    out = []
    if isinstance(obj, _DummyObj) and hasattr(obj, "state"):
        for k, v in obj.state.items():
            if k == "_modules" and isinstance(v, dict):
                for mk, mv in v.items():
                    out.extend(_walk_tensors(mv, prefix + mk + "."))
            elif isinstance(v, tuple) and len(v) == 3 and v[0] == "TENSOR":
                out.append((prefix + k, v[1], v[2]))
            elif isinstance(v, _DummyObj):
                out.extend(_walk_tensors(v, prefix + k + "."))
    return out


def inspect_pt_archive(pt_path):
    """Parse TorchScript .pt ZIP archive metadata & parameter storage map without requiring torch.jit."""
    with zipfile.ZipFile(pt_path, "r") as z:
        dp = [fn for fn in z.namelist() if fn.endswith("data.pkl")][0]
        archive_prefix = dp[: -len("data.pkl")]
        obj = _TorchScriptUnpickler(io.BytesIO(z.read(dp))).load()
        st = getattr(obj, "state", {})
        tensors = _walk_tensors(obj)
        storage_map = OrderedDict()
        for tname, sid, shape in tensors:
            if sid not in storage_map:
                storage_map[sid] = {"name": tname, "shape": shape, "aliases": [tname]}
            else:
                storage_map[sid]["aliases"].append(tname)
        return {
            "archive_prefix": archive_prefix,
            "channels_in": st.get("channels_in", 3),
            "channels_out": st.get("channels_out", 3),
            "n_levels": st.get("n_levels", 3),
            "scale": st.get("scale", 0.5),
            "k_size": st.get("k_size", 5),
            "storage_map": storage_map,
        }


def load_pt_state_dict_numpy(pt_path):
    """Load state_dict as {name: np.ndarray(float32)} directly from a TorchScript .pt ZIP archive."""
    import numpy as np

    meta = inspect_pt_archive(pt_path)
    prefix = meta["archive_prefix"]
    state = OrderedDict()
    with zipfile.ZipFile(pt_path, "r") as z:
        for sid, info in meta["storage_map"].items():
            raw = z.read(f"{prefix}data/{sid}")
            arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(info["shape"])
            for alias in info["aliases"]:
                state[alias] = arr
    return state, meta




def export_torchscript(state_dict, out_pt, mo=None):
    """Write a TorchScript (.pt) model from a WildCat state_dict.

    The file can be converted to a .cat with Nuke's own CatFileCreator node
    (Torch Script File = this file, Channels In / Out = rgba.red, rgba.green, rgba.blue)
    and then used in the Inference node. Nuke 17 bundles libtorch 2.7: files written by
    PyTorch newer than 2.11 may not load there, so export with PyTorch <= 2.11.
    """
    import torch
    from .models import build_generator

    net, _ = build_generator(mo or {})
    net.load_state_dict({k: (v.detach().cpu() if hasattr(v, "detach") else torch.from_numpy(v)) for k, v in state_dict.items()})
    net.eval()
    scripted = torch.jit.script(net)
    os.makedirs(os.path.dirname(os.path.abspath(out_pt)), exist_ok=True)
    tmp = out_pt + ".tmp"
    scripted.save(tmp)
    os.replace(tmp, out_pt)
    return out_pt


def torch_version_ok_for_nuke():
    """True when the running PyTorch writes TorchScript files Nuke 17 can read (<= 2.11)."""
    import torch
    try:
        major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    except ValueError:
        return True
    return (major, minor) <= (2, 11)
