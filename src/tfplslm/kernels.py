"""Import the pinned upstream Triton SSD kernels without optional CUDA extensions.

The namespace packages deliberately skip upstream's top-level __init__, which
imports the Mamba-1 compiled extension. No upstream kernel is modified.
"""
import sys
import types
from pathlib import Path


def load_ssd():
    root = Path(__file__).resolve().parents[2] / "vendor/mamba/mamba_ssm"
    if not root.is_dir():
        raise RuntimeError("Run bash scripts/setup_server.sh to fetch pinned Mamba kernels")
    for name, path in [("mamba_ssm", root), ("mamba_ssm.ops", root / "ops"),
                       ("mamba_ssm.ops.triton", root / "ops/triton"),
                       ("mamba_ssm.utils", root / "utils")]:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    return mamba_chunk_scan_combined
