#!/usr/bin/env python3
"""Run one test file against a source-built extension and custom OPP.

This development helper deliberately bypasses the installed ``ascend_kernel``
wheel.  Set ``MAMBA2_TEST_EXTENSION_LIB`` to ``libascend_kernel.so`` and
``MAMBA2_TEST_OPP_ROOT`` to the source-built OPP install directory.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
from pathlib import Path
import runpy
import sys
import types


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file", type=Path)
    parser.add_argument(
        "test_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to the target test or benchmark script",
    )
    args = parser.parse_args()

    extension = Path(os.environ["MAMBA2_TEST_EXTENSION_LIB"]).resolve()
    opp_root = Path(os.environ["MAMBA2_TEST_OPP_ROOT"]).resolve()
    vendor_root = opp_root / "vendors" / "customize"
    op_api = vendor_root / "op_api" / "lib" / "libcust_opapi.so"
    missing = [path for path in (extension, vendor_root, op_api) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing source candidate artifact(s): "
            + ", ".join(str(path) for path in missing)
        )

    current = [
        entry
        for entry in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":")
        if entry
    ]
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(
        [str(vendor_root), *[entry for entry in current if entry != str(vendor_root)]]
    )
    os.environ["MAMBA_CHUNK_MIX_OP_API_LIB"] = str(op_api)

    # OPP discovery happens while torch_npu initializes, so candidate paths
    # must be installed before either torch_npu or the extension is imported.
    import torch
    import torch_npu  # noqa: F401

    ctypes.CDLL(str(op_api), mode=ctypes.RTLD_GLOBAL)
    torch.ops.load_library(str(extension))

    # Expose the source-tree Python API without importing the installed wheel.
    # Registration was performed explicitly above, while loading mamba2.py
    # directly keeps both API tests and operator-only tests on this candidate.
    package = types.ModuleType("ascend_kernel")
    mamba2_path = (
        Path(__file__).resolve().parents[1]
        / "python"
        / "ascend_kernel"
        / "ascend_kernel"
        / "mamba2.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_mamba2_source_candidate", mamba2_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source API from {mamba2_path}")
    mamba2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mamba2)
    package.mamba_chunk_scan_combined = mamba2.mamba_chunk_scan_combined
    package.mamba2_ssd_fwd = mamba2.mamba2_ssd_fwd
    sys.modules["ascend_kernel"] = package
    target = args.test_file.resolve()
    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), *args.test_args]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
