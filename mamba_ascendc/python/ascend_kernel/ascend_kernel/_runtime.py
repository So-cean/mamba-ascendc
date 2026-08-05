"""Runtime discovery for libraries and custom OPP files bundled in the wheel."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Dict


_PACKAGE_ROOT = Path(__file__).resolve().parent
_VENDOR_ROOT = _PACKAGE_ROOT / "opp" / "vendors" / "customize"
_OP_API_LIBRARY = _VENDOR_ROOT / "op_api" / "lib" / "libcust_opapi.so"
_EXTENSION_LIBRARY = _PACKAGE_ROOT / "lib" / "libascend_kernel.so"
_OP_API_HANDLE = None


def _prepend_env_path(name: str, path: Path) -> None:
    value = str(path)
    current = [entry for entry in os.environ.get(name, "").split(":") if entry]
    if value not in current:
        os.environ[name] = ":".join([value, *current])


def get_custom_opp_path() -> str:
    """Return the bundled CANN custom OPP vendor directory."""
    return str(_VENDOR_ROOT)


def get_op_api_library_path() -> str:
    """Return the bundled custom ACLNN API library."""
    return str(_OP_API_LIBRARY)


def configure_runtime() -> None:
    """Register and preload the wheel-bundled custom operator runtime."""
    global _OP_API_HANDLE

    missing = [
        str(path)
        for path in (_VENDOR_ROOT, _OP_API_LIBRARY, _EXTENSION_LIBRARY)
        if not path.exists()
    ]
    if missing:
        raise ImportError(
            "mamba-ascendc wheel is incomplete; missing runtime artifact(s): "
            + ", ".join(missing)
        )

    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", _VENDOR_ROOT)
    os.environ.setdefault("MAMBA_CHUNK_MIX_OP_API_LIB", str(_OP_API_LIBRARY))
    os.environ.setdefault("MAMBA_ASCENDC_CHUNK_MIX", "1")
    os.environ.setdefault("MAMBA_ASCENDC_CHUNK128", "1")
    os.environ.setdefault("MAMBA_ASCENDC_NATIVE_PREPROCESS", "0")

    if _OP_API_HANDLE is None:
        try:
            _OP_API_HANDLE = ctypes.CDLL(
                str(_OP_API_LIBRARY), mode=ctypes.RTLD_GLOBAL
            )
        except OSError as error:
            raise ImportError(
                "failed to load the bundled libcust_opapi.so; activate a compatible "
                "CANN/torch_npu environment before importing ascend_kernel"
            ) from error


def runtime_info() -> Dict[str, str]:
    """Return resolved package runtime paths for diagnostics."""
    return {
        "package_root": str(_PACKAGE_ROOT),
        "custom_opp_path": str(_VENDOR_ROOT),
        "op_api_library": str(_OP_API_LIBRARY),
        "extension_library": str(_EXTENSION_LIBRARY),
    }
