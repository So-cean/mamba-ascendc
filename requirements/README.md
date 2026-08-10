# Dependency sets

The repository does not install an accelerator runtime implicitly. Install a
version-matched platform stack first, then install the corresponding Python
dependencies:

```bash
# CPU reference and repository development
python -m pip install -r requirements/dev.txt

# NVIDIA GPU benchmark; install CUDA PyTorch first
python -m pip install -r requirements/gpu.txt

# Ascend NPU; install CANN, torch, torch_npu, and Triton-Ascend first
python -m pip install -r requirements/npu.txt
```

The exact CI environment is intentionally pinned in
`.github/requirements-ci.txt` so README SVG regeneration remains deterministic.
It is not a runtime dependency lock file.
