# Copyright (c) 2024, Tri Dao, Albert Gu.
# Public Mamba-2 SSD forward entry point for Ascend adaptation.

from mamba_torch.ssd_reference import ssd_chunk_scan_ref


def mamba_chunk_scan_combined(
    x, dt, A, B, C, chunk_size,
    D=None, z=None, seq_idx=None,
    dt_bias=None, dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    initial_states=None,
    return_final_states=False,
    return_final_state=None,
    backend=None,
    **kwargs,
):
    """Dispatch Mamba-2 SSD forward to Triton-Ascend or CPU reference.

    NPU tensors always use the Triton implementation. CPU tensors use the
    PyTorch reference. ``backend`` can be set to ``"triton"`` or
    ``"reference"`` to make the choice explicit in tests.
    """
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported arguments: {unsupported}")
    if return_final_state is not None:
        if return_final_states and not return_final_state:
            raise ValueError("conflicting final-state flags")
        return_final_states = return_final_state

    selected_backend = backend or ("triton" if x.device.type == "npu" else "reference")
    if selected_backend == "triton":
        if x.device.type != "npu":
            raise ValueError(f"Triton-Ascend requires NPU inputs, got {x.device}")
        from .triton.ssd_combined import mamba_chunk_scan_combined as triton_forward

        return triton_forward(
            x, dt, A, B, C, chunk_size,
            D=D, z=z, dt_bias=dt_bias, initial_states=initial_states,
            seq_idx=seq_idx, dt_softplus=dt_softplus, dt_limit=dt_limit,
            return_final_states=return_final_states,
        )
    if selected_backend != "reference":
        raise ValueError(f"unknown backend: {selected_backend}")
    if seq_idx is not None:
        raise NotImplementedError("CPU reference does not yet support seq_idx")

    return ssd_chunk_scan_ref(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        chunk_size=chunk_size,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        initial_states=initial_states,
        return_final_state=return_final_states,
    )


def mamba_split_conv1d_scan_combined(
    zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A,
    D=None, chunk_size=256, seq_idx=None,
    activation="silu",
    rmsnorm_weight=None, rmsnorm_eps=1e-5,
    outproj_weight=None, outproj_bias=None,
    headdim=64, ngroups=1, norm_before_gate=False,
    initial_states=None, dt_limit=(0.0, float("inf")),
    **kwargs,
):
    """PyTorch fallback for the fully-fused path.

    Since causal_conv1d and Triton RMSNorm are not yet available on NPU,
    we decompose into PyTorch ops.
    """
    # Infer dimensions
    batch, seqlen, _ = zxbcdt.shape
    d_in_proj = zxbcdt.shape[-1]

    # Split: [z, x, B, C, dt]
    # Need to figure out split sizes from headdim and ngroups
    # This is the same logic as Mamba2Simple forward
    # But since we don't have d_inner here directly, we need to derive
    # For simplicity in fallback, we just do the slow path manually
    raise NotImplementedError(
        "mamba_split_conv1d_scan_combined fallback not yet implemented. "
        "Use Mamba2Simple with use_mem_eff_path=False for now."
    )
