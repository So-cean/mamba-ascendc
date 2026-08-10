import hashlib
import os
from pathlib import Path

import torch
import torch.nn.functional as F

import ascend_kernel  # noqa: F401


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _print_runtime_hashes():
    repo = Path(os.environ.get(
        "MAMBA_ASCEND_REPO",
        str(Path(__file__).resolve().parents[3]),
    ))
    package = Path(ascend_kernel.__file__).resolve().parent
    opp_build = repo / "mamba_ascendc_ops/mamba2_ssd_chunk_mix/build_out"
    deploy_package = (
        opp_build / "_CPack_Packages/Linux/External/"
        "custom_opp_ubuntu_x86_64.run/packages/vendors/customize"
    )
    loaded_kernel = (
        package / "opp/vendors/customize/op_impl/ai_core/tbe/kernel/ascend950"
    )
    candidates = {
        "build_extension": [
            repo / "mamba_ascendc/python/ascend_kernel/ascend_kernel/lib/"
            "libascend_kernel.so"
        ],
        "loaded_extension": [package / "lib/libascend_kernel.so"],
        "build_projection_kernel": list((
            opp_build / "op_kernel/binary/ascend950/"
            "mamba2_ssd_state_projection"
        ).glob("*.o")),
        "loaded_projection_kernel": list((
            loaded_kernel / "mamba2_ssd_state_projection"
        ).glob("*.o")),
        "build_vector_kernel": list((
            opp_build / "op_kernel/binary/ascend950/"
            "mamba2_ssd_state_vector_epilogue"
        ).glob("*.o")),
        "loaded_vector_kernel": list((
            loaded_kernel / "mamba2_ssd_state_vector_epilogue"
        ).glob("*.o")),
        "build_tiling": [
            deploy_package / "op_impl/ai_core/tbe/op_tiling/lib/linux/"
            "x86_64/libcust_opmaster_rt2.0.so"
        ],
        "loaded_tiling": [
            package / "opp/vendors/customize/op_impl/ai_core/tbe/op_tiling/"
            "lib/linux/x86_64/libcust_opmaster_rt2.0.so"
        ],
        "build_opapi": [deploy_package / "op_api/lib/libcust_opapi.so"],
        "loaded_opapi": [
            package / "opp/vendors/customize/op_api/lib/libcust_opapi.so"
        ],
    }
    print("ascend_kernel_extension", ascend_kernel.__file__, flush=True)
    print("ASCEND_CUSTOM_OPP_PATH", os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
          flush=True)
    hashes = {}
    for label, paths in candidates.items():
        existing = [path for path in paths if path.is_file()]
        assert existing, f"missing runtime hash target: {label}"
        unique_hashes = {_sha256(path) for path in existing}
        assert len(unique_hashes) == 1, (
            f"ambiguous build/runtime provenance for {label}: {existing}"
        )
        hashes[label] = unique_hashes.pop()
        for path in existing:
            print(f"HASH {label} {_sha256(path)} {path}", flush=True)
    for build_label, loaded_label in (
        ("build_extension", "loaded_extension"),
        ("build_projection_kernel", "loaded_projection_kernel"),
        ("build_vector_kernel", "loaded_vector_kernel"),
        ("build_tiling", "loaded_tiling"),
        ("build_opapi", "loaded_opapi"),
    ):
        assert hashes[build_label] == hashes[loaded_label], (
            f"provenance mismatch: {build_label}={hashes[build_label]} "
            f"{loaded_label}={hashes[loaded_label]}"
        )
    return hashes


PROVENANCE_HASHES = _print_runtime_hashes()


def _metrics(actual, expected):
    error = actual.float() - expected.float()
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    return (
        error.abs().max().item(),
        (torch.linalg.vector_norm(error) / denominator).item(),
        F.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        ).item(),
    )


def _projection_reference(states_start, c_cube, groups):
    batch, heads, chunks, _, _ = states_start.shape
    heads_per_group = heads // groups
    result = []
    for chunk in range(chunks):
        c_chunk = c_cube[:, chunk].repeat_interleave(heads_per_group, dim=1)
        state_t = states_start[:, :, chunk].transpose(-1, -2).half()
        result.append(torch.matmul(c_chunk, state_t).float())
    return torch.stack(result, dim=2)


def test_state_projection_h1_k1_patterns():
    torch.manual_seed(20260807)
    state = torch.randn(1, 1, 1, 64, 64, device="npu") * 0.1
    for pattern in ("zero", "identity", "random"):
        if pattern == "zero":
            c_cube = torch.zeros(1, 1, 1, 64, 64,
                                 dtype=torch.float16, device="npu")
        elif pattern == "identity":
            c_cube = torch.eye(64, dtype=torch.float16, device="npu")
            c_cube = c_cube.reshape(1, 1, 1, 64, 64)
        else:
            c_cube = (torch.randn(1, 1, 1, 64, 64, device="npu") * 0.1).half()
        expected = _projection_reference(state, c_cube, 1)
        repeats = []
        for repetition in range(3):
            actual = torch.ops.mamba_ascend.mamba2_ssd_state_projection(
                state, c_cube
            )
            torch.npu.synchronize()
            repeats.append(actual.clone())
            metrics = _metrics(actual, expected)
            print(
                f"projection {pattern} repetition={repetition}: {metrics}",
                flush=True,
            )
            assert torch.isfinite(actual).all()
            if pattern == "zero":
                assert metrics[0] <= 1e-6
                assert torch.allclose(actual, expected, atol=1e-6, rtol=0.0)
            else:
                assert metrics[0] <= 5e-2
                assert metrics[1] <= 5e-3
                assert metrics[2] >= 0.999
        for repeat in repeats[1:]:
            assert torch.equal(repeat, repeats[0])


def test_state_projection_workspace_reuse():
    # 64 logical tasks exceed the 20 Cube cores on the current 950PR and also
    # exercise B>1, G>1 and per-core workspace reuse in one invocation.
    torch.manual_seed(20260808)
    batch, heads, groups, chunks = 2, 4, 2, 8
    state = 0.1 * torch.randn(
        batch, heads, chunks, 64, 64, device="npu"
    )
    c_cube = (0.1 * torch.randn(
        batch, chunks, groups, 64, 64, device="npu"
    )).half()
    expected = _projection_reference(state, c_cube, groups)
    repeats = []
    for repetition in range(3):
        actual = torch.ops.mamba_ascend.mamba2_ssd_state_projection(
            state, c_cube
        )
        torch.npu.synchronize()
        repeats.append(actual.clone())
        metrics = _metrics(actual, expected)
        print(f"projection workspace repetition={repetition}: {metrics}",
              flush=True)
        assert torch.isfinite(actual).all()
        assert metrics[0] <= 5e-2
        assert metrics[1] <= 5e-3
        assert metrics[2] >= 0.999
    for repeat in repeats[1:]:
        assert torch.equal(repeat, repeats[0])


def _full_reference(chunk_states, da, c_cube, y_diag, x, d, z, initial):
    batch, heads, chunks, head_dim, state_dim = chunk_states.shape
    groups = c_cube.shape[2]
    heads_per_group = heads // groups
    state = initial.clone()
    outputs = []
    for chunk in range(chunks):
        c_chunk = c_cube[:, chunk].repeat_interleave(heads_per_group, dim=1)
        y_off = torch.matmul(c_chunk, state.transpose(-1, -2).half()).float()
        y = y_diag[:, :, chunk] + y_off * torch.exp(
            da[:, :, chunk]
        ).unsqueeze(-1)
        outputs.append(y)
        decay = torch.exp(da[:, :, chunk, -1]).unsqueeze(-1).unsqueeze(-1)
        state = decay * state + chunk_states[:, :, chunk]
    out = torch.stack(outputs, dim=2).permute(0, 2, 3, 1, 4)
    out = out.reshape(batch, chunks * 64, heads, head_dim)
    out = (out + x * d.view(1, 1, heads, head_dim)) * F.silu(z)
    return out, state


def test_state_two_launch_public_all():
    torch.manual_seed(20260807)
    batch, heads, groups, chunks = 2, 4, 2, 3
    chunk_states = 0.05 * torch.randn(
        batch, heads, chunks, 64, 64, device="npu"
    )
    da_step = -(0.001 + 0.01 * torch.rand(
        batch, heads, chunks, 64, device="npu"
    ))
    da = torch.cumsum(da_step, dim=-1)
    c_cube = (0.1 * torch.randn(
        batch, chunks, groups, 64, 64, device="npu"
    )).half()
    y_diag = 0.1 * torch.randn(
        batch, heads, chunks, 64, 64, device="npu"
    )
    x = 0.2 * torch.randn(batch, chunks * 64, heads, 64, device="npu")
    d = 0.2 * torch.randn(heads, 64, device="npu")
    z = 0.2 * torch.randn_like(x)
    initial = 0.05 * torch.randn(batch, heads, 64, 64, device="npu")

    expected, expected_final = _full_reference(
        chunk_states, da, c_cube, y_diag, x, d, z, initial
    )
    repeated_out = []
    repeated_final = []
    for repetition in range(2):
        states_start, final = torch.ops.mamba_ascend.mamba2_ssd_state_passing(
            chunk_states, da, initial
        )
        y_off = torch.ops.mamba_ascend.mamba2_ssd_state_projection(
            states_start, c_cube
        )
        actual = torch.ops.mamba_ascend.mamba2_ssd_state_vector_epilogue(
            y_off, da, y_diag, x, d, z
        )
        torch.npu.synchronize()
        repeated_out.append(actual.clone())
        repeated_final.append(final.clone())
        out_metrics = _metrics(actual, expected)
        final_metrics = _metrics(final, expected_final)
        print(
            f"public ALL repetition={repetition} out={out_metrics} "
            f"final={final_metrics}",
            flush=True,
        )
        for metrics in (out_metrics, final_metrics):
            assert metrics[0] <= 5e-2
            assert metrics[1] <= 5e-3
            assert metrics[2] >= 0.999
    assert torch.equal(repeated_out[0], repeated_out[1])
    assert torch.equal(repeated_final[0], repeated_final[1])
