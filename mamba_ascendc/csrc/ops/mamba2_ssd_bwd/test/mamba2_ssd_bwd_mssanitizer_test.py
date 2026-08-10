"""Python-mode mssanitizer carrier for the native Mamba2 SSD backward."""

from __future__ import annotations

import argparse
import json

import torch

import ascend_kernel


def run_case(shape):
    batch, seqlen, nheads, headdim, dstate, ngroups = shape
    x = torch.rand(batch, seqlen, nheads, headdim, device="npu")
    dt = 0.002 + 0.004 * torch.rand(batch, seqlen, nheads, device="npu")
    A = -(0.005 + 0.005 * torch.rand(nheads, device="npu"))
    B = 0.05 + 0.1 * torch.rand(batch, seqlen, ngroups, dstate, device="npu")
    C = 0.05 + 0.1 * torch.rand(batch, seqlen, ngroups, dstate, device="npu")
    out, final_state = ascend_kernel.mamba2_ssd_fwd(
        x, dt, A, B, C, 64, return_final_state=True
    )
    result = torch.ops.mamba_ascend.mamba2_ssd_bwd(
        x, dt, A, B, C, torch.ones_like(out), final_state,
        None, None, None, None, torch.ones_like(final_state),
        False, 0.0, torch.finfo(torch.float32).max,
    )
    torch.npu.synchronize()
    return [list(tensor.shape) for tensor in result]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = [(1, 64, 1, 64, 64, 1), (1, 64, 4, 64, 64, 2)]
    results = []
    for shape in cases:
        try:
            outputs = run_case(shape)
            results.append({"shape": list(shape), "status": "PASS", "outputs": outputs})
        except Exception as error:
            results.append({"shape": list(shape), "status": "FAIL", "error": str(error)})
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"results": results}, handle, indent=2)
    if any(item["status"] != "PASS" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
