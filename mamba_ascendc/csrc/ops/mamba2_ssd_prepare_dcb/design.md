# Mamba2 SSD PrepareDcb group-reduction design

## Interface and algebra

```text
mamba2_ssd_prepare_dcb(
    d_w[B,H,K,64,64] fp16,
    d_a_cumsum[B,H,K,64] fp32,
    groups: int,
) -> d_cb_group[B,G,K,64,64] fp16
```

For `heads_per_group = H/G`:

```text
d_cb_h[i,j] = d_w_h[i,j] * exp(d_a_h[i] - d_a_h[j]), i >= j
d_cb_group   = cast_fp16(sum_h_in_group(cast_fp32(d_cb_h)))
```

`B` and `C` are group-owned in Mamba2.  The only consumers of `d_cb` are
`d_cb @ B` and `d_cb^T @ C`, followed by a sum over the heads in each group.
By distributivity, reducing `d_cb` before both matmuls removes their per-head
outputs without changing the mathematical gradient.

## Mapping and layout

- one AIV task owns `(batch, group, chunk)`;
- the task streams the group's heads in blocks of at most four;
- one strided `DataCopyPad` compacts up to four 8-KiB `d_w` matrices into UB;
- `d_a_cumsum` uses the same head-block descriptor;
- Exp, causal masking and accumulation use FP32 in UB;
- only the final group sum is rounded to FP16 and written to GM.

At the H256 stress shape (`B=8,H=256,G=64,K=64`) the previous output was a
1-GiB `[B,H,K,64,64]` tensor.  The group-owned output is 256 MiB.  The two
downstream BMM outputs are reduced by the same factor.

## UB allocation

Worst case `head_block=4`:

| Buffer | Bytes |
|---|---:|
| dW FP16 head block | 32 KiB |
| dA FP32 head block | 1 KiB |
| dCB FP16 group output | 8 KiB |
| two FP32 matrix temporaries | 32 KiB |
| FP32 group accumulator | 16 KiB |
| Broadcast scratch | 8 KiB |
| **Total** | **97 KiB** |

The host queries the runtime UB size and rejects the launch when this working
set does not fit.  GM/UB transfers use `DataCopyPad`; FP16 dW is promoted to
FP32 before Exp/Mul/accumulation.

## Verification gates

1. Existing 40-case public backward precision matrix remains 40/40.
2. Direct grouped result agrees with a PyTorch FP32 group-reduction reference.
3. The H256 profile must show group-shaped dB/dC BatchMatMul inputs/outputs.
4. Keep only if H256 backward improves over the 112.297-ms quick baseline.
