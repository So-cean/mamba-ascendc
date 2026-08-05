"""Minimal pure Vision-Mamba2 network with pluggable SSD backends.

The module deliberately keeps the vision network independent from CUDA/NPU
packages.  ``SsdCore`` resolves the selected backend lazily, so the same model
definition can be instantiated with identical CPU-created weights on CUDA and
Ascend NPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_sequence(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
    length = x.shape[1]
    padded_length = ((length + multiple - 1) // multiple) * multiple
    if padded_length == length:
        return x.contiguous(), length
    shape = (x.shape[0], padded_length, *x.shape[2:])
    padded = x.new_zeros(shape)
    padded[:, :length].copy_(x)
    return padded, length


class SsdCore(nn.Module):
    """Backend-neutral ``mamba_chunk_scan_combined`` call boundary."""

    def __init__(self, backend: str = "auto") -> None:
        super().__init__()
        if backend not in {"auto", "cuda", "npu", "reference"}:
            raise ValueError(f"unsupported SSD backend: {backend}")
        self.backend = backend
        self._op: Callable | None = None

    def _resolve(self, device_type: str) -> Callable:
        backend = self.backend
        if backend == "auto":
            backend = "npu" if device_type == "npu" else "cuda"
        if backend == "cuda":
            from mamba_ssm.ops.triton.ssd_combined import (
                mamba_chunk_scan_combined,
            )

            return mamba_chunk_scan_combined
        if backend == "npu":
            import ascend_kernel

            return ascend_kernel.mamba2_ssd_fwd
        from mamba_torch.ssd_reference import ssd_chunk_scan_ref

        return ssd_chunk_scan_ref

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        chunk_size: int,
        D: torch.Tensor,
        z: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> torch.Tensor:
        if self._op is None:
            self._op = self._resolve(x.device.type)
        output = self._op(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
        )
        return output[0] if isinstance(output, tuple) else output


class CrossScan2d(nn.Module):
    """Run row/column forward and reverse Mamba scans and merge the outputs."""

    def __init__(
        self, backend: str, directions: int = 4, streaming_merge: bool = True
    ) -> None:
        super().__init__()
        if directions not in {1, 2, 4}:
            raise ValueError("directions must be 1, 2, or 4")
        self.directions = directions
        self.streaming_merge = streaming_merge
        self.ssd = SsdCore(backend)

    @staticmethod
    def _column_order(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return (
            x.reshape(x.shape[0], height, width, *x.shape[2:])
            .transpose(1, 2)
            .reshape(x.shape[0], height * width, *x.shape[2:])
            .contiguous()
        )

    @staticmethod
    def _restore_column(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return (
            x.reshape(x.shape[0], width, height, *x.shape[2:])
            .transpose(1, 2)
            .reshape(x.shape[0], height * width, *x.shape[2:])
            .contiguous()
        )

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        chunk_size: int,
        D: torch.Tensor,
        z: torch.Tensor,
        dt_bias: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if x.shape[1] != height * width:
            raise ValueError("sequence length does not match the 2D resolution")

        transforms: list[
            tuple[
                Callable[[torch.Tensor], torch.Tensor],
                Callable[[torch.Tensor], torch.Tensor],
            ]
        ] = [(lambda value: value, lambda value: value)]
        if self.directions >= 2:
            transforms.append(
                (lambda value: value.flip(1).contiguous(),
                 lambda value: value.flip(1).contiguous())
            )
        if self.directions == 4:
            transforms.extend(
                [
                    (
                        lambda value: self._column_order(value, height, width),
                        lambda value: self._restore_column(value, height, width),
                    ),
                    (
                        lambda value: self._column_order(
                            value, height, width
                        ).flip(1).contiguous(),
                        lambda value: self._restore_column(
                            value.flip(1).contiguous(), height, width
                        ),
                    ),
                ]
            )

        outputs = []
        merged = None
        for transform, restore in transforms:
            x_scan, real_length = _pad_sequence(transform(x), chunk_size)
            dt_scan, _ = _pad_sequence(transform(dt), chunk_size)
            b_scan, _ = _pad_sequence(transform(B), chunk_size)
            c_scan, _ = _pad_sequence(transform(C), chunk_size)
            z_scan, _ = _pad_sequence(transform(z), chunk_size)
            y = self.ssd(
                x_scan,
                dt_scan,
                A,
                b_scan,
                c_scan,
                chunk_size,
                D,
                z_scan,
                dt_bias,
            )
            restored = restore(y[:, :real_length])
            if self.streaming_merge:
                if merged is None:
                    merged = restored
                else:
                    merged.add_(restored)
            else:
                outputs.append(restored)
        if self.streaming_merge:
            return merged.mul_(1.0 / len(transforms))
        return torch.stack(outputs, dim=0).mean(dim=0)


class VisionMamba2Mixer(nn.Module):
    """A 2D Mamba-2 mixer whose only sequence operator is SSD."""

    def __init__(
        self,
        dim: int,
        *,
        expand: int = 2,
        headdim: int = 64,
        dstate: int = 128,
        ngroups: int = 1,
        chunk_size: int = 128,
        directions: int = 4,
        streaming_merge: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        inner_dim = dim * expand
        if inner_dim % headdim:
            raise ValueError("dim * expand must be divisible by headdim")
        nheads = inner_dim // headdim
        if nheads % ngroups:
            raise ValueError("nheads must be divisible by ngroups")

        self.dim = dim
        self.inner_dim = inner_dim
        self.headdim = headdim
        self.dstate = dstate
        self.ngroups = ngroups
        self.nheads = nheads
        self.chunk_size = chunk_size

        projection_dim = 2 * inner_dim + 2 * ngroups * dstate + nheads
        self.in_proj = nn.Linear(dim, projection_dim, bias=False)
        conv_dim = inner_dim + 2 * ngroups * dstate
        self.dwconv = nn.Conv2d(
            conv_dim, conv_dim, kernel_size=3, padding=1,
            groups=conv_dim, bias=True,
        )
        self.A_log = nn.Parameter(torch.empty(nheads))
        self.dt_bias = nn.Parameter(torch.empty(nheads))
        self.D = nn.Parameter(torch.ones(nheads, headdim))
        self.out_norm = nn.LayerNorm(inner_dim)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)
        self.scan = CrossScan2d(backend, directions, streaming_merge)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.A_log.uniform_(math.log(1.0), math.log(16.0))
            dt = torch.exp(
                torch.rand_like(self.dt_bias)
                * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            )
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, u: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, length, _ = u.shape
        if length != height * width:
            raise ValueError("mixer input does not match height * width")
        projected = self.in_proj(u)
        z, xbc, dt = torch.split(
            projected,
            [
                self.inner_dim,
                self.inner_dim + 2 * self.ngroups * self.dstate,
                self.nheads,
            ],
            dim=-1,
        )
        # Materialize independent scan inputs before the convolution replaces
        # ``xbc``.  Otherwise the split views keep the much larger projection
        # storage alive throughout all four directional scans.
        z = z.reshape(batch, length, self.nheads, self.headdim).contiguous()
        dt = dt.contiguous()
        xbc = xbc.reshape(batch, height, width, -1).permute(0, 3, 1, 2)
        xbc = F.silu(self.dwconv(xbc))
        xbc = xbc.permute(0, 2, 3, 1).reshape(batch, length, -1).contiguous()
        del projected
        x, B, C = torch.split(
            xbc,
            [self.inner_dim, self.ngroups * self.dstate,
             self.ngroups * self.dstate],
            dim=-1,
        )
        x = x.reshape(batch, length, self.nheads, self.headdim).contiguous()
        B = B.reshape(batch, length, self.ngroups, self.dstate).contiguous()
        C = C.reshape(batch, length, self.ngroups, self.dstate).contiguous()
        del xbc
        A = (-torch.exp(self.A_log.float())).contiguous()
        y = self.scan(
            x,
            dt,
            A,
            B,
            C,
            self.chunk_size,
            self.D,
            z,
            self.dt_bias,
            height,
            width,
        )
        y = y.reshape(batch, length, self.inner_dim)
        return self.out_proj(self.out_norm(y))


class FeedForward(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisionMamba2Block(nn.Module):
    def __init__(self, dim: int, **mixer_kwargs) -> None:
        super().__init__()
        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = VisionMamba2Mixer(dim, **mixer_kwargs)
        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim)

    @staticmethod
    def _cpe(x: torch.Tensor, conv: nn.Conv2d, h: int, w: int) -> torch.Tensor:
        feature = x.reshape(x.shape[0], h, w, x.shape[-1]).permute(0, 3, 1, 2)
        return conv(feature).flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self._cpe(x, self.cpe1, h, w)
        x = x + self.mixer(self.norm1(x), h, w)
        x = x + self._cpe(x, self.cpe2, h, w)
        return x + self.ffn(self.norm2(x))


class PatchDownsample(nn.Module):
    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self, x: torch.Tensor, h: int, w: int
    ) -> tuple[torch.Tensor, int, int]:
        x = x.reshape(x.shape[0], h, w, x.shape[-1]).permute(0, 3, 1, 2)
        x = self.conv(x)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), h, w


@dataclass(frozen=True)
class VisionMamba2Config:
    image_size: int = 256
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 1000
    dims: tuple[int, ...] = (96, 192, 384, 768)
    depths: tuple[int, ...] = (2, 2, 6, 2)
    ngroups: tuple[int, ...] = (1, 2, 4, 8)
    chunk_sizes: tuple[int, ...] = (128, 128, 128, 64)
    expand: int = 2
    headdim: int = 64
    dstate: int = 128
    directions: int = 4
    streaming_merge: bool = True


class PureVisionMamba2(nn.Module):
    """Hierarchical vision network with Mamba-2 in every mixer block."""

    def __init__(self, config: VisionMamba2Config, backend: str = "auto") -> None:
        super().__init__()
        count = len(config.dims)
        if not (
            len(config.depths) == len(config.ngroups)
            == len(config.chunk_sizes) == count
        ):
            raise ValueError("stage configuration lengths must match")
        self.config = config
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.dims[0],
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for stage, (dim, depth, groups, chunk) in enumerate(
            zip(config.dims, config.depths, config.ngroups, config.chunk_sizes)
        ):
            blocks = nn.ModuleList(
                [
                    VisionMamba2Block(
                        dim,
                        expand=config.expand,
                        headdim=config.headdim,
                        dstate=config.dstate,
                        ngroups=groups,
                        chunk_size=chunk,
                        directions=config.directions,
                        streaming_merge=config.streaming_merge,
                        backend=backend,
                    )
                    for _ in range(depth)
                ]
            )
            self.stages.append(blocks)
            if stage + 1 < count:
                self.downsamples.append(
                    PatchDownsample(dim, config.dims[stage + 1])
                )
        self.final_norm = nn.LayerNorm(config.dims[-1])
        self.head = nn.Linear(config.dims[-1], config.num_classes)

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(image)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        for stage, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x, h, w)
            if stage < len(self.downsamples):
                x, h, w = self.downsamples[stage](x, h, w)
        return self.final_norm(x).mean(dim=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(image))


def fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    npu_explicit: bool = False,
) -> torch.Tensor:
    """SDPA adapter using the dedicated NPU FA interface when requested.

    Inputs use BNSD layout.  Native ``scaled_dot_product_attention`` is the
    default because torch_npu can route supported training shapes to FA.  The
    explicit path follows the documented ``npu_fusion_attention`` mapping.
    """

    scale = 1.0 / math.sqrt(q.shape[-1])
    if q.device.type != "npu" or not npu_explicit:
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=causal, scale=scale
        )

    import torch_npu

    attention_mask = None
    sparse_mode = 0
    if causal:
        length = q.shape[-2]
        attention_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        sparse_mode = 2
    return torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        q.shape[1],
        input_layout="BNSD",
        pse=None,
        atten_mask=attention_mask,
        scale=scale,
        pre_tockens=2147483647,
        next_tockens=2147483647,
        keep_prob=1.0,
        sparse_mode=sparse_mode,
    )[0]


def count_mamba_modules(model: nn.Module) -> dict[str, int]:
    return {
        "mixer_blocks": sum(isinstance(m, VisionMamba2Mixer) for m in model.modules()),
        "ssd_cores": sum(isinstance(m, SsdCore) for m in model.modules()),
    }


__all__ = [
    "CrossScan2d",
    "PureVisionMamba2",
    "SsdCore",
    "VisionMamba2Block",
    "VisionMamba2Config",
    "VisionMamba2Mixer",
    "count_mamba_modules",
    "fused_attention",
]
