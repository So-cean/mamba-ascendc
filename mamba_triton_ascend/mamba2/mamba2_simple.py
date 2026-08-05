# Copyright (c) 2024, Tri Dao, Albert Gu.
# NPU migration: host-side PyTorch ops + reference fallback for kernels.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_triton_ascend.mamba2.ops.ssd_combined import mamba_chunk_scan_combined


def causal_conv1d_pytorch(x, weight, bias=None, activation="silu"):
    """Pure PyTorch causal 1D convolution.

    Args:
        x: (batch, dim, seqlen)
        weight: (dim, width)
        bias: (dim,) or None
        activation: "silu" or "swish"
    Returns:
        (batch, dim, seqlen)
    """
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    # causal: pad left by width-1, no right pad
    x_pad = F.pad(x, (width - 1, 0))
    # Conv1d: groups=dim means each channel has its own filter
    # weight shape for Conv1d is (out_channels, in_channels/groups, kernel_size)
    # Here out_channels = dim, in_channels/groups = 1, kernel_size = width
    weight_3d = weight.unsqueeze(1)  # (dim, 1, width)
    out = F.conv1d(x_pad, weight_3d, bias=bias, groups=dim)
    if activation in ["silu", "swish"]:
        out = F.silu(out)
    return out


class Mamba2Simple(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=False,  # NPU: fused path not yet supported
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        if self.learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs)
            )
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        # NPU fallback: simple LayerNorm instead of Triton RMSNormGated
        self.norm = nn.LayerNorm(self.d_inner, eps=1e-5, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log.float())  # (nheads,)
        initial_states = (
            self.init_states.unsqueeze(0).expand(batch, -1, -1, -1)
            if self.learnable_init_states else None
        )
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path:
            raise NotImplementedError(
                "use_mem_eff_path=True (fully-fused path) is not yet supported on NPU. "
                "Please use use_mem_eff_path=False (default)."
            )

        # Slow path: decomposed into PyTorch ops
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
        assert self.activation in ["silu", "swish"]

        # 1D Causal Convolution (PyTorch fallback)
        xBC = causal_conv1d_pytorch(
            x=xBC.transpose(1, 2),
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
        ).transpose(1, 2)
        xBC = xBC[:, :seqlen, :]

        # Split into 3 main branches: X, B, C
        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )

        # Reshape for SSD chunk scan
        x = x.reshape(batch, seqlen, self.nheads, self.headdim)
        B = B.reshape(batch, seqlen, self.ngroups, self.d_state)
        C = C.reshape(batch, seqlen, self.ngroups, self.d_state)

        # Reshape z for ssd_chunk_scan_ref: (B, L, d_inner) -> (B, L, nheads, headdim)
        z_reshaped = z.reshape(batch, seqlen, self.nheads, self.headdim) if z is not None else None

        y = mamba_chunk_scan_combined(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            chunk_size=self.chunk_size,
            D=self.D,
            z=z_reshaped,
            seq_idx=seq_idx,
            initial_states=initial_states,
            **dt_limit_kwargs,
        )
        y = y.reshape(batch, seqlen, self.d_inner)

        # Extra normalization layer right before output projection
        # NPU fallback: simple LayerNorm instead of Triton RMSNormGated
        y = self.norm(y)
        out = self.out_proj(y)
        return out
