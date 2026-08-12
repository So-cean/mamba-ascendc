"""Canonical Mamba-2 forward benchmark shapes and suite definitions.

The public shape order is ``[B, L, H, P, N, chunk, G]``.  Keeping this table
in one module prevents the A100 and Ascend runners from silently benchmarking
different workloads.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ShapeCase:
    """One forward benchmark workload."""

    batch: int
    seqlen: int
    nheads: int
    headdim: int
    dstate: int
    chunk_size: int
    ngroups: int
    axis: str
    tier: str = "standard"

    @property
    def public_tuple(self) -> tuple[int, int, int, int, int, int, int]:
        """Return canonical ``B,L,H,P,N,chunk,G`` order."""
        return (
            self.batch,
            self.seqlen,
            self.nheads,
            self.headdim,
            self.dstate,
            self.chunk_size,
            self.ngroups,
        )

    @property
    def internal_tuple(self) -> tuple[int, int, int, int, int, int, int]:
        """Return the historical runner order ``B,L,H,P,N,G,chunk``."""
        return (
            self.batch,
            self.seqlen,
            self.nheads,
            self.headdim,
            self.dstate,
            self.ngroups,
            self.chunk_size,
        )

    @property
    def logical_tasks(self) -> int:
        """Return the number of logical head/chunk tasks, including a tail."""
        chunks = (self.seqlen + self.chunk_size - 1) // self.chunk_size
        return self.batch * self.nheads * chunks

    @property
    def expected_910b3_path(self) -> str:
        """Describe the expected optimized dispatch with Cube/MIX enabled."""
        aligned = (
            self.seqlen % self.chunk_size == 0
            and self.chunk_size % 16 == 0
            and self.headdim % 16 == 0
            and self.dstate % 16 == 0
            and self.headdim * self.dstate <= 8192
        )
        if not aligned:
            return "generic"

        execution_chunk = self.chunk_size
        if execution_chunk > 128:
            if self.dstate == 128 and execution_chunk % 128 == 0:
                execution_chunk = 128
            elif execution_chunk % 64 == 0:
                execution_chunk = 64
        cube_mix = (
            self.headdim == 64
            and self.dstate in (64, 128)
            and self.seqlen % 64 == 0
            and execution_chunk in (64, 128)
        )
        return "cube_mix" if cube_mix else "aligned"

    @property
    def full_feature_input_mib(self) -> float:
        """Estimate FP32 public inputs, including D/z/bias/initial state."""
        elements = (
            2 * self.batch * self.seqlen * self.nheads * self.headdim
            + self.batch * self.seqlen * self.nheads
            + self.nheads
            + 2 * self.batch * self.seqlen * self.ngroups * self.dstate
            + self.nheads * self.headdim
            + self.nheads
            + self.batch * self.nheads * self.headdim * self.dstate
        )
        return elements * 4 / 2**20


def _case(
    batch: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    dstate: int,
    chunk_size: int,
    ngroups: int,
    axis: str,
    tier: str = "standard",
) -> ShapeCase:
    return ShapeCase(
        batch,
        seqlen,
        nheads,
        headdim,
        dstate,
        chunk_size,
        ngroups,
        axis,
        tier,
    )


# The matrix deliberately covers generic, aligned, and Cube/MIX dispatches.
# Scaling families change one public dimension at a time wherever possible.
SHAPE_CASES: dict[str, ShapeCase] = {
    # Stable names retained for historical reports and CLI compatibility.
    "tiny": _case(1, 128, 2, 64, 64, 64, 1, "legacy", "smoke"),
    "small": _case(2, 512, 8, 64, 64, 64, 1, "legacy"),
    "medium": _case(4, 2048, 16, 64, 128, 128, 4, "legacy"),
    "extreme": _case(8, 4096, 32, 64, 128, 128, 8, "legacy", "stress"),
    "super_extreme": _case(8, 8192, 32, 64, 128, 128, 8, "legacy", "stress"),
    "ultra_extreme": _case(8, 16384, 32, 64, 128, 128, 8, "legacy", "stress"),

    # Correctness and dispatch coverage.
    "generic_tiny": _case(1, 16, 1, 8, 8, 8, 1, "dispatch", "smoke"),
    "generic_tail_l65": _case(1, 65, 4, 16, 16, 32, 2, "tail", "smoke"),
    "generic_tail_l513": _case(1, 513, 8, 32, 32, 64, 2, "tail"),
    "aligned_p16_n32": _case(1, 256, 8, 16, 32, 64, 2, "dispatch", "smoke"),
    "aligned_p32_n64": _case(1, 256, 8, 32, 64, 64, 2, "dispatch"),
    "cube_n64_c64": _case(1, 512, 8, 64, 64, 64, 2, "dispatch", "smoke"),
    "cube_n128_c128": _case(1, 512, 8, 64, 128, 128, 2, "dispatch", "smoke"),
    "logical_chunk256": _case(1, 1024, 16, 64, 128, 256, 4, "chunk", "smoke"),
    "logical_chunk512": _case(1, 2048, 16, 64, 128, 512, 4, "chunk"),

    # Sequence scaling: all other dimensions are fixed.
    **{
        f"seq_{length}": _case(8, length, 32, 64, 128, 128, 8, "sequence", "stress")
        for length in (512, 1024, 2048, 4096, 8192, 16384)
    },

    # Batch scaling at an otherwise fixed medium workload.
    **{
        f"batch_{batch}": _case(batch, 2048, 16, 64, 128, 128, 4, "batch")
        for batch in (1, 2, 4, 8, 16)
    },

    # Head scaling. G=4 is fixed, so this also exposes heads-per-group effects.
    **{
        f"head_{heads}": _case(4, 2048, heads, 64, 128, 128, 4, "heads")
        for heads in (4, 8, 16, 32, 64)
    },

    # Group scaling with fixed H=16.
    **{
        f"group_{groups}": _case(4, 2048, 16, 64, 128, 128, groups, "groups")
        for groups in (1, 2, 4, 8, 16)
    },

    # Inner dimensions isolate Vector/aligned versus Cube/MIX mappings.
    **{
        f"headdim_{headdim}": _case(2, 1024, 16, headdim, 64, 64, 4, "headdim")
        for headdim in (16, 32, 64, 128)
    },
    **{
        f"dstate_{dstate}": _case(2, 1024, 16, 64, dstate, 64, 4, "dstate")
        for dstate in (16, 32, 64, 128)
    },
    **{
        f"chunk_{chunk}": _case(2, 2048, 16, 64, 128, chunk, 4, "chunk")
        for chunk in (16, 32, 64, 128, 256, 512)
    },

    # Sustained-throughput gates retained for comparison with existing data.
    "heavy_h128": _case(8, 4096, 128, 64, 64, 64, 32, "stress", "stress"),
    "heavy_h256": _case(8, 4096, 256, 64, 64, 64, 64, "stress", "stress"),
}


SUITES: dict[str, tuple[str, ...]] = {
    "smoke": tuple(
        name for name, case in SHAPE_CASES.items() if case.tier == "smoke"
    ),
    "dispatch": tuple(
        name
        for name, case in SHAPE_CASES.items()
        if case.axis in {"dispatch", "tail"}
    ),
    "sequence": tuple(name for name, case in SHAPE_CASES.items() if case.axis == "sequence"),
    "batch": tuple(name for name, case in SHAPE_CASES.items() if case.axis == "batch"),
    "heads": tuple(name for name, case in SHAPE_CASES.items() if case.axis == "heads"),
    "groups": tuple(name for name, case in SHAPE_CASES.items() if case.axis == "groups"),
    "inner": tuple(
        name
        for name, case in SHAPE_CASES.items()
        if case.axis in {"headdim", "dstate", "chunk"}
    ),
    "stress": tuple(
        name for name, case in SHAPE_CASES.items() if case.tier == "stress"
    ),
}
SUITES["standard"] = tuple(
    dict.fromkeys(
        SUITES["smoke"]
        + ("generic_tail_l513", "aligned_p32_n64", "logical_chunk512")
        + ("seq_1024", "seq_4096")
        + ("batch_1", "batch_4", "batch_16")
        + ("head_4", "head_16", "head_64")
        + ("group_1", "group_4", "group_16")
        + ("headdim_16", "headdim_64", "headdim_128")
        + ("dstate_16", "dstate_64", "dstate_128")
        + ("chunk_16", "chunk_64", "chunk_128", "chunk_512")
    )
)
SUITES["all"] = tuple(SHAPE_CASES)


# Backward-compatible tuple dictionaries used by existing benchmark imports.
CASES = {name: case.internal_tuple for name, case in SHAPE_CASES.items()}
BASE_CASES = {name: CASES[name] for name in SUITES["standard"]}
SEQUENCE_CASES = {name: CASES[name] for name in SUITES["sequence"]}


def select_case_names(
    cases: list[str] | tuple[str, ...] | None,
    suite: str | None,
) -> tuple[str, ...]:
    """Resolve explicit cases or one named suite without hidden additions."""
    if cases:
        return tuple(cases)
    return SUITES[suite or "standard"]


def _row(name: str) -> dict[str, object]:
    case = SHAPE_CASES[name]
    return {
        "case": name,
        "shape_b_l_h_p_n_c_g": list(case.public_tuple),
        "axis": case.axis,
        "tier": case.tier,
        "expected_910b3_path": case.expected_910b3_path,
        "logical_tasks": case.logical_tasks,
        "full_feature_input_mib": round(case.full_feature_input_mib, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITES), default="standard")
    parser.add_argument("--cases", nargs="+", choices=tuple(SHAPE_CASES))
    parser.add_argument("--format", choices=("jsonl", "markdown"), default="markdown")
    args = parser.parse_args()
    rows = [_row(name) for name in select_case_names(args.cases, args.suite)]
    if args.format == "jsonl":
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
        return
    print("| Case | Shape `[B,L,H,P,N,chunk,G]` | Axis | Expected 910B3 path | Inputs |")
    print("|---|---|---|---|---:|")
    for row in rows:
        shape = "[" + ",".join(str(value) for value in row["shape_b_l_h_p_n_c_g"]) + "]"
        print(
            f"| `{row['case']}` | `{shape}` | {row['axis']} | "
            f"{row['expected_910b3_path']} | {row['full_feature_input_mib']:.3f} MiB |"
        )


if __name__ == "__main__":
    main()
