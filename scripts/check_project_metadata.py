#!/usr/bin/env python3
"""Validate public project metadata against canonical repository data."""

from __future__ import annotations

import configparser
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_URL = "https://github.com/So-cean/mamba-ascendc"
CODEMETA_CONTEXT = "https://w3id.org/codemeta/3.1"
HEAVY_SHAPE = [8, 4096, 256, 64, 64, 64, 64]


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject non-object roots."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError(f"{path.relative_to(ROOT)} must contain a JSON object")
    return data


def project_version() -> str:
    """Read the canonical package version."""
    config = configparser.ConfigParser()
    config.read(ROOT / "mamba_ascendc" / "config.ini")
    return config.get("global", "version")


def require_text(path: Path, snippets: list[str]) -> str:
    """Require a non-empty UTF-8 file and all expected snippets."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise AssertionError(f"{path.relative_to(ROOT)} must not be empty")
    for snippet in snippets:
        if snippet not in text:
            raise AssertionError(
                f"{path.relative_to(ROOT)} is missing required text: {snippet}"
            )
    return text


def main() -> None:
    """Validate version, URLs, APIs, and benchmark facts across metadata files."""
    version = project_version()
    release_url = f"{REPOSITORY_URL}/releases/tag/v{version}"

    package_config = configparser.ConfigParser()
    package_config.read(
        ROOT
        / "mamba_ascendc"
        / "python"
        / "ascend_kernel"
        / "ascend_kernel"
        / "config.ini"
    )
    assert package_config.get("global", "version") == version

    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert isinstance(citation, dict)
    assert citation["cff-version"] == "1.2.0"
    assert citation["version"] == version
    assert citation["repository-code"] == REPOSITORY_URL
    assert citation["url"] == release_url
    assert citation["license"] == "Apache-2.0"

    codemeta = load_json(ROOT / "codemeta.json")
    assert codemeta["@context"] == CODEMETA_CONTEXT
    assert codemeta["type"] == "SoftwareSourceCode"
    assert codemeta["@id"] == REPOSITORY_URL
    assert codemeta["codeRepository"] == REPOSITORY_URL
    assert codemeta["version"] == version
    assert codemeta["releaseNotes"] == release_url
    assert "selective scan" in codemeta["keywords"]
    assert "torch_npu" in codemeta["keywords"]

    sys_path = ROOT / "benchmarks"
    import sys

    sys.path.insert(0, str(sys_path))
    from mamba2_shape_matrix import SHAPE_CASES, SUITES  # noqa: PLC0415

    assert len(SHAPE_CASES) >= 50
    assert len(SUITES["standard"]) >= 30
    assert {case.expected_910b3_path for case in SHAPE_CASES.values()} == {
        "generic",
        "aligned",
        "cube_mix",
    }

    benchmark = load_json(ROOT / "benchmarks" / "results" / "readme_benchmarks.json")
    gate = benchmark["h256_training_gate"]
    assert gate["shape"] == HEAVY_SHAPE
    a100_ms = gate["a100"]["forward_ms"]
    ascend_ms = gate["ascend_910b3"]["forward_ms"]
    throughput_percent = a100_ms / ascend_ms * 100
    assert round(throughput_percent, 2) == 70.82
    matrix = benchmark["a100_inference_shape_matrix"]
    assert matrix["cases_total"] == matrix["cases_ok"] == len(SHAPE_CASES) == 52
    assert len(matrix["sequence_scaling"]) == 6
    assert matrix["precision_smoke"]["cases"] == 6
    assert matrix["precision_smoke"]["passed"] == 6
    assert matrix["precision_smoke"]["finite"] is True

    facts = [
        REPOSITORY_URL,
        f"v{version}",
        "mamba2_ssd_fwd",
        "mamba_chunk_scan_combined",
        "31.888 ms",
        "22.583 ms",
        "70.82%",
        "52/52",
        "15.041 ms",
    ]
    require_text(
        ROOT / "README.md",
        [*facts, "selective scan", "codemeta.json", "CITATION.cff"],
    )
    require_text(
        ROOT / "llms.txt",
        [*facts, "selective scan", "readme_benchmarks.json"],
    )
    require_text(
        ROOT / "docs" / "index.md",
        [REPOSITORY_URL, "SoftwareSourceCode", "selective scan", "52-case"],
    )
    require_text(
        ROOT / "scripts" / "update_github_discovery.sh",
        ["selective-scan", "torch-npu", "source[path]=/docs"],
    )

    print(f"Validated project metadata for mamba-ascendc v{version}")


if __name__ == "__main__":
    main()
