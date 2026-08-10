#!/usr/bin/env python3
"""Invoke pytest after run_source_candidate has loaded a source build."""

from __future__ import annotations

import sys

import pytest


if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:]))
