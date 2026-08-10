#!/usr/bin/env python
# coding=utf-8

"""Build the binary mamba-ascendc wheel."""

from configparser import ConfigParser
from pathlib import Path

import setuptools
from setuptools import find_namespace_packages
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""

    def has_ext_modules(self):
        return True


WORKING_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = WORKING_DIR / "ascend_kernel"
# setup.py lives at <repository>/mamba_ascendc/python/ascend_kernel/setup.py.
REPOSITORY_ROOT = WORKING_DIR.parents[2]
config = ConfigParser()
config.read(PACKAGE_DIR / "config.ini")
_version = config.get("global", "version")


def package_files(directory: str):
    """Return non-Python runtime files relative to the package root."""
    root = PACKAGE_DIR / directory
    if not root.is_dir():
        return []
    return [
        str(path.relative_to(PACKAGE_DIR))
        for path in root.rglob("*")
        if path.is_file()
    ]


runtime_files = (
    package_files("lib")
    + package_files("opp")
    + package_files("licenses")
    + ["config.ini"]
)
if not (PACKAGE_DIR / "lib" / "libascend_kernel.so").is_file():
    raise RuntimeError("libascend_kernel.so is missing; build the C++ extension first")
if not (
    PACKAGE_DIR
    / "opp"
    / "vendors"
    / "customize"
    / "op_api"
    / "lib"
    / "libcust_opapi.so"
).is_file():
    raise RuntimeError(
        "bundled custom OPP is missing; use scripts/build_mamba_ascendc_wheel.sh"
    )


setuptools.setup(
    name="mamba-ascendc",
    version=_version,
    description="Mamba-2 SSD forward and backward operators for Ascend NPU",
    long_description=(REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/So-cean/mamba-ascendc",
    project_urls={
        "Source": "https://github.com/So-cean/mamba-ascendc",
        "Issues": "https://github.com/So-cean/mamba-ascendc/issues",
        "Benchmarks": "https://github.com/So-cean/mamba-ascendc#benchmark",
    },
    packages=find_namespace_packages(
        include=("ascend_kernel", "ascend_kernel.*"), exclude=("tests*",)
    ),
    distclass=BinaryDistribution,
    license="Apache-2.0",
    python_requires=">=3.10,<3.12",
    package_data={"ascend_kernel": runtime_files},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
        "Topic :: Software Development :: Libraries",
    ],
)
