"""Representative device-memory gate for Mamba2SsdDtBwd."""

import torch

import ascend_kernel  # noqa: F401

from dt_bwd_precision_common import all_cases, evaluate


def main():
    # T=128, multiple batch/head tasks, bias+softplus+finite clamp.
    case = all_cases()[19]
    metrics, passed = evaluate(case)
    if not passed:
        raise AssertionError(metrics)
    print("mamba2_ssd_dt_bwd mssanitizer case passed")


if __name__ == "__main__":
    main()
