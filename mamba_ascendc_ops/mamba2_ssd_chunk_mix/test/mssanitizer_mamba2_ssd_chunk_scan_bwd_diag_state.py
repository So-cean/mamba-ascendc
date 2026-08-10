"""Small multi-task driver for mssanitizer on the fused backward kernel."""

from chunk_scan_bwd_diag_state_precision_common import all_cases, evaluate


def main() -> None:
    # Case 09 has two heads and two chunks.  It exercises per-core workspace
    # reuse and both AIV subcores without making sanitizer runs unnecessarily
    # large.
    case = all_cases()[8]
    output_metrics, passed = evaluate(case)
    if not passed:
        raise RuntimeError(output_metrics)
    print("mamba2_ssd_chunk_scan_bwd_diag_state mssanitizer driver: PASS")


if __name__ == "__main__":
    main()
