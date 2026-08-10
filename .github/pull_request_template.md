## Summary

- What changed:
- Why:

## Validation

- [ ] CPU reference tests pass.
- [ ] Python syntax check passes.
- [ ] NPU precision was checked when AscendC/Triton code changed.
- [ ] Relevant heavy-shape benchmark was rerun when performance-sensitive code changed.
- [ ] README figures/data were updated when published results changed.
- [ ] Logs and diffs contain no private paths, credentials, node names, or scheduler job IDs.

## Benchmark protocol

If performance changed, record device, shape, dtype, optional inputs, timing API,
warmup, repeat count, p50 latency, baseline, and profiler evidence.
