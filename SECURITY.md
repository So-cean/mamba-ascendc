# Security policy

## Supported versions

Security fixes are applied to the latest release and the current `main`
branch. The initial supported release line is `0.1.x`.

## Reporting a vulnerability

Use GitHub private vulnerability reporting instead of a public issue when the
problem involves arbitrary code execution, unsafe memory access, sensitive
data exposure, malicious model/input handling, or package supply-chain risk:

<https://github.com/So-cean/mamba-ascendc/security/advisories/new>

Include the affected version/commit, device and CANN version, minimal
reproduction, expected impact, and whether mssanitizer reproduces the problem.
Remove credentials, private paths, node names, scheduler IDs, and unrelated
profiler data from the report.

Precision differences and ordinary performance regressions should use the
public issue templates instead.
