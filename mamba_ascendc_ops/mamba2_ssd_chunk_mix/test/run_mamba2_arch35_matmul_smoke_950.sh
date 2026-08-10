#!/usr/bin/env bash

# Internal-only runner. The five-second watchdog starts at the first custom-op
# launch marker, so Python import and first NPU context initialization are not
# incorrectly counted as kernel execution time.

set -uo pipefail

test_log="${1:-fwd_arch35_smoke_test.log}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
test_script="${2:-${script_dir}/test_mamba2_arch35_matmul_smoke_950.py}"

python -u "${test_script}" >"${test_log}" 2>&1 &
test_pid=$!

startup_deadline=$((SECONDS + 30))
while kill -0 "${test_pid}" 2>/dev/null; do
    if grep -q '"stage": "launch"' "${test_log}"; then
        break
    fi
    if (( SECONDS >= startup_deadline )); then
        kill -TERM "${test_pid}" 2>/dev/null || true
        wait "${test_pid}" 2>/dev/null || true
        exit 124
    fi
    sleep 0.1
done

if ! kill -0 "${test_pid}" 2>/dev/null; then
    wait "${test_pid}"
    exit $?
fi

(
    sleep 5
    kill -TERM "${test_pid}" 2>/dev/null || true
) &
watchdog_pid=$!

wait "${test_pid}"
status=$?
kill -TERM "${watchdog_pid}" 2>/dev/null || true
wait "${watchdog_pid}" 2>/dev/null || true
exit "${status}"
