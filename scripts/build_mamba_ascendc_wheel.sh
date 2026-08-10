#!/usr/bin/env bash

# Build one pip-installable wheel containing the PyTorch extension and custom OPP.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ops_root="${repo_root}/mamba_ascendc_ops/mamba2_ssd_chunk_mix"
extension_root="${repo_root}/mamba_ascendc"
package_root="${extension_root}/python/ascend_kernel/ascend_kernel"
opp_staging="${package_root}/opp"
wheel_output="${MAMBA_WHEEL_OUTPUT_DIR:-${repo_root}/dist}"
compute_unit="${MAMBA_ASCEND_COMPUTE_UNIT:-ascend910b}"
soc_version="${MAMBA_ASCEND_SOC_VERSION:-Ascend910_9382}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "Source the selected CANN set_env.sh before building the wheel." >&2
    exit 1
fi
if [[ ! -f "${ASCEND_HOME_PATH}/version.cfg" &&
      ! -f "${ASCEND_HOME_PATH}/share/info/runtime/version.info" &&
      ! -f "${ASCEND_HOME_PATH}/opp/version.info" ]]; then
    echo "Invalid CANN Toolkit path: ${ASCEND_HOME_PATH}." >&2
    exit 1
fi
python -c 'import torch, torch_npu, wheel' >/dev/null

cleanup_staging() {
    if [[ -d "${opp_staging}" ]]; then
        rm -rf -- "${opp_staging}"
    fi
}
trap cleanup_staging EXIT
cleanup_staging
mkdir -p "${opp_staging}" "${wheel_output}"

pushd "${ops_root}" >/dev/null
ASCEND_COMPUTE_UNIT="${compute_unit}" ./build.sh
opp_installer="$(find build_out -maxdepth 2 -type f -name 'custom_opp_*.run' -print -quit)"
if [[ -z "${opp_installer}" ]]; then
    echo "Custom OPP installer was not produced." >&2
    exit 1
fi
"${opp_installer}" --quiet --install-path="${opp_staging}"
popd >/dev/null

test -f "${opp_staging}/vendors/customize/op_api/lib/libcust_opapi.so"
test -d "${opp_staging}/vendors/customize/op_impl/ai_core/tbe/kernel/${compute_unit}"

pushd "${extension_root}" >/dev/null
MAMBA_ASCENDC_DEV_BUILD=0 ./build.sh "${soc_version}"
wheel_path="$(find output -maxdepth 1 -type f -name 'mamba_ascendc-*.whl' -print -quit)"
if [[ -z "${wheel_path}" ]]; then
    echo "mamba-ascendc wheel was not produced." >&2
    exit 1
fi
cp -f -- "${wheel_path}" "${wheel_output}/"
popd >/dev/null

final_wheel="${wheel_output}/$(basename "${wheel_path}")"
python - "${final_wheel}" <<'PY'
import sys
import zipfile

wheel = sys.argv[1]
required = (
    "ascend_kernel/lib/libascend_kernel.so",
    "ascend_kernel/opp/vendors/customize/op_api/lib/libcust_opapi.so",
    "ascend_kernel/opp/vendors/customize/op_impl/ai_core/tbe/kernel/",
)
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
for path in required:
    if not any(name.startswith(path) for name in names):
        raise SystemExit(f"wheel validation failed: missing {path}")
print(wheel)
PY

(
    cd "$(dirname "${final_wheel}")"
    sha256sum "$(basename "${final_wheel}")" \
        > "$(basename "${final_wheel}").sha256"
)
echo "Built ${final_wheel}"
