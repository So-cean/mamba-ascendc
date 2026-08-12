#!/usr/bin/env bash
set -euo pipefail

repository="So-cean/mamba-ascendc"
homepage="https://so-cean.github.io/mamba-ascendc/"
description="Native AscendC Mamba2 selective scan / SSD forward-backward custom operator for Huawei Ascend 910B3 and 950PR, with CANN, torch_npu, A100 benchmarks and msprof profiling."

gh repo edit "${repository}" \
    --description "${description}" \
    --homepage "${homepage}" \
    --add-topic ai-infrastructure \
    --add-topic ascend-910b \
    --add-topic ascend-950 \
    --add-topic ascend-npu \
    --add-topic ascendc \
    --add-topic cann \
    --add-topic custom-operator \
    --add-topic huawei-ascend \
    --add-topic kernel-optimization \
    --add-topic mamba \
    --add-topic mamba2 \
    --add-topic performance-optimization \
    --add-topic pytorch \
    --add-topic selective-scan \
    --add-topic ssd \
    --add-topic state-space-models \
    --add-topic torch-npu \
    --add-topic triton-ascend

# GitHub Pages can only be enabled after docs/index.md exists on the default
# branch.  A 409 means the page is already configured and is therefore fine.
response="$(mktemp)"
status="$({ gh api --method POST "repos/${repository}/pages" \
    -f source[branch]=main -f source[path]=/docs >"${response}"; } 2>&1 && echo ok || echo failed)"
if [[ "${status}" == "failed" ]] && ! grep -Eq 'already exists|HTTP 409' "${response}"; then
    cat "${response}" >&2
    exit 1
fi
rm -f "${response}"

gh repo view "${repository}" \
    --json nameWithOwner,description,homepageUrl,repositoryTopics,url,visibility
