#!/usr/bin/env bash

set -euo pipefail

base_ref="${1:-}"
head_ref="${2:-HEAD}"

if [[ -z "${base_ref}" ]] || ! git cat-file -e "${base_ref}^{commit}" 2>/dev/null; then
    base_ref="$(git rev-parse "${head_ref}^")"
fi

mapfile -d '' python_files < <(
    git diff --name-only -z --diff-filter=ACMR \
        "${base_ref}" "${head_ref}" -- '*.py'
)
if (( ${#python_files[@]} )); then
    ruff check --force-exclude "${python_files[@]}"
fi

mapfile -d '' native_candidates < <(
    git diff --name-only -z --diff-filter=ACMR \
        "${base_ref}" "${head_ref}" -- \
        '*.c' '*.cc' '*.cpp' '*.cxx' '*.h' '*.hh' '*.hpp'
)

native_files=()
for path in "${native_candidates[@]}"; do
    case "${path}" in
        mamba_ascendc_ops/*/cmake/*|mamba_triton_ascend/*)
            continue
            ;;
    esac
    native_files+=("${path}")
done

if (( ${#native_files[@]} )); then
    clang-format --dry-run --Werror "${native_files[@]}"
fi
