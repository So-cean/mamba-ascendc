#!/bin/bash
set -e

BUILD_KERNELS_MODULE="ON"
DEBUG_MODE="OFF"
# Source iteration is the safe default.  Release packaging must opt in via
# scripts/build_mamba_ascendc_wheel.sh, which sets this switch to 0 explicitly.
DEV_BUILD="${MAMBA_ASCENDC_DEV_BUILD:-1}"

while getopts ":a:hd" opt; do
    case ${opt} in
        a )
            BUILD_KERNELS_MODULE="OFF"
            case "$OPTARG" in
                kernels )
                    BUILD_KERNELS_MODULE="ON"
                    ;;
                * )
                    echo "Error: Invalid Value"
                    echo "Allowed value: kernels"
                    exit 1
                    ;;
            esac
            ;;
        d )
            DEBUG_MODE="ON"
            ;;
        h )
            echo "Use './build.sh' build all modules."
            echo "Use './build.sh -a <target>' to build specific parts of the project."
            echo "    <target> can be:"
            echo "    kernels           Only build ascend_kernel."
            exit 1
            ;;
        \? )
            echo "Error: unknown flag: -$OPTARG" 1>&2
            echo "Run './build.sh -h' for more information."
            exit 1
            ;;
        : )
            echo "Error: -$OPTARG requires a value" 1>&2
            echo "Run './build.sh -h' for more information."
            exit 1
            ;;
    esac
done

shift $((OPTIND -1))


export DEBUG_MODE=$DEBUG_MODE

SOC_VERSION="${1:-Ascend910_9382}"


### Use the CANN Toolkit selected by the active conda environment.
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is unset. Source the selected CANN set_env.sh before building."
    exit 1
fi
_CANN_TOOLKIT_INSTALL_PATH="${ASCEND_HOME_PATH}"
# CANN 8.x installations expose version.cfg at the toolkit root.  The CANN
# 9.0 unified image replaces it with component version.info files while keeping
# set_env.sh and the public include/lib directories at the same root.
if [[ ! -f "${_CANN_TOOLKIT_INSTALL_PATH}/version.cfg" &&
      ! -f "${_CANN_TOOLKIT_INSTALL_PATH}/share/info/runtime/version.info" &&
      ! -f "${_CANN_TOOLKIT_INSTALL_PATH}/opp/version.info" ]]; then
    echo "Invalid CANN Toolkit path: ${_CANN_TOOLKIT_INSTALL_PATH}."
    exit 1
fi
echo -e "\e[1;32mUsing active CANN Toolkit: ${_CANN_TOOLKIT_INSTALL_PATH}\e[0m"
echo -e "\e[1;33mDouble Checking Environment Variables:\e[0m"
echo -e "\e[1;32mASCEND_HOME_PATH: ${ASCEND_HOME_PATH}\e[0m"
echo -e "\e[1;32mASCEND_TOOLKIT_HOME: ${ASCEND_TOOLKIT_HOME}\e[0m"


ASCEND_INCLUDE_DIR=${ASCEND_TOOLKIT_HOME}/$(arch)-linux/include
CURRENT_DIR=$(pwd)
PROJECT_ROOT=$(dirname "$CURRENT_DIR")
OUTPUT_DIR=$CURRENT_DIR/output
mkdir -p $OUTPUT_DIR
echo "outpath: ${OUTPUT_DIR}"

COMPILE_OPTIONS=""

function build_kernels()
{
    CMAKE_DIR=""
    BUILD_DIR="build"

    cd "$CMAKE_DIR" || exit

    if [[ "$DEV_BUILD" != "1" || ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
        rm -rf "$BUILD_DIR"
        mkdir -p "$BUILD_DIR"
        cmake $COMPILE_OPTIONS \
        -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
        -DASCEND_HOME_PATH=$ASCEND_HOME_PATH \
        -DASCEND_INCLUDE_DIR=$ASCEND_INCLUDE_DIR \
        -DSOC_VERSION=$SOC_VERSION \
        -DBUILD_KERNELS_MODULE=$BUILD_KERNELS_MODULE \
        -B "$BUILD_DIR" \
        -S .
    fi

    if [[ "$DEV_BUILD" == "1" ]]; then
        # Development iterations load this shared library directly.  Keep the
        # CMake tree and rebuild only changed dependencies; wheel/OPP packaging
        # belongs to the final release gate.
        cmake --build "$BUILD_DIR" --target ascend_kernel -j 16
    else
        cmake --build "$BUILD_DIR" --target install -j 16
    fi
    cd -
}

function make_ascend_kernel_package()
{
    cd python/ascend_kernel || exit

    rm -rf "$CURRENT_DIR"/python/ascend_kernel/dist
    cp -v "${CURRENT_DIR}/config.ini" "${CURRENT_DIR}/python/ascend_kernel/ascend_kernel/"
    python3 -m pip wheel . --wheel-dir dist --no-deps --no-build-isolation
    mv -v "$CURRENT_DIR"/python/ascend_kernel/dist/mamba_ascendc*.whl ${OUTPUT_DIR}/
    rm -rf "$CURRENT_DIR"/python/ascend_kernel/dist
    cd -
}

function main()
{

    build_kernels
    if [[ "$DEV_BUILD" == "1" ]]; then
        return
    fi
    if pip3 show wheel;then
        echo "wheel has been installed"
    else
        pip3 install wheel==0.45.1
    fi
    if [[ "$BUILD_KERNELS_MODULE" == "ON" ]]; then
        make_ascend_kernel_package
    fi

}

main
