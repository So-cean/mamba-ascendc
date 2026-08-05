# set the library output dir to the python dir for wheel package build
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/ascend_kernel/ascend_kernel/lib)

# host side files
FILE(GLOB OP_SRCS
    ${PROJECT_OP_SRC_BASE}/register.cpp
    ${PROJECT_OP_SRC_BASE}/aclnn/mamba2_ssd_chunk_mix.cpp
    ${PROJECT_OP_SRC_BASE}/aclnn/mamba2_ssd_off_epilogue.cpp
    ${PROJECT_OP_SRC_BASE}/aclnn/mamba2_ssd_state_epilogue.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_preprocess/op_host/mamba2_ssd_preprocess.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_state_passing/op_host/mamba2_ssd_state_passing.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_prepare/op_host/mamba2_ssd_prepare.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_fwd/op_host/mamba2_ssd_fwd.cpp
    )

# set the so name
set(OP_PLUGIN_NAME ascend_kernel)

# kernel side files without workspace
ascendc_library(no_workspace_kernel STATIC
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_fwd/op_kernel/mamba2_ssd_fwd.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_preprocess/op_kernel/mamba2_ssd_preprocess.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_state_passing/op_kernel/mamba2_ssd_state_passing.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_prepare/op_kernel/mamba2_ssd_prepare.cpp
)

# create shared library libascend_kernel.so
add_library(${OP_PLUGIN_NAME} SHARED ${OP_SRCS})

target_link_libraries(${OP_PLUGIN_NAME} PRIVATE
        no_workspace_kernel
        torch_npu
        ascendcl
        tiling_api
        nnopbase
        opapi
        register
        platform
        ascendalog
        dl
)

target_link_directories(${OP_PLUGIN_NAME} PRIVATE
        ${TORCH_DIR}/lib
        ${TORCH_NPU_DIR}/lib
)

target_include_directories(${OP_PLUGIN_NAME} PRIVATE
        ${PROJECT_OP_SRC_BASE}/utils
        ${PROJECT_SOURCE_DIR}/include
        ${TORCH_DIR}/include
        ${TORCH_DIR}/include/torch/csrc/api/include
        ${TORCH_NPU_DIR}/include/third_party/acl/inc
        ${TORCH_NPU_DIR}/include/third_party/hccl/inc
        ${TORCH_NPU_DIR}/include
        ${PYTHON_INCLUDE_DIR}
        ${ASCEND_INCLUDE_DIR}/external
        ${ASCEND_INCLUDE_DIR}/experiment/platform
        ${ASCEND_INCLUDE_DIR}/experiment/runtime
)
