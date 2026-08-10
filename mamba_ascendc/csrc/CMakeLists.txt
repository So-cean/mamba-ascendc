# set the library output dir to the python dir for wheel package build
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/ascend_kernel/ascend_kernel/lib)

# host side files
FILE(GLOB OP_SRCS
    ${PROJECT_OP_SRC_BASE}/register.cpp
    ${PROJECT_OP_SRC_BASE}/aclnn/mamba2_arch35_matmul_smoke.cpp
    ${PROJECT_OP_SRC_BASE}/aclnn/mamba2_ssd_chunk_mix.cpp
    ${PROJECT_OP_SRC_BASE}/aclnn/mamba2_ssd_chunk_mix_grouped.cpp
    ${PROJECT_OP_SRC_BASE}/aclnn/mamba2_ssd_chunk_scan_bwd_off.cpp
    ${PROJECT_OP_SRC_BASE}/aclnn/mamba2_ssd_bwd_off_group_reduce.cpp
    ${PROJECT_OP_SRC_BASE}/aclnn/mamba2_ssd_chunk_scan_bwd_diag_state.cpp
    ${PROJECT_OP_SRC_BASE}/aclnn/mamba2_ssd_off_epilogue.cpp
    ${PROJECT_OP_SRC_BASE}/aclnn/mamba2_ssd_state_projection.cpp
    ${PROJECT_OP_SRC_BASE}/aclnn/mamba2_ssd_state_projection_grouped.cpp
    ${PROJECT_OP_SRC_BASE}/aclnn/mamba2_ssd_state_vector_epilogue.cpp
    ${PROJECT_OP_SRC_BASE}/aclnn/mamba2_ssd_state_vector_epilogue_grouped.cpp
    ${PROJECT_OP_SRC_BASE}/aclnn/mamba2_ssd_state_vector_epilogue_grouped_train.cpp
    ${PROJECT_OP_SRC_BASE}/aclnn/mamba2_ssd_state_epilogue.cpp
    ${PROJECT_OP_SRC_BASE}/aclnn/mamba2_ssd_state_epilogue_train.cpp
    ${PROJECT_OP_SRC_BASE}/aclnn/mamba2_ssd_state_epilogue_train_states.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_preprocess/op_host/mamba2_ssd_preprocess.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_state_passing/op_host/mamba2_ssd_state_passing.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_state_passing_grouped/op_host/mamba2_ssd_state_passing_grouped.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_state_passing_bwd/op_host/mamba2_ssd_state_passing_bwd.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_state_passing_bwd_half/op_host/mamba2_ssd_state_passing_bwd_half.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_dt_bwd/op_host/mamba2_ssd_dt_bwd.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_bwd_gate/op_host/mamba2_ssd_bwd_gate.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_bwd_diag_finalize/op_host/mamba2_ssd_bwd_diag_finalize.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_bwd_off_prepare/op_host/mamba2_ssd_bwd_off_prepare.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_bwd_off_finalize/op_host/mamba2_ssd_bwd_off_finalize.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_bwd_off_finalize_grouped/op_host/mamba2_ssd_bwd_off_finalize_grouped.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_prepare/op_host/mamba2_ssd_prepare.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_prepare_split/op_host/mamba2_ssd_prepare_split.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_prepare_dcb/op_host/mamba2_ssd_prepare_dcb.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_fwd/op_host/mamba2_ssd_fwd.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_bwd/op_host/mamba2_ssd_bwd.cpp
    )

# set the so name
set(OP_PLUGIN_NAME ascend_kernel)

# kernel side files without workspace
ascendc_library(no_workspace_kernel STATIC
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_fwd/op_kernel/mamba2_ssd_fwd.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_bwd/op_kernel/mamba2_ssd_bwd.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_preprocess/op_kernel/mamba2_ssd_preprocess.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_state_passing/op_kernel/mamba2_ssd_state_passing.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_state_passing_grouped/op_kernel/mamba2_ssd_state_passing_grouped.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_state_passing_bwd/op_kernel/mamba2_ssd_state_passing_bwd.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_state_passing_bwd_half/op_kernel/mamba2_ssd_state_passing_bwd_half.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_dt_bwd/op_kernel/mamba2_ssd_dt_bwd.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_bwd_gate/op_kernel/mamba2_ssd_bwd_gate.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_bwd_diag_finalize/op_kernel/mamba2_ssd_bwd_diag_finalize.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_bwd_off_prepare/op_kernel/mamba2_ssd_bwd_off_prepare.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_bwd_off_finalize/op_kernel/mamba2_ssd_bwd_off_finalize.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_bwd_off_finalize_grouped/op_kernel/mamba2_ssd_bwd_off_finalize_grouped.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_prepare/op_kernel/mamba2_ssd_prepare.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_prepare_split/op_kernel/mamba2_ssd_prepare_split.cpp
    ${PROJECT_OP_SRC_BASE}/ops/mamba2_ssd_prepare_dcb/op_kernel/mamba2_ssd_prepare_dcb.cpp
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
