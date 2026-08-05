# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0)
endif()

add_ops_compile_options(Mamba2SsdChunkMix OPTIONS --tiling_key=2,3,7)
add_ops_compile_options(Mamba2SsdOffEpilogue OPTIONS --tiling_key=3)
add_ops_compile_options(Mamba2SsdStateEpilogue OPTIONS --tiling_key=4,8)

add_kernels_compile()
