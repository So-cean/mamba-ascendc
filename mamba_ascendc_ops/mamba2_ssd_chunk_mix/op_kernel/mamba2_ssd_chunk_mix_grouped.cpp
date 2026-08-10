// Ascend 950PR group-owned ChunkMix variant.  The implementation is shared
// with the accepted key-9 kernel; this compile-time contract only changes the
// wide state GEMM destination and removes the legacy scatter synchronization.

#define MAMBA2_CHUNK_MIX_GROUPED_OUTPUT
#include "mamba2_ssd_chunk_mix.cpp"
