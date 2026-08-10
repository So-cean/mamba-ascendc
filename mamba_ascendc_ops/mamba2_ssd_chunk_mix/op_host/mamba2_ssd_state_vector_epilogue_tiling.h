#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Mamba2SsdStateVectorEpilogueTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, heads);
    TILING_DATA_FIELD_DEF(uint32_t, chunks);
    TILING_DATA_FIELD_DEF(uint32_t, taskCount);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Mamba2SsdStateVectorEpilogue,
                           Mamba2SsdStateVectorEpilogueTilingData)
}  // namespace optiling
