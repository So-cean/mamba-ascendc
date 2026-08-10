#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Mamba2SsdStateVectorEpilogueGroupedTrainTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, heads);
    TILING_DATA_FIELD_DEF(uint32_t, chunks);
    TILING_DATA_FIELD_DEF(uint32_t, groups);
    TILING_DATA_FIELD_DEF(uint32_t, headsPerGroup);
    TILING_DATA_FIELD_DEF(uint32_t, taskCount);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, groupedDiag);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    Mamba2SsdStateVectorEpilogueGroupedTrain,
    Mamba2SsdStateVectorEpilogueGroupedTrainTilingData)
}  // namespace optiling
