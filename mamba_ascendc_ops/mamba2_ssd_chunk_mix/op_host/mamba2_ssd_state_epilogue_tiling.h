#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Mamba2SsdStateEpilogueTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, heads);
TILING_DATA_FIELD_DEF(uint32_t, chunks);
TILING_DATA_FIELD_DEF(uint32_t, groups);
TILING_DATA_FIELD_DEF(uint32_t, stateDim);
TILING_DATA_FIELD_DEF(uint32_t, taskCount);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, workspaceBytesPerCore);
TILING_DATA_FIELD_DEF(uint32_t, headsPerGroup);
TILING_DATA_FIELD_DEF(uint32_t, headsPerTask);
TILING_DATA_FIELD_DEF(uint32_t, hasInitial);
TILING_DATA_FIELD_DEF(uint32_t, aivPerAic);
TILING_DATA_FIELD_DEF(uint32_t, inputGrouped);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Mamba2SsdStateEpilogue,
                           Mamba2SsdStateEpilogueTilingData)
REGISTER_TILING_DATA_CLASS(Mamba2SsdStateEpilogueTrain,
                           Mamba2SsdStateEpilogueTilingData)
REGISTER_TILING_DATA_CLASS(Mamba2SsdStateEpilogueTrainStates,
                           Mamba2SsdStateEpilogueTilingData)
} // namespace optiling
