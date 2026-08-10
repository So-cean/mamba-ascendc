#ifndef MAMBA2_ARCH35_MATMUL_SMOKE_TILING_H
#define MAMBA2_ARCH35_MATMUL_SMOKE_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Mamba2Arch35MatmulSmokeTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Mamba2Arch35MatmulSmoke,
                           Mamba2Arch35MatmulSmokeTilingData)
} // namespace optiling

#endif // MAMBA2_ARCH35_MATMUL_SMOKE_TILING_H
