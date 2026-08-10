// Copyright (c) 2026, mamba-ascendc authors.
#pragma once

#if __has_include("version/cann_version.h")
#include "version/cann_version.h"
#endif

#include "tiling/platform/platform_ascendc.h"

#if defined(CANN_MAJOR) && CANN_MAJOR >= 9
#define MAMBA2_ASCEND950_CONFIG "ascend950"
#else
// CANN 8.x uses the legacy product key for the same architecture family.
#define MAMBA2_ASCEND950_CONFIG "ascend910_95"
#endif

namespace mamba2_compat {

// CANN 8.x does not declare SocVersion::ASCEND950.  Keep the enum reference
// out of those translation units so one source tree can build for both
// Ascend 910B3 (CANN 8.x) and Ascend 950PR (CANN 9.x).
inline bool IsAscend950(const platform_ascendc::PlatformAscendC &platform)
{
#if defined(CANN_MAJOR) && CANN_MAJOR >= 9
    return platform.GetSocVersion() ==
           platform_ascendc::SocVersion::ASCEND950;
#else
    (void)platform;
    return false;
#endif
}

}  // namespace mamba2_compat
