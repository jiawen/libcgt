#pragma once

#include <cstdint>

namespace libcgt { namespace camera_wrappers {

enum class PixelFormat : uint32_t
{
    INVALID = 0,

    DEPTH_MM_U16 = 1,
    DEPTH_M_F32 = 2,

    RGBA_U8888 = 16,
    RGB_U888 = 17,
    BGRA_U8888 = 18,
    BGR_U888 = 19,

    GRAY_U8 = 128,
    GRAY_U16 = 129,
    GRAY_U32 = 130
};

uint32_t pixelSizeBytes( PixelFormat format );

} } // camera_wrappers, libcgt
