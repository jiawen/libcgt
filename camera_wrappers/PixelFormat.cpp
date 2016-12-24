#include "PixelFormat.h"

namespace libcgt { namespace camera_wrappers {

uint32_t pixelSizeBytes( PixelFormat format )
{
    switch( format )
    {
    case PixelFormat::DEPTH_MM_U16:
        return 2;
    case PixelFormat::DEPTH_M_F32:
        return 4;
    case PixelFormat::DEPTH_M_F16:
        return 2;

    case PixelFormat::RGBA_U8888:
        return 4;
    case PixelFormat::RGB_U888:
        return 3;
    case PixelFormat::BGRA_U8888:
        return 4;
    case PixelFormat::BGR_U888:
        return 3;

    case PixelFormat::GRAY_U8:
        return 1;
    case PixelFormat::GRAY_U16:
        return 2;
    case PixelFormat::GRAY_U32:
        return 4;

    case PixelFormat::DISPARITY_S8:
        return 1;
    case PixelFormat::DISPARITY_S16:
        return 2;
    case PixelFormat::DISPARITY_S32:
        return 4;
    case PixelFormat::DISPARITY_F16:
        return 2;
    case PixelFormat::DISPARITY_F32:
        return 4;

    case PixelFormat::DEPTH_UNCALIBRATED_S8:
        return 1;
    case PixelFormat::DEPTH_UNCALIBRATED_U8:
        return 1;
    case PixelFormat::DEPTH_UNCALIBRATED_S16:
        return 2;
    case PixelFormat::DEPTH_UNCALIBRATED_U16:
        return 2;
    case PixelFormat::DEPTH_UNCALIBRATED_S32:
        return 4;
    case PixelFormat::DEPTH_UNCALIBRATED_U32:
        return 4;
    case PixelFormat::DEPTH_UNCALIBRATED_F16:
        return 2;
    case PixelFormat::DEPTH_UNCALIBRATED_F32:
        return 4;

    default:
        return 0;
    }
}

} } // camera_wrappers, libcgt
