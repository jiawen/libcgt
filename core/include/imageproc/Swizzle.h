#pragma once

#include <common/Array2DView.h>
#include <common/BasicTypes.h>

namespace libcgt
{
namespace core
{
namespace imageproc
{
namespace swizzle
{
    // RGBA source
    // 4 -> 4
    void RGBAToBGRA( Array2DView< const uint8x4 > input, Array2DView< uint8x4 > output );
    void RGBAToARGB( Array2DView< const uint8x4 > input, Array2DView< uint8x4 > output );

    // BGRA source
    // 4 -> 4
    void BGRAToRGBA( Array2DView< const uint8x4 > input, Array2DView< uint8x4 > output );

    // 4 -> 3
    void RGBAToRGB( Array2DView< const uint8x4 > input, Array2DView< uint8x3 > output );
    void RGBAToBGR( Array2DView< const uint8x4 > input, Array2DView< uint8x3 > output );

    // BGR source
    // 3 to 4
    void BGRToRGBA( Array2DView< const uint8x3 > input, Array2DView< uint8x4 > output, uint8_t alpha = 255 );
    void BGRToBGRA( Array2DView< const uint8x3 > input, Array2DView< uint8x4 > output, uint8_t alpha = 255 );
}
}
}
}
