#include "imageproc/Swizzle.h"

#include <common/ArrayUtils.h>

using libcgt::core::arrayutils::map;

namespace libcgt { namespace core { namespace imageproc {

// TODO: return bool (just return map()).
void RGBAToBGRA( Array2DView< const uint8x4 > src, Array2DView< uint8x4 > dst )
{
    map( src, dst,
        [&] ( uint8x4 rgba )
        {
            return uint8x4{ rgba.z, rgba.y, rgba.x, rgba.w };
        }
    );
}

void RGBAToARGB( Array2DView< const uint8x4 > src, Array2DView< uint8x4 > dst )
{
    map( src, dst,
        [&] ( uint8x4 rgba )
        {
            return uint8x4{ rgba.w, rgba.x, rgba.y, rgba.z };
        }
    );
}

void RGBAToRGB( Array2DView< const uint8x4 > src, Array2DView< uint8x3 > dst )
{
    map( src, dst,
        [&] ( uint8x4 rgba )
        {
            return uint8x3{ rgba.x, rgba.y, rgba.z };
        }
    );
}

void RGBAToBGR( Array2DView< const uint8x4 > src, Array2DView< uint8x3 > dst )
{
    map( src, dst,
        [&] ( uint8x4 rgba )
        {
            return uint8x3{ rgba.z, rgba.y, rgba.x };
        }
    );
}

void BGRAToRGBA( Array2DView< const uint8x4 > src, Array2DView< uint8x4 > dst )
{
    map( src, dst,
        [&] ( uint8x4 bgra )
        {
            return uint8x4{ bgra.z, bgra.y, bgra.x, bgra.w };
        }
    );
}

void BGRAToBGR( Array2DView< const uint8x4 > src, Array2DView< uint8x3 > dst )
{
    map( src, dst,
        [&]( uint8x4 bgra )
        {
            return uint8x3{ bgra.x, bgra.y, bgra.z };
        }
    );
}

void BGRAToRGB( Array2DView< const uint8x4 > src, Array2DView< uint8x3 > dst )
{
    map( src, dst,
        [&] ( uint8x4 bgra )
        {
            return uint8x3{ bgra.z, bgra.y, bgra.x };
        }
    );
}

void BGRToRGBA( Array2DView< const uint8x3 > src, Array2DView< uint8x4 > dst,
    uint8_t alpha )
{
    map( src, dst,
        [&] ( uint8x3 bgr )
        {
            return uint8x4{ bgr.z, bgr.y, bgr.x, alpha };
        }
    );
}

void BGRToBGRA( Array2DView< const uint8x3 > src, Array2DView< uint8x4 > dst,
    uint8_t alpha )
{
    map( src, dst,
        [&] ( uint8x3 bgr )
        {
            return uint8x4{ bgr.x, bgr.y, bgr.z, alpha };
        }
    );
}

void RGBToBGRA( Array2DView< const uint8x3 > src, Array2DView< uint8x4 > dst,
    uint8_t alpha )
{
    map( src, dst,
        [&] ( uint8x3 rgb )
        {
            return uint8x4{ rgb.z, rgb.y, rgb.x, alpha };
        }
    );
}

void RGBToRGBA( Array2DView< const uint8x3 > src, Array2DView< uint8x4 > dst,
    uint8_t alpha )
{
    map( src, dst,
        [&] ( uint8x3 rgb )
        {
            return uint8x4{ rgb.x, rgb.y, rgb.z, alpha };
        }
    );
}

void RGBToBGR( Array2DView< const uint8x3 > src, Array2DView< uint8x3 > dst )
{
    map( src, dst,
        [&] ( uint8x3 rgb )
        {
            return uint8x3{ rgb.z, rgb.y, rgb.x };
        }
    );
}

} } } // imageproc, core, libcgt
