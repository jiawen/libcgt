#pragma once

#include <common/Array2DView.h>
#include <common/BasicTypes.h>

namespace libcgt { namespace core { namespace imageproc {

// RGBA source, 4 -> 4.
void RGBAToBGRA( Array2DView< const uint8x4 > src, Array2DView< uint8x4 > dst );
void RGBAToARGB( Array2DView< const uint8x4 > src, Array2DView< uint8x4 > dst );

// RGBA source, 4 -> 3.
void RGBAToRGB( Array2DView< const uint8x4 > src, Array2DView< uint8x3 > dst );
void RGBAToBGR( Array2DView< const uint8x4 > src, Array2DView< uint8x3 > dst );

// BGRA source, 4 -> 4.
void BGRAToRGBA( Array2DView< const uint8x4 > src, Array2DView< uint8x4 > dst );

// BGRA source, 4 -> 3.
void BGRAToBGR( Array2DView< const uint8x4 > src, Array2DView< uint8x3 > dst );
void BGRAToRGB( Array2DView< const uint8x4 > src, Array2DView< uint8x3 > dst );

// BGR source, 3 to 4.
void BGRToRGBA( Array2DView< const uint8x3 > src, Array2DView< uint8x4 > dst,
    uint8_t alpha = 255 );
void BGRToBGRA( Array2DView< const uint8x3 > src, Array2DView< uint8x4 > dst,
    uint8_t alpha = 255 );

// RGB source, 3 to 4.
void RGBToBGRA( Array2DView< const uint8x3 > src, Array2DView< uint8x4 > dst,
    uint8_t alpha = 255 );
void RGBToRGBA( Array2DView< const uint8x3 > src, Array2DView< uint8x4 > dst,
    uint8_t alpha = 255 );

// RGB source, 3 to 3.
void RGBToBGR( Array2DView< const uint8x3 > src, Array2DView< uint8x3 > dst );

} } } // imageproc, core, libcgt
