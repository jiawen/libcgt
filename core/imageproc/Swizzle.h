#pragma once

#include "libcgt/core/common/ArrayView.h"
#include "libcgt/core/common/BasicTypes.h"

namespace libcgt { namespace core { namespace imageproc {

// RGBA source, 4 -> 4.
void RGBAToBGRA( Array2DReadView< uint8x4 > src,
	Array2DWriteView< uint8x4 > dst );
void RGBAToARGB( Array2DReadView< uint8x4 > src,
	Array2DWriteView< uint8x4 > dst );

// RGBA source, 4 -> 3.
void RGBAToRGB( Array2DReadView< uint8x4 > src,
	Array2DWriteView< uint8x3 > dst );
void RGBAToBGR( Array2DReadView< uint8x4 > src,
	Array2DWriteView< uint8x3 > dst );

// BGRA source, 4 -> 4.
void BGRAToRGBA( Array2DReadView< uint8x4 > src,
	Array2DWriteView< uint8x4 > dst );

// BGRA source, 4 -> 3.
void BGRAToBGR( Array2DReadView< uint8x4 > src,
	Array2DWriteView< uint8x3 > dst );
void BGRAToRGB( Array2DReadView< uint8x4 > src,
	Array2DWriteView< uint8x3 > dst );

// BGR source, 3 to 4.
void BGRToRGBA( Array2DReadView< uint8x3 > src,
	Array2DWriteView< uint8x4 > dst,
    uint8_t alpha = 255 );
void BGRToBGRA( Array2DReadView< uint8x3 > src,
	Array2DWriteView< uint8x4 > dst,
    uint8_t alpha = 255 );

// RGB source, 3 to 4.
void RGBToBGRA( Array2DReadView< uint8x3 > src,
	Array2DWriteView< uint8x4 > dst,
    uint8_t alpha = 255 );
void RGBToRGBA( Array2DReadView< uint8x3 > src,
	Array2DWriteView< uint8x4 > dst,
    uint8_t alpha = 255 );

// RGB source, 3 to 3.
void RGBToBGR( Array2DReadView< uint8x3 > src,
	Array2DWriteView< uint8x3 > dst );

} } } // imageproc, core, libcgt
