#pragma once

#include <common/Array2DView.h>
#include <common/BasicTypes.h>

class Swizzle
{
public:

	// RGBA source
	// 4 -> 4
	static void RGBAToBGRA( Array2DView< ubyte4 > input, Array2DView< ubyte4 > output );
	static void RGBAToARGB( Array2DView< ubyte4 > input, Array2DView< ubyte4 > output );

	// 4 -> 3
	static void RGBAToRGB( Array2DView< ubyte4 > input, Array2DView< ubyte3 > output );
	static void RGBAToBGR( Array2DView< ubyte4 > input, Array2DView< ubyte3 > output );

	// BGR source
	// 3 to 4
	static void BGRToRGBA( Array2DView< ubyte3 > input, Array2DView< ubyte4 > output, ubyte alpha = 255 );
	static void BGRToBGRA( Array2DView< ubyte3 > input, Array2DView< ubyte4 > output, ubyte alpha = 255 );
};
