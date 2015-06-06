#include "imageproc/Swizzle.h"

void libcgt::core::imageproc::swizzle::RGBAToBGRA( Array2DView< const uint8x4 > input, Array2DView< uint8x4 > output )
{
	if( input.size() != output.size() )
	{
		return;
	}

	int w = input.width();
	int h = input.height();

	for( int y = 0; y < h; ++y )
	{
		for( int x = 0; x < w; ++x )
		{
            uint8x4 rgba = input[ { x, y } ];
			uint8x4 bgra = { rgba.z, rgba.y, rgba.x, rgba.w };			
            output[ { x, y } ] = bgra;
		}
	}
}

void libcgt::core::imageproc::swizzle::RGBAToARGB( Array2DView< const uint8x4 > input, Array2DView< uint8x4 > output )
{
	if( input.size() != output.size() )
	{
		return;
	}

	int w = input.width();
	int h = input.height();

	for( int y = 0; y < h; ++y )
	{
		for( int x = 0; x < w; ++x )
		{
            uint8x4 rgba = input[ { x, y } ];
			uint8x4 argb = { rgba.w, rgba.x, rgba.y, rgba.z };
            output[ { x, y } ] = argb;
		}
	}
}

void libcgt::core::imageproc::swizzle::BGRAToRGBA( Array2DView< const uint8x4 > input, Array2DView< uint8x4 > output )
{
	if( input.size() != output.size() )
	{
		return;
	}

	int w = input.width();
	int h = input.height();

	for( int y = 0; y < h; ++y )
	{
		for( int x = 0; x < w; ++x )
		{
            uint8x4 bgra = input[ { x, y } ];
			uint8x4 rgba = { bgra.z, bgra.y, bgra.x, bgra.w };
            output[ { x, y } ] = rgba;
		}
	}
}

void libcgt::core::imageproc::swizzle::RGBAToRGB( Array2DView< const uint8x4 > input, Array2DView< uint8x3 > output )
{
	if( input.size() != output.size() )
	{
		return;
	}

	int w = input.width();
	int h = input.height();

	for( int y = 0; y < h; ++y )
	{
		for( int x = 0; x < w; ++x )
		{
            uint8x4 rgba = input[ { x, y } ];
			uint8x3 rgb = { rgba.x, rgba.y, rgba.z };
            output[ { x, y } ] = rgb;
		}
	}
}

void libcgt::core::imageproc::swizzle::RGBAToBGR( Array2DView< const uint8x4 > input, Array2DView< uint8x3 > output )
{
	if( input.size() != output.size() )
	{
		return;
	}

	int w = input.width();
	int h = input.height();

	for( int y = 0; y < h; ++y )
	{
		for( int x = 0; x < w; ++x )
		{
            uint8x4 rgba = input[ { x, y } ];
			uint8x3 bgr = { rgba.z, rgba.y, rgba.x };
            output[ { x, y } ] = bgr;
		}
	}
}

void libcgt::core::imageproc::swizzle::BGRToRGBA( Array2DView< const uint8x3 > input, Array2DView< uint8x4 > output, uint8_t alpha )
{
	if( input.size() != output.size() )
	{
		return;
	}

	int w = input.width();
	int h = input.height();

	for( int y = 0; y < h; ++y )
	{
		for( int x = 0; x < w; ++x )
		{
            uint8x3 bgr = input[ { x, y } ];
			uint8x4 rgba = { bgr.z, bgr.y, bgr.x, alpha };
            output[ { x, y } ] = rgba;
		}
	}
}

void libcgt::core::imageproc::swizzle::BGRToBGRA( Array2DView< const uint8x3 > input, Array2DView< uint8x4 > output, uint8_t alpha )
{
	if( input.size() != output.size() )
	{
		return;
	}

	int w = input.width();
	int h = input.height();

	for( int y = 0; y < h; ++y )
	{
		for( int x = 0; x < w; ++x )
		{
            uint8x3 bgr = input[ { x, y } ];
			uint8x4 bgra = { bgr.x, bgr.y, bgr.z, alpha };
            output[ { x, y } ] = bgra;
		}
	}
}
