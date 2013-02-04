#include "imageproc/Swizzle.h"

// static
void Swizzle::RGBAToBGRA( Array2DView< ubyte4 > input, Array2DView< ubyte4 > output )
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
			ubyte4 rgba = input( x, y );
			ubyte4 bgra = { rgba.z, rgba.y, rgba.x, rgba.w };			
			output( x, y ) = bgra;
		}
	}
}

// static
void Swizzle::RGBAToARGB( Array2DView< ubyte4 > input, Array2DView< ubyte4 > output )
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
			ubyte4 rgba = input( x, y );
			ubyte4 argb = { rgba.w, rgba.x, rgba.y, rgba.z };
			output( x, y ) = argb;
		}
	}
}

// static
void Swizzle::RGBAToRGB( Array2DView< ubyte4 > input, Array2DView< ubyte3 > output )
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
			ubyte4 rgba = input( x, y );
			ubyte3 rgb = { rgba.x, rgba.y, rgba.z };
			output( x, y ) = rgb;
		}
	}
}

// static
void Swizzle::RGBAToBGR( Array2DView< ubyte4 > input, Array2DView< ubyte3 > output )
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
			ubyte4 rgba = input( x, y );
			ubyte3 bgr = { rgba.z, rgba.y, rgba.x };
			output( x, y ) = bgr;
		}
	}
}

void Swizzle::BGRToRGBA( Array2DView< ubyte3 > input, Array2DView< ubyte4 > output, ubyte alpha )
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
			ubyte3 bgr = input( x, y );
			ubyte4 rgba = { bgr.z, bgr.y, bgr.x, alpha };
			output( x, y ) = rgba;
		}
	}
}

void Swizzle::BGRToBGRA( Array2DView< ubyte3 > input, Array2DView< ubyte4 > output, ubyte alpha )
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
			ubyte3 bgr = input( x, y );
			ubyte4 bgra = { bgr.x, bgr.y, bgr.z, alpha };
			output( x, y ) = bgra;
		}
	}
}