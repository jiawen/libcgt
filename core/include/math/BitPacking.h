#pragma once

#include <common/BasicTypes.h>

class BitPacking
{
public:

	// packs 16-bit (x,y) into a 32-bit z value in a Morton curve
	// From: http://graphics.stanford.edu/~seander/bithacks.html
	static uint mortonPack( ushort x, ushort y );

	// unpacks a 32-bit Morton curve packed z value
	// into two 16-bit x and y values
	static void mortonUnpack( uint z, ushort& x, ushort& y );

	// TODO: return a Vector2i

	// TODO: 3d morton (3x10 -> int32, 3x21 -> int64)
	// TODO: 2d, 3d hilbert
	// http://and-what-happened.blogspot.co.uk/2011/08/fast-2d-and-3d-hilbert-curves-and.html
};

#include "math/BitPacking.inl"