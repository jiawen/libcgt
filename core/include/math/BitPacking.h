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
};

#include "math/BitPacking.inl"