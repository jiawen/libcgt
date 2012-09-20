#pragma once

#include "common/BasicTypes.h"
#include "vecmath/Vector2i.h"
#include "vecmath/Vector3i.h"

class BitPacking
{
public:

	// packs 16-bit (x,y) into a 32-bit Morton curve index
	// From: http://graphics.stanford.edu/~seander/bithacks.html
	static uint mortonPack2D( ushort x, ushort y );

	// unpacks a 32-bit Morton curve index
	// into two 16-bit x and y values
	static void mortonUnpack2D( uint index, ushort& x, ushort& y );
	static Vector2i mortonUnpack2D( uint z );

	// packs 5-bit (x,y,z) into a 16-bit Morton curve index
	static ushort mortonPack3D_5bit( ubyte x, ubyte y, ubyte z );

	// unpacks a 16-bit Morton curve index
	// into three 5-bit x, y, and z values
	static void mortonUnpack3D_5bit( ushort index, ubyte& x, ubyte& y, ubyte& z );
	static Vector3i mortonUnpack3D_5bit( ushort index );

	// packs 10-bit (x,y,z) into a 32-bit Morton curve index
	static uint mortonPack3D_10bit( ushort x, ushort y, ushort z );

	// unpacks a 32-bit Morton curve index
	// into three 10-bit x, y, and z values
	static void mortonUnpack3D_10bit( uint index, ushort& x, ushort& y, ushort& z );
	static Vector3i mortonUnpack3D_10bit( uint index );

	// TODO: 3d morton (3x21 -> int64)
	// TODO: 2d, 3d hilbert
	// http://and-what-happened.blogspot.co.uk/2011/08/fast-2d-and-3d-hilbert-curves-and.html
};

#include "math/BitPacking.inl"