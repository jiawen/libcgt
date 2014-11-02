#pragma once

#include <cstdint>

#include "common/Array1DView.h"
#include "vecmath/Vector2i.h"
#include "vecmath/Vector3i.h"

class BitPacking
{
public:

	// [ b1 b0 ] --> [ b0 b1 ]
	static uint16_t byteSwap16( uint16_t x );

	// Given 2 16-bit values packed into a 32-bit word,
	// swap the bytes within each 16-bit value and
	// repack them into a 32-bit word.
	// [ b3 b2 b1 b0 ] -->
	// [ b2 b3 b0 b1 ]
	static uint32_t byteSwap16x2( uint32_t x );

	// Given 4 16-bit values packed into a 64-bit word,
	// swap the bytes within each 16-bit value and
	// repack them into a 64-bit word.
	// [ b7 b6 b5 b4 b3 b2 b1 b0 ] -->
	// [ b6 b7 b4 b5 b2 b3 b0 b1 ]
	static uint64_t byteSwap16x4( uint64_t x );

	// Given a 32-bit word, reverses the byte ordering.
	// [ b3 b2 b1 b0 ] -->
	// [ b0 b1 b2 b3 ]
	static uint32_t byteSwap32( uint32_t x );

	// Given 2 32-bit values packed into a 64-bit word,
	// swap the bytes within each 32-bit value and
	// repack them into a 64-bit word.
	// [ b7 b6 b5 b4 b3 b2 b1 b0 ] -->
	// [ b4 b5 b6 b7 b0 b1 b2 b3 ]
	static uint64_t byteSwap32x2( uint64_t x );

	// Given a 64-bit word, reverses the byte ordering.
	// [ b7 b6 b5 b4 b3 b2 b1 b0 ] -->
	// [ b0 b1 b2 b3 b4 b5 b6 b7 ]
	static uint64_t byteSwap64( uint64_t x );

	// Efficiently performs 16-bit byte swapping on an Array1DView.
	static void byteSwap16( Array1DView< uint16_t > view16 );

	// packs 16-bit (x,y) into a 32-bit Morton curve index
	// From: http://graphics.stanford.edu/~seander/bithacks.html
	static uint32_t mortonPack2D( uint16_t x, uint16_t y );

	// unpacks a 32-bit Morton curve index
	// into two 16-bit x and y values
	static void mortonUnpack2D( uint32_t index, uint16_t& x, uint16_t& y );
	static Vector2i mortonUnpack2D( uint32_t z );

	// packs 5-bit (x,y,z) into a 16-bit Morton curve index
	static uint16_t mortonPack3D_5bit( uint8_t x, uint8_t y, uint8_t z );

	// unpacks a 16-bit Morton curve index
	// into three 5-bit x, y, and z values
	static void mortonUnpack3D_5bit( uint16_t index, uint8_t& x, uint8_t& y, uint8_t& z );
	static Vector3i mortonUnpack3D_5bit( uint16_t index );

	// packs 10-bit (x,y,z) into a 32-bit Morton curve index
	static uint32_t mortonPack3D_10bit( uint16_t x, uint16_t y, uint16_t z );

	// unpacks a 32-bit Morton curve index
	// into three 10-bit x, y, and z values
	static void mortonUnpack3D_10bit( uint32_t index, uint16_t& x, uint16_t& y, uint16_t& z );
	static Vector3i mortonUnpack3D_10bit( uint32_t index );

	// TODO: 3d morton (3x21 -> int64)
	// TODO: 2d, 3d hilbert
	// http://and-what-happened.blogspot.co.uk/2011/08/fast-2d-and-3d-hilbert-curves-and.html
};

#include "math/BitPacking.inl"