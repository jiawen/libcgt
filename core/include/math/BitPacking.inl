// static
uint BitPacking::mortonPack2D( ushort x, ushort y )
{
	static const unsigned int B[] = { 0x55555555, 0x33333333, 0x0f0f0f0f, 0x00ff00ff };
	static const unsigned int S[] = { 1, 2, 4, 8 };

	// Interleave lower 16 bits of x and y, so the bits of x
	// are in the even positions and bits from y in the odd.
	// z gets the resulting 32-bit Morton Number.  
	uint x32 = x;
	uint y32 = y;
	uint index;

	x32 = ( x32 | ( x32 << S[3] ) ) & B[3];
	x32 = ( x32 | ( x32 << S[2] ) ) & B[2];
	x32 = ( x32 | ( x32 << S[1] ) ) & B[1];
	x32 = ( x32 | ( x32 << S[0] ) ) & B[0];

	y32 = ( y32 | ( y32 << S[3] ) ) & B[3];
	y32 = ( y32 | ( y32 << S[2] ) ) & B[2];
	y32 = ( y32 | ( y32 << S[1] ) ) & B[1];
	y32 = ( y32 | ( y32 << S[0] ) ) & B[0];

	index = x32 | ( y32 << 1 );

	return index;
}

// static
void BitPacking::mortonUnpack2D( uint index, ushort& x, ushort& y )
{
	uint64 index64 = index;

	// pack into 64-bits:
	// [y | x]
	// 0xAAAAAAAA extracts odd bits
	// 0x55555555 extracts even bits
	// only shift y by 31 since we want the bottom bit (which was the 1st, not the 0th)
	// to be in bit w[32]
	uint64 w = ( ( index64 & 0xAAAAAAAA ) << 31 ) | ( index64 & 0x55555555 );

	w = ( w | ( w >> 1 ) ) & 0x3333333333333333;
	w = ( w | ( w >> 2 ) ) & 0x0f0f0f0f0f0f0f0f; 
	w = ( w | ( w >> 4 ) ) & 0x00ff00ff00ff00ff;
	w = ( w | ( w >> 8 ) ) & 0x0000ffff0000ffff;

	x = w & 0x000000000000ffff;
	y = ( w & 0x0000ffff00000000 ) >> 32;
}

// static
Vector2i BitPacking::mortonUnpack2D( uint index )
{
	ushort x;
	ushort y;
	mortonUnpack2D( index, x, y );
	return Vector2i( x, y );
}

// static
ushort BitPacking::mortonPack3D_5bit( ubyte x, ubyte y, ubyte z )
{
	uint index0 = x;
	uint index1 = y;
	uint index2 = z;

	index0 &= 0x0000001f;
	index1 &= 0x0000001f;
	index2 &= 0x0000001f;
	index0 *= 0x01041041;
	index1 *= 0x01041041;
	index2 *= 0x01041041;
	index0 &= 0x10204081;
	index1 &= 0x10204081;
	index2 &= 0x10204081;
	index0 *= 0x00011111;
	index1 *= 0x00011111;
	index2 *= 0x00011111;
	index0 &= 0x12490000;
	index1 &= 0x12490000;
	index2 &= 0x12490000;

	return static_cast< ushort >( ( index0 >> 16 ) | ( index1 >> 15 ) | ( index2 >> 14 ) );
}

// static
void BitPacking::mortonUnpack3D_5bit( ushort index, ubyte& x, ubyte& y, ubyte& z )
{
	uint value0 = index;
	uint value1 = ( value0 >> 1 );
	uint value2 = ( value0 >> 2 );
	
	value0 &= 0x00001249;
	value1 &= 0x00001249;
	value2 &= 0x00001249;
	value0 |= ( value0 >> 2 );
	value1 |= ( value1 >> 2 );
	value2 |= ( value2 >> 2 );
	value0 &= 0x000010c3;
	value1 &= 0x000010c3;
	value2 &= 0x000010c3;
	value0 |= ( value0 >> 4 );
	value1 |= ( value1 >> 4 );
	value2 |= ( value2 >> 4 );
	value0 &= 0x0000100f;
	value1 &= 0x0000100f;
	value2 &= 0x0000100f;
	value0 |= ( value0 >> 8 );
	value1 |= ( value1 >> 8 );
	value2 |= ( value2 >> 8 );
	value0 &= 0x0000001f;
	value1 &= 0x0000001f;
	value2 &= 0x0000001f;

	x = static_cast< ubyte >( value0 );
	y = static_cast< ubyte >( value1 );
	z = static_cast< ubyte >( value2 );
}

// static
Vector3i BitPacking::mortonUnpack3D_5bit( ushort index )
{
	ubyte x;
	ubyte y;
	ubyte z;
	mortonUnpack3D_5bit( index, x, y, z );
	return Vector3i( x, y, z );
}

// static
uint BitPacking::mortonPack3D_10bit( ushort x, ushort y, ushort z )
{
	uint index0 = x;
	uint index1 = y;
	uint index2 = z;

	index0 &= 0x000003ff;
	index1 &= 0x000003ff;
	index2 &= 0x000003ff;
	index0 |= ( index0 << 16 );
	index1 |= ( index1 << 16 );
	index2 |= ( index2 << 16 );
	index0 &= 0x030000ff;
	index1 &= 0x030000ff;
	index2 &= 0x030000ff;
	index0 |= ( index0 << 8 );
	index1 |= ( index1 << 8 );
	index2 |= ( index2 << 8 );
	index0 &= 0x0300f00f;
	index1 &= 0x0300f00f;
	index2 &= 0x0300f00f;
	index0 |= ( index0 << 4 );
	index1 |= ( index1 << 4 );
	index2 |= ( index2 << 4 );
	index0 &= 0x030c30c3;
	index1 &= 0x030c30c3;
	index2 &= 0x030c30c3;
	index0 |= ( index0 << 2 );
	index1 |= ( index1 << 2 );
	index2 |= ( index2 << 2 );
	index0 &= 0x09249249;
	index1 &= 0x09249249;
	index2 &= 0x09249249;

	return( index0 | ( index1 << 1 ) | ( index2 << 2 ) );
}

// static
void BitPacking::mortonUnpack3D_10bit( uint index, ushort& x, ushort& y, ushort& z )
{
	uint value0 = index;
	uint value1 = ( value0 >> 1 );
	uint value2 = ( value0 >> 2 );

	value0 &= 0x09249249;
	value1 &= 0x09249249;
	value2 &= 0x09249249;
	value0 |= ( value0 >> 2 );
	value1 |= ( value1 >> 2 );
	value2 |= ( value2 >> 2 );
	value0 &= 0x030c30c3;
	value1 &= 0x030c30c3;
	value2 &= 0x030c30c3;
	value0 |= ( value0 >> 4 );
	value1 |= ( value1 >> 4 );
	value2 |= ( value2 >> 4 );
	value0 &= 0x0300f00f;
	value1 &= 0x0300f00f;
	value2 &= 0x0300f00f;
	value0 |= ( value0 >> 8 );
	value1 |= ( value1 >> 8 );
	value2 |= ( value2 >> 8 );
	value0 &= 0x030000ff;
	value1 &= 0x030000ff;
	value2 &= 0x030000ff;
	value0 |= ( value0 >> 16 );
	value1 |= ( value1 >> 16 );
	value2 |= ( value2 >> 16 );
	value0 &= 0x000003ff;
	value1 &= 0x000003ff;
	value2 &= 0x000003ff;

	x = value0;
	y = value1;
	z = value2;
}

// static
Vector3i BitPacking::mortonUnpack3D_10bit( uint index )
{
	ushort x;
	ushort y;
	ushort z;
	mortonUnpack3D_10bit( index, x, y, z );
	return Vector3i( x, y, z );
}

/*
// for 32-bit architectures
// morton1 - extract even bits

uint32_t morton1(uint32_t x)
{
x = x & 0x55555555;
x = (x | (x >> 1)) & 0x33333333;
x = (x | (x >> 2)) & 0x0F0F0F0F;
x = (x | (x >> 4)) & 0x00FF00FF;
x = (x | (x >> 8)) & 0x0000FFFF;
return x;
}

// morton2 - extract odd and even bits

void morton2(uint32_t *x, uint32_t *y, uint32_t z)
{
*x = morton1(z);
*y = morton1(z >> 1);
}
*/
