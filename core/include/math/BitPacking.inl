// static
uint BitPacking::mortonPack( ushort x, ushort y )
{
	static const unsigned int B[] = { 0x55555555, 0x33333333, 0x0f0f0f0f, 0x00ff00ff };
	static const unsigned int S[] = { 1, 2, 4, 8 };

	// Interleave lower 16 bits of x and y, so the bits of x
	// are in the even positions and bits from y in the odd.
	// z gets the resulting 32-bit Morton Number.  
	uint x32 = x;
	uint y32 = y;
	uint z;

	x32 = ( x32 | ( x32 << S[3] ) ) & B[3];
	x32 = ( x32 | ( x32 << S[2] ) ) & B[2];
	x32 = ( x32 | ( x32 << S[1] ) ) & B[1];
	x32 = ( x32 | ( x32 << S[0] ) ) & B[0];

	y32 = ( y32 | ( y32 << S[3] ) ) & B[3];
	y32 = ( y32 | ( y32 << S[2] ) ) & B[2];
	y32 = ( y32 | ( y32 << S[1] ) ) & B[1];
	y32 = ( y32 | ( y32 << S[0] ) ) & B[0];

	z = x32 | ( y32 << 1 );

	return z;
}

// static
void BitPacking::mortonUnpack( uint z, ushort& x, ushort& y )
{
	uint64 z64 = z;

	// pack into 64-bits:
	// [y | x]
	// 0xAAAAAAAA extracts odd bits
	// 0x55555555 extracts even bits
	// only shift y by 31 since we want the bottom bit (which was the 1st, not the 0th)
	// to be in bit w[32]
	uint64 w = ( ( z64 & 0xAAAAAAAA ) << 31 ) | ( z64 & 0x55555555 );

	w = ( w | ( w >> 1 ) ) & 0x3333333333333333;
	w = ( w | ( w >> 2 ) ) & 0x0f0f0f0f0f0f0f0f; 
	w = ( w | ( w >> 4 ) ) & 0x00ff00ff00ff00ff;
	w = ( w | ( w >> 8 ) ) & 0x0000ffff0000ffff;

	x = w & 0x000000000000ffff;
	y = ( w & 0x0000ffff00000000 ) >> 32;
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
