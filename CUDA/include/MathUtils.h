#pragma once

#include <common/BasicTypes.h>

#define MAX_UNSIGNED_SHORT ( 1 << 16 )
#define SUB2IND( x, y, w ) ( ( y ) * ( w ) + ( x ) )

namespace libcgt
{
	namespace cuda
	{
		__host__ __device__ __inline__
		bool isEven( int x );

		__host__ __device__ __inline__
		bool isOdd( int x );

		__host__ __device__ __inline__
		float log2Float( float x );

		// population count
		__host__ __device__ __inline__
		uint ones32( uint x );

		__host__ __device__ __inline__
		uint floorLog2( uint x );

		__host__ __device__ __inline__
		uint ceilLog2( uint x );

		// returns the next random integer between lo (inclusive) and hi (exclusive)
		__host__ __inline__
		int nextRandomIntRange( int lo, int hi );

		__host__ __inline__
		ushort nextRandomUnsignedShortRange( ushort lo, ushort hi );

		// returns a random float in [0,1]
		__host__ __inline__
		float nextRandomFloat();

		// returns a random float4 in [0,1]^4
		__host__ __inline__
		float4 nextRandomFloat4();

		// given an array of length "arraySize"
		// and bins of size "binSize"
		// computes the minimum number of bins needed
		// to cover all arraySize elements
		// (the last bin may not be full)
		__inline__ __host__ __device__
		int numBins( int arraySize, int binSize );

		// same as numBins, but in 2D
		// output.z = 1
		__inline__ __host__ __device__
		dim3 numBins2D( int inputWidth, int inputHeight, dim3 blockSize );

		// same as numBins, but in 3D
		__inline__ __host__ __device__
		dim3 numBins3D( dim3 inputSize, dim3 blockSize );

		// given the index of a bin "binIndex"
		// where bins are size "binSize"
		// and where there's a total number n in the array
		// tells you how many elements are in the "binIndex"-th bin
		// (will be binSize for all but the last bin)
		__inline__ __host__ __device__
		int numElementsInBin( int binIndex, int binSize, int n );

		__host__ __device__ __inline__
		int roundToInt( float x );

		// efficiently computes x % divisor, where divisor is a power of two
		// by the magic formula:
		// x % ( 2^p ) = x & ( 2^p - 1 )
		__host__ __device__ __inline__
		uint modPowerOfTwoWithDivisor( uint x, uint divisor );

		// efficiently computes x % (2^p) by the magic formula:
		// x % ( 2^p ) = x & ( 2^p - 1 )
		__host__ __device__ __inline__
		uint modPowerOfTwoWithPower( uint x, uint p );

		// ----- float <--> byte -----

		// converts a float in [0,1] to
		// a byte in [0,255]
		// the behavior for f outside [0,1] is undefined
		__host__ __device__ __inline__
		ubyte floatToUByteNormalized( float f );

		// converts a byte in [0,255] to
		// a float in [0,1]
		__host__ __device__ __inline__
		float unsignedByteToFloatNormalized( ubyte b );

		// converts a float in [-1,1] to
		// a byte in [-127,127]
		// the behavior for f outside [-1,1] is undefined
		__host__ __device__ __inline__
		sbyte floatToByteSignedNormalized( float f );		

		// ----- float3 <--> byte3 -----

		__host__ __device__ __inline__
		float3 signedByte3ToFloat3( char3 sb );		

		// ----- float4 <--> byte4 -----

		// f is a float in [0,1]
		// convert it to a unsigned byte in [0,255]
		__host__ __device__ __inline__
		uchar4 float4ToUnignedByte4( float4 f );

		// f is a float in [-1,1]
		// convert it to a signed byte in [-127,127]
		__host__ __device__ __inline__
		char4 float4ToSignedByte4( float4 f );		

		// sb is a signed byte in [-127,127]
		// convert it to a floating point in [-1,1]
		__host__ __device__ __inline__
		float4 signedByte4ToFloat4( char4 sb );

		// ----- float3 --> byte4 -----

		__inline__ __host__ __device__
		uchar4 float3ToUnsignedByte4Normalized( float3 rgb, ubyte a = 255 );

		// ----- byte4 --> float3 -----

		__host__ __device__ __inline__
		float3 signedByte4ToFloat3( char4 sb );		


		// converts a byte in [0,255] to a float in [0,1],
		// dropping the last component
		__host__ __device__ __inline__
		float3 unsignedByte4ToFloat3( uchar4 b );

		__host__ __device__ __inline__
		bool isPowerOfTwo( int x );

		// rescale x in [0,oldRange)
		// to [0,newRange)
		__host__ __device__ __inline__
		int rescaleRange( int x, int oldRange, int newRange );

		__host__ __device__ __inline__
		int roundUpToNearestPowerOfTwo( int x );

		// leave x alone if it's already a multiple of 4
		__host__ __device__ __inline__
		int roundUpToNearestMultipleOfFour( int x );

		__host__ __device__ __inline__
		int roundUpToNearestMultipleOf256( int x );

		// sign extend a byte to B bits long
		// and store the result in a short
		// bits [15:B] (0-based indexing) are set to 0
		template< unsigned B >
		__host__ __device__ __inline__
		ushort signExtend( sbyte x );

		// converts a B bit quantity, stored in a short
		// to a fully sign extended int
		template< unsigned B >
		__host__ __device__ __inline__
		int convertToSignedInt( ushort x );
	}
}


__host__ __device__ __inline__
bool libcgt::cuda::isEven( int x )
{	
	return( ( x & 0x1 ) == 0 );
}

__host__ __device__ __inline__
bool libcgt::cuda::isOdd( int x )
{
	return( ( x & 0x1 ) == 1 );
}

__host__ __device__ __inline__
float libcgt::cuda::log2Float( float x )
{
	return log( x ) / log( 2.0f );
}

// population count
__host__ __device__ __inline__
uint libcgt::cuda::ones32( uint x )
{
	// 32-bit recursive reduction using SWAR...
	// but first step is mapping 2-bit values
	// into sum of 2 1-bit values in sneaky way
	x -= ((x >> 1) & 0x55555555);
	x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
	x = (((x >> 4) + x) & 0x0f0f0f0f);
	x += (x >> 8);
	x += (x >> 16);
	return( x & 0x0000003f );
}

__host__ __device__ __inline__
uint libcgt::cuda::floorLog2( uint x )
{
	x |= ( x >> 1 );
	x |= ( x >> 2 );
	x |= ( x >> 4 );
	x |= ( x >> 8 );
	x |= ( x >> 16 );

	// return -1 when taking log of 0
	// return( ones32( x ) - 1 );

	// return 0 when it's 0
	return( ones32( x >> 1 ) );
}

__host__ __device__ __inline__
uint libcgt::cuda::ceilLog2( uint x )
{
	int y = (x & (x - 1));

	y |= -y;
	y >>= 31;
	x |= (x >> 1);
	x |= (x >> 2);
	x |= (x >> 4);
	x |= (x >> 8);
	x |= (x >> 16);

	// return -1 when taking log of 0
	// return( ones32( x ) - 1 - y );

	return( ones32( x >> 1 ) - y );
}


__host__ __inline__
int libcgt::cuda::nextRandomIntRange( int lo, int hi )
{
	int range = hi - lo;
	int randInt = rand() % range;
	return lo + randInt;
}

__host__ __inline__
ushort libcgt::cuda::nextRandomUnsignedShortRange( ushort lo, ushort hi )
{
	return static_cast< ushort >( nextRandomIntRange( lo, hi ) );
}

__host__ __inline__
float libcgt::cuda::nextRandomFloat()
{
	float u = rand() / static_cast< float >( RAND_MAX );
	return u;
}

// returns a random float4 in [0,1]^4
__host__ __inline__
float4 libcgt::cuda::nextRandomFloat4()
{
	float4 u4 = make_float4( nextRandomFloat(), nextRandomFloat(), nextRandomFloat(), nextRandomFloat() );
	return u4;
}

__inline__ __host__ __device__
int libcgt::cuda::numBins( int arraySize, int binSize )
{
	float nf = ceil( static_cast< float >( arraySize ) / binSize );
	return static_cast< int >( nf );

	// benchmarking shows that float version is actually faster
	//return( ( arraySize + binSize - 1 ) / binSize );
}

__inline__ __host__ __device__
dim3 libcgt::cuda::numBins2D( int inputWidth, int inputHeight, dim3 blockSize )
{
	return dim3
		(
		numBins( inputWidth, blockSize.x ),
		numBins( inputHeight, blockSize.y ),
		1
		);
}

__inline__ __host__ __device__
dim3 libcgt::cuda::numBins3D( dim3 inputSize, dim3 blockSize )
{
	return dim3
	(
		numBins( inputSize.x, blockSize.x ),
		numBins( inputSize.y, blockSize.y ),
		numBins( inputSize.z, blockSize.z )
	);
}

__inline__ __host__ __device__
int libcgt::cuda::numElementsInBin( int binIndex, int binSize, int n )
{
	// if it's not the last bin, then it's just binSize
	// otherwise, it's n % binSize
	return
		( ( binIndex + 1 ) * binSize > n ) ?
		( n % binSize ) : binSize;
}

__host__ __device__ __inline__
int libcgt::cuda::roundToInt( float x )
{
	return static_cast< int >( x + 0.5f );
}

__host__ __device__ __inline__
uint libcgt::cuda::modPowerOfTwoWithDivisor( uint x, uint divisor )
{
	return( x & ( divisor - 1 ) );
}

__host__ __device__ __inline__
uint libcgt::cuda::modPowerOfTwoWithPower( uint x, uint p )
{
	return modPowerOfTwoWithDivisor( x, 1 << p );
}

__host__ __device__ __inline__
ubyte libcgt::cuda::floatToUByteNormalized( float f )
{
	return static_cast< ubyte >( 255 * f );
}

__host__ __device__ __inline__
float libcgt::cuda::unsignedByteToFloatNormalized( ubyte b )
{
	const float rcp = 1.f / 255.f;
	return rcp * b;
}

__host__ __device__ __inline__
	sbyte libcgt::cuda::floatToByteSignedNormalized( float f )
{
	return static_cast< sbyte >( floor( f * 127 + 0.5f ) );
}

__host__ __device__ __inline__
float3 libcgt::cuda::signedByte3ToFloat3( char3 sb )
{
	const float rcp = 1.f / 127.f;

	return make_float3
	(
		rcp * sb.x,
		rcp * sb.y,
		rcp * sb.z
	);
}

__host__ __device__ __inline__
float4 libcgt::cuda::signedByte4ToFloat4( char4 sb )
{
	const float rcp = 1.f / 127.f;

	return make_float4
	(
		rcp * sb.x,
		rcp * sb.y,
		rcp * sb.z,
		rcp * sb.w
	);
}

__host__ __device__ __inline__
uchar4 libcgt::cuda::float3ToUnsignedByte4Normalized( float3 rgb, ubyte a )
{
	return make_uchar4
	(
		static_cast< ubyte >( 255 * rgb.x ),
		static_cast< ubyte >( 255 * rgb.y ),
		static_cast< ubyte >( 255 * rgb.z ),
		a
	);
}

__host__ __device__ __inline__
uchar4 libcgt::cuda::float4ToUnignedByte4( float4 f )
{
	const float s = 255.f;

	return make_uchar4
	(
		static_cast< ubyte >( s * f.x ),
		static_cast< ubyte >( s * f.y ),
		static_cast< ubyte >( s * f.z ),
		static_cast< ubyte >( s * f.w )
	);
}

__host__ __device__ __inline__
	char4 libcgt::cuda::float4ToSignedByte4( float4 f )
{
	const float s = 127.f;

	return make_char4
	(
		static_cast< sbyte >( s * f.x ),
		static_cast< sbyte >( s * f.y ),
		static_cast< sbyte >( s * f.z ),
		static_cast< sbyte >( s * f.w )
	);
}

__host__ __device__ __inline__
float3 libcgt::cuda::signedByte4ToFloat3( char4 sb )
{
	const float rcp = 1.f / 127.f;

	return make_float3
		(
		rcp * sb.x,
		rcp * sb.y,
		rcp * sb.z
		);
}

__host__ __device__ __inline__
float3 libcgt::cuda::unsignedByte4ToFloat3( uchar4 b )
{
	const float rcp = 1.f / 255.f;

	return make_float3
		(
		rcp * b.x,
		rcp * b.y,
		rcp * b.z
		);
}

__host__ __device__ __inline__
bool libcgt::cuda::isPowerOfTwo( int x )
{
	if( x < 1 )
	{
		return false;
	}
	else
	{
		return( ( x & ( x - 1 ) ) == 0 );

		// for unsigned int, the following takes care of 0 without branching
		// !(v & (v - 1)) && v;
	}
}

__host__ __device__ __inline__
int libcgt::cuda::rescaleRange( int x, int oldRange, int newRange )
{
	float f = static_cast< float >( x ) / oldRange;
	int g = static_cast< int >( f * newRange );
	if( g < 0 )
	{
		g = 0;
	}
	if( g >= newRange )
	{
		g = newRange - 1;
	}
	return g;
}

__host__ __device__ __inline__
int libcgt::cuda::roundUpToNearestPowerOfTwo( int x )
{
	if( x < 1 )
	{
		return 1;
	}

	float log2x = log2Float( static_cast< float >( x ) );
	float nextLog2 = ceil( log2x );
	return static_cast< int >( exp2f( nextLog2 ) );
}

__host__ __device__ __inline__
int libcgt::cuda::roundUpToNearestMultipleOfFour( int x )
{
	return ( x + 3 ) & ~0x3;
}

__host__ __device__ __inline__
int libcgt::cuda::roundUpToNearestMultipleOf256( int x )
{
	return ( x + 255 ) & ( ~0xff );
}

template< unsigned B >
__host__ __device__ __inline__
ushort libcgt::cuda::signExtend( sbyte x )
{	
	short y = x;
	y = y & ( ( 1u << B ) - 1 ); // clear bits above B
	return y;
}

template< unsigned B >
__host__ __device__ __inline__
int libcgt::cuda::convertToSignedInt( ushort x )
{
	int r; // result
	const int m = 1u << ( B - 1 );

	int y = x; // sign extend the short into an int
	y = y & ( ( 1u << B ) - 1 );  // (Skip this if bits in x above position b are already zero.)
	r = ( y ^ m ) - m;

	return r;
}
