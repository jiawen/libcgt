#pragma once

#include <common/BasicTypes.h>
#include <helper_math.h>

#include "Rect2i.h"
#include "VectorFunctions.h"

#define MAX_UNSIGNED_SHORT ( 1 << 16 )
#define SUB2IND( x, y, w ) ( ( y ) * ( w ) + ( x ) )

namespace libcgt { namespace cuda { namespace math {

__inline__ __host__ __device__
int2 incrementX( const int2& xy );

__inline__ __host__ __device__
int2 incrementY( const int2& xy );

__inline__ __host__ __device__
int3 incrementX( const int3& xyz );

__inline__ __host__ __device__
int3 incrementY( const int3& xyz );

__inline__ __host__ __device__
int3 incrementZ( const int3& xyz );

__inline__ __host__ __device__
int sign( int x );

__inline__ __host__ __device__
int sign( float x );

__inline__ __host__ __device__
int3 sign( const float3& v );

__inline__ __host__ __device__
int3 isPositive( const int3& v );

__inline__ __host__ __device__
int3 isPositive( const float3& v );

__inline__ __host__ __device__
bool isEven( int x );

__inline__ __host__ __device__
bool isOdd( int x );

__inline__ __host__ __device__
int mod( int x, int N );

__inline__ __host__ __device__
int3 mod( const int3& v0, const int3& v1 );

__inline__ __host__ __device__
int flipY( int y, int height );

__inline__ __host__ __device__
float log2Float( float x );

// population count
__inline__ __host__ __device__
uint ones32( uint x );

__inline__ __host__ __device__
uint floorLog2( uint x );

__inline__ __host__ __device__
uint ceilLog2( uint x );

// when given a task of size inputLength (typically an array)
// and bins of size "binSize"
// computes the minimum number of bins needed
// to cover all inputLength elements
// (the last bin may not be full)
__inline__ __host__ __device__
int numBins( int inputLength, int binSize );

// same as numBins, binSize.y and binSize.z are ignored
__inline__ __host__ __device__
int numBins( int inputLength, const dim3& binSize );

// same as numBins, but in 2D, binSize.z is ignored
// output.z = 1
__inline__ __host__ __device__
dim3 numBins2D( int inputWidth, int inputHeight, int binWidth, int binHeight );

__inline__ __host__ __device__
dim3 numBins2D( const int2& inputSize, const int2& binSize );

__inline__ __host__ __device__
dim3 numBins2D( int inputWidth, int inputHeight, const dim3& binSize );

__inline__ __host__ __device__
dim3 numBins2D( const int2& inputSize, const dim3& binSize );

// same as numBins, but in 3D
__inline__ __host__ __device__
dim3 numBins3D( int inputWidth, int inputHeight, int inputDepth, int binWidth, int binHeight, int binDepth );

__inline__ __host__ __device__
dim3 numBins3D( int inputWidth, int inputHeight, int inputDepth, const dim3& binSize );

__inline__ __host__ __device__
dim3 numBins3D( const int3& inputSize, const dim3& binSize );

__inline__ __host__ __device__
dim3 numBins3D( const int3& inputSize, const int3& binSize );

// given the index of a bin "binIndex"
// where bins are size "binSize"
// and where there's a total number n in the array
// tells you how many elements are in the "binIndex"-th bin
// (will be binSize for all but the last bin)
__inline__ __host__ __device__
int numElementsInBin( int binIndex, int binSize, int n );

__inline__ __host__ __device__
bool inRectangle( const int2& xy, const Rect2i& r );

__inline__ __host__ __device__
bool inBox( int x, int y, int z, int width, int height, int depth );

__inline__ __host__ __device__
bool inBox( int x, int y, int z, int x0, int y0, int z0, int width, int height, int depth );

__inline__ __host__ __device__
bool inBox( const int3& xyz, int width, int height, int depth );

__inline__ __host__ __device__
bool inBox( const int3& xyz, const int3& size );

__inline__ __host__ __device__
bool inBox( const int3& xyz, const int3& origin, const int3& size );

__inline__ __host__ __device__
int clampToRangeExclusive( int x, int origin, int size );

__inline__ __host__ __device__
int clampToRangeExclusive( int x, int size );

__inline__ __host__ __device__
int2 clampToRectangleExclusive( const int2& v, const int2& origin, const int2& size );

__inline__ __host__ __device__
int2 clampToRectangleExclusive( const int2& v, const int2& size );

__inline__ __host__ __device__
int3 clampToBoxExclusive( const int3& v, const int3& origin, const int3& size );

__inline__ __host__ __device__
int3 clampToBoxExclusive( const int3& v, const int3& size );

//__inline__ __host__ __device__
//float3 clampToBox( const float3& v, const Box3f& box );

__inline__ __host__ __device__
float3 clampToBox( const float3& v, const float3& origin, const float3& size );

__inline__ __host__ __device__
int subscriptToIndex2D( int x, int y, int width );

__inline__ __host__ __device__
int subscriptToIndex2D( int x, int y, const int2& size );

__inline__ __host__ __device__
int subscriptToIndex2D( const int2& subscript, const int2& size );

template< typename T >
__inline__ __host__ __device__
int subscriptToIndex2DPitched( int x, int y, int rowPitch );

template< typename T >
__inline__ __host__ __device__
int subscriptToIndex2DPitched( const int2& subscript, int rowPitch );

__inline__ __host__ __device__
int subscriptToIndex3D( int x, int y, int z, int width, int height );

__inline__ __host__ __device__
int subscriptToIndex3D( int x, int y, int z, const int3& size );

__inline__ __host__ __device__
int subscriptToIndex3D( const int3& subscript, const int3& size );

template< typename T >
__inline__ __host__ __device__
int subscriptToIndex3DPitched( int x, int y, int z, int rowPitch, int height );

template< typename T >
__inline__ __host__ __device__
int subscriptToIndex3DPitched( const int3& subscript, int rowPitch, int height );

__inline__ __host__ __device__
int2 indexToSubscript2D( int index, int width );

__inline__ __host__ __device__
int2 indexToSubscript2D( int index, const int2& size );

__inline__ __host__ __device__
int3 indexToSubscript3D( int index, int width, int height );

__inline__ __host__ __device__
int3 indexToSubscript3D( int index, const int3& size );

__inline__ __host__ __device__
int floorToInt( float x );

__inline__ __host__ __device__
int2 floorToInt( const float2& v );

__inline__ __host__ __device__
int3 floorToInt( const float3& v );

__inline__ __host__ __device__
int4 floorToInt( const float4& v );

__inline__ __host__ __device__
int ceilToInt( float x );

__inline__ __host__ __device__
int2 ceilToInt( const float2& v );

__inline__ __host__ __device__
int3 ceilToInt( const float3& v );

__inline__ __host__ __device__
int4 ceilToInt( const float4& v );

__inline__ __host__ __device__
int roundToInt( float x );

__inline__ __host__ __device__
int2 roundToInt( const float2& v );

__inline__ __host__ __device__
int3 roundToInt( const float3& v );

__inline__ __host__ __device__
int4 roundToInt( const float4& v );

// efficiently computes x % divisor, where divisor is a power of two
// by the magic formula:
// x % ( 2^p ) = x & ( 2^p - 1 ) = x & ( divisor - 1 )
__inline__ __host__ __device__
uint modPowerOfTwo( uint x, uint divisor );

// efficiently computes x % (2^p) by the magic formula:
// x % ( 2^p ) = x & ( 2^p - 1 ) = x & ( ( 1 << p ) - 1 )
__inline__ __host__ __device__
uint modPowerOfTwoWithPower( uint x, uint p );

// ----- float <--> byte -----

// converts a float in [0,1] to
// a byte in [0,255]
// the behavior for f outside [0,1] is undefined
__inline__ __host__ __device__
uint8_t floatToUByteNormalized( float f );

// converts a byte in [0,255] to
// a float in [0,1]
__inline__ __host__ __device__
float unsignedByteToFloatNormalized( uint8_t b );

// converts a float in [-1,1] to
// a byte in [-127,127]
// the behavior for f outside [-1,1] is undefined
__inline__ __host__ __device__
int8_t floatToByteSignedNormalized( float f );

// ----- float3 <--> byte3 -----

__inline__ __host__ __device__
float3 signedByte3ToFloat3( char3 sb );

// ----- float4 <--> byte4 -----

// f is a float in [0,1]
// convert it to a unsigned byte in [0,255]
__inline__ __host__ __device__
uchar4 float4ToUnignedByte4( float4 f );

// f is a float in [-1,1]
// convert it to a signed byte in [-127,127]
__inline__ __host__ __device__
char4 float4ToSignedByte4( float4 f );

// sb is a signed byte in [-127,127]
// convert it to a floating point in [-1,1]
__inline__ __host__ __device__
float4 signedByte4ToFloat4( char4 sb );

// ----- float3 --> byte4 -----

__inline__ __host__ __device__
uchar4 float3ToUnsignedByte4Normalized( float3 rgb, uint8_t a = 255 );

// ----- byte4 --> float3 -----

__inline__ __host__ __device__
float3 signedByte4ToFloat3( char4 sb );


// converts a byte in [0,255] to a float in [0,1],
// dropping the last component
__inline__ __host__ __device__
float3 unsignedByte4ToFloat3( uchar4 b );

__inline__ __host__ __device__
bool isPowerOfTwo( int x );

// rescale x in [0,oldRange)
// to [0,newRange)
__inline__ __host__ __device__
int rescaleRange( int x, int oldRange, int newRange );

__inline__ __host__ __device__
float minimum( const float2& v );

__inline__ __host__ __device__
float minimum( const float3& v );

__inline__ __host__ __device__
float minimum( const float4& v );

__inline__ __host__ __device__
float maximum( const float2& v );

__inline__ __host__ __device__
float maximum( const float3& v );

__inline__ __host__ __device__
float maximum( const float4& v );

__inline__ __host__ __device__
int minimumComponent( const float2& v );

__inline__ __host__ __device__
int minimumComponent( const float3& v );

__inline__ __host__ __device__
int minimumComponent( const float4& v );

__inline__ __host__ __device__
int product( const int3& v );

#if 0
// TODO: fix this
__inline__ __host__ __device__
int roundUpToNearestPowerOfTwo( int x );
#endif

// leave x alone if it's already a multiple of 4
__inline__ __host__ __device__
int roundUpToNearestMultipleOfFour( int x );

__inline__ __host__ __device__
int roundUpToNearestMultipleOf256( int x );

// sign extend a byte to B bits long
// and store the result in a short
// bits [15:B] (0-based indexing) are set to 0
template< unsigned B >
__inline__ __host__ __device__
ushort signExtend( int8_t x );

// converts a B bit quantity, stored in a short
// to a fully sign extended int
template< unsigned B >
__inline__ __host__ __device__
int convertToSignedInt( ushort x );

} } } // math, cuda, libcgt

#include "MathUtils.inl"
