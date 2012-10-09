int libcgt::cuda::sign( int x )
{
	if( x < 0 )
	{
		return -1;
	}
	if( x > 0 )
	{
		return 1;
	}
	return 0;
}

int libcgt::cuda::sign( float x )
{
	if( x < 0 )
	{
		return -1;
	}
	if( x > 0 )
	{
		return 1;
	}
	return 0;
}

int3 libcgt::cuda::sign( const float3& v )
{
	return make_int3( sign( v.x ), sign( v.y ), sign( v.z ) );
}

bool libcgt::cuda::isEven( int x )
{	
	return( ( x & 0x1 ) == 0 );
}

bool libcgt::cuda::isOdd( int x )
{
	return( ( x & 0x1 ) == 1 );
}

float libcgt::cuda::log2Float( float x )
{
	return log( x ) / log( 2.0f );
}

// population count
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

int libcgt::cuda::numBins( int inputLength, int binSize )
{
	float nf = ceil( static_cast< float >( inputLength ) / binSize );
	return static_cast< int >( nf );

	// benchmarking shows that float version is actually faster
	//return( ( inputLength + binSize - 1 ) / binSize );
}

int libcgt::cuda::numBins( int inputLength, const dim3& binSize )
{
	return numBins( inputLength, binSize.x );
}

dim3 libcgt::cuda::numBins2D( int inputWidth, int inputHeight, int binWidth, int binHeight )
{
	return dim3
	(
		numBins( inputWidth, binWidth ),
		numBins( inputHeight, binHeight ),
		1
	);
}

dim3 libcgt::cuda::numBins2D( const int2& inputSize, const int2& binSize )
{
	return numBins2D( inputSize.x, inputSize.y, binSize.x, binSize.y );
}

dim3 libcgt::cuda::numBins2D( int inputWidth, int inputHeight, const dim3& binSize )
{
	return numBins2D( inputWidth, inputHeight, binSize.x, binSize.y );
}

dim3 libcgt::cuda::numBins2D( const int2& inputSize, const dim3& binSize )
{
	return numBins2D( inputSize.x, inputSize.y, binSize.x, binSize.y );
}

dim3 libcgt::cuda::numBins3D( int inputWidth, int inputHeight, int inputDepth, int binWidth, int binHeight, int binDepth )
{
	return dim3
	(
		numBins( inputWidth, binWidth ),
		numBins( inputHeight, binHeight ),
		numBins( inputDepth, binDepth )
	);
}

dim3 libcgt::cuda::numBins3D( int inputWidth, int inputHeight, int inputDepth, const dim3& binSize )
{
	return numBins3D( inputWidth, inputHeight, inputDepth, binSize.x, binSize.y, binSize.z );
}

dim3 libcgt::cuda::numBins3D( const int3& inputSize, const dim3& binSize )
{
	return numBins3D( inputSize.x, inputSize.y, inputSize.z, binSize );
}

dim3 libcgt::cuda::numBins3D( const int3& inputSize, const int3& binSize )
{
	return numBins3D( inputSize.x, inputSize.y, inputSize.z, binSize.x, binSize.y, binSize.z );
}

int libcgt::cuda::numElementsInBin( int binIndex, int binSize, int n )
{
	// if it's not the last bin, then it's just binSize
	// otherwise, it's n % binSize
	return
	( ( binIndex + 1 ) * binSize > n ) ?
	( n % binSize ) : binSize;
}

bool libcgt::cuda::inRectangle( int x, int y, int width, int height )
{
	return libcgt::cuda::inRectangle( x, y, 0, 0, width, height );
}

bool libcgt::cuda::inRectangle( int x, int y, int x0, int y0, int width, int height )
{
	return( x >= x0 && x < x0 + width && y >= y0 && y < y0 + height );
}

bool libcgt::cuda::inRectangle( const int2& xy, int width, int height )
{
	return libcgt::cuda::inRectangle( xy.x, xy.y, width, height );
}

bool libcgt::cuda::inRectangle( const int2& xy, int x0, int y0, int width, int height )
{
	return libcgt::cuda::inRectangle( xy.x, xy.y, x0, y0, width, height );
}

bool libcgt::cuda::inRectangle( const int2& xy, const int2& size )
{
	return libcgt::cuda::inRectangle( xy.x, xy.y, 0, 0, size.x, size.y );
}

bool libcgt::cuda::inRectangle( const int2& xy, const int2& origin, const int2& size )
{
	return libcgt::cuda::inRectangle( xy.x, xy.y, origin.x, origin.y, size.x, size.y );
}

bool libcgt::cuda::inBox( int x, int y, int z, int width, int height, int depth )
{
	return libcgt::cuda::inBox( x, y, z, 0, 0, 0, width, height, depth );
}

bool libcgt::cuda::inBox( int x, int y, int z, int x0, int y0, int z0, int width, int height, int depth )
{
	return
	(
		x >= x0 && x < x0 + width &&
		y >= y0 && y < y0 + height &&
		z >= z0 && z < z0 + depth
	);
}

bool libcgt::cuda::inBox( const int3& xyz, int width, int height, int depth )
{
	return inBox( xyz.x, xyz.y, xyz.z, 0, 0, 0, width, height, depth );
}

bool libcgt::cuda::inBox( const int3& xyz, const int3& size )
{
	return inBox( xyz.x, xyz.y, xyz.z, 0, 0, 0, size.x, size.y, size.z );
}

bool libcgt::cuda::inBox( const int3& xyz, const int3& origin, const int3& size )
{
	return inBox( xyz.x, xyz.y, xyz.z, origin.x, origin.y, origin.z, size.x, size.y, size.z );
}

int libcgt::cuda::subscriptToIndex3D( int x, int y, int z, int width, int height )
{
	return
	(
		z * width * height +
		y * height +
		x
	);
}

int libcgt::cuda::subscriptToIndex3D( int x, int y, int z, const int3& size )
{
	return subscriptToIndex3D( x, y, z, size.x, size.y );
}

int libcgt::cuda::subscriptToIndex3D( const int3& subscript, const int3& size )
{
	return subscriptToIndex3D( subscript.x, subscript.y, subscript.z, size.x, size.y );
}

int3 libcgt::cuda::indexToSubscript3D( int index, int width, int height )
{
	int wh = width * height;
	int z = index / wh;

	int ky = index - z * wh;
	int y = ky / width;

	int x = ky - y * width;
	return make_int3( x, y, z );
}

int3 libcgt::cuda::indexToSubscript3D( int index, const int3& size )
{
	return indexToSubscript3D( index, size.x, size.y );
}

int libcgt::cuda::floorToInt( float x )
{
	return static_cast< int >( floor( x ) );
}

int2 libcgt::cuda::floorToInt( const float2& v )
{
	return make_int2( floorToInt( v.x ), floorToInt( v.y ) );
}

int3 libcgt::cuda::floorToInt( const float3& v )
{
	return make_int3( floorToInt( v.x ), floorToInt( v.y ), floorToInt( v.z ) );
}

int4 libcgt::cuda::floorToInt( const float4& v )
{
	return make_int4( floorToInt( v.x ), floorToInt( v.y ), floorToInt( v.z ), floorToInt( v.w ) );
}

int libcgt::cuda::ceilToInt( float x )
{
	return static_cast< int >( ceil( x ) );
}

int2 libcgt::cuda::ceilToInt( const float2& v )
{
	return make_int2( ceilToInt( v.x ), ceilToInt( v.y ) );
}

int3 libcgt::cuda::ceilToInt( const float3& v )
{
	return make_int3( ceilToInt( v.x ), ceilToInt( v.y ), ceilToInt( v.z ) );
}

int4 libcgt::cuda::ceilToInt( const float4& v )
{
	return make_int4( ceilToInt( v.x ), ceilToInt( v.y ), ceilToInt( v.z ), ceilToInt( v.w ) );
}

int libcgt::cuda::roundToInt( float x )
{
	return static_cast< int >( x + 0.5f );
}

int2 libcgt::cuda::roundToInt( const float2& v )
{
	return make_int2( roundToInt( v.x ), roundToInt( v.y ) );
}

int3 libcgt::cuda::roundToInt( const float3& v )
{
	return make_int3( roundToInt( v.x ), roundToInt( v.y ), roundToInt( v.z ) );
}

int4 libcgt::cuda::roundToInt( const float4& v )
{
	return make_int4( roundToInt( v.x ), roundToInt( v.y ), roundToInt( v.z ), roundToInt( v.w ) );
}

uint libcgt::cuda::modPowerOfTwo( uint x, uint divisor )
{
	return( x & ( divisor - 1 ) );
}

uint libcgt::cuda::modPowerOfTwoWithPower( uint x, uint p )
{
	return modPowerOfTwo( x, 1 << p );
}

ubyte libcgt::cuda::floatToUByteNormalized( float f )
{
	return static_cast< ubyte >( 255 * f );
}

float libcgt::cuda::unsignedByteToFloatNormalized( ubyte b )
{
	const float rcp = 1.f / 255.f;
	return rcp * b;
}

sbyte libcgt::cuda::floatToByteSignedNormalized( float f )
{
	return static_cast< sbyte >( floor( f * 127 + 0.5f ) );
}

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

#if 0
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
#endif

int libcgt::cuda::roundUpToNearestMultipleOfFour( int x )
{
	return ( x + 3 ) & ~0x3;
}

int libcgt::cuda::roundUpToNearestMultipleOf256( int x )
{
	return ( x + 255 ) & ( ~0xff );
}

template< unsigned B >
ushort libcgt::cuda::signExtend( sbyte x )
{	
	short y = x;
	y = y & ( ( 1u << B ) - 1 ); // clear bits above B
	return y;
}

template< unsigned B >
int libcgt::cuda::convertToSignedInt( ushort x )
{
	int r; // result
	const int m = 1u << ( B - 1 );

	int y = x; // sign extend the short into an int
	y = y & ( ( 1u << B ) - 1 );  // (Skip this if bits in x above position b are already zero.)
	r = ( y ^ m ) - m;

	return r;
}
