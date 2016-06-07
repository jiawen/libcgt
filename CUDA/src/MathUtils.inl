namespace libcgt { namespace cuda { namespace math {

int sign( int x )
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

int sign( float x )
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

int3 sign( const float3& v )
{
    return make_int3( sign( v.x ), sign( v.y ), sign( v.z ) );
}

int3 isPositive( const int3& v )
{
    const uint* px = reinterpret_cast< const uint* >( &( v.x ) );
    const uint* py = reinterpret_cast< const uint* >( &( v.y ) );
    const uint* pz = reinterpret_cast< const uint* >( &( v.z ) );

    return make_int3
    (
        ( ~( *px ) ) >> 31,
        ( ~( *py ) ) >> 31,
        ( ~( *pz ) ) >> 31
    );
}

int3 isPositive( const float3& v )
{
    const uint* px = reinterpret_cast< const uint* >( &( v.x ) );
    const uint* py = reinterpret_cast< const uint* >( &( v.y ) );
    const uint* pz = reinterpret_cast< const uint* >( &( v.z ) );

    return make_int3
    (
        ( ~( *px ) ) >> 31,
        ( ~( *py ) ) >> 31,
        ( ~( *pz ) ) >> 31
    );
}

bool isEven( int x )
{
    return( ( x & 0x1 ) == 0 );
}

bool isOdd( int x )
{
    return( ( x & 0x1 ) == 1 );
}

int mod( int x, int N )
{
    return ( ( x % N ) + N ) % N;
}

int3 mod( const int3& v0, const int3& v1 )
{
    return make_int3
    (
        mod( v0.x, v1.x ),
        mod( v0.y, v1.y ),
        mod( v0.z, v1.z )
    );
}

int flipUD( int y, int height )
{
    return( height - y - 1 );
}

float log2Float( float x )
{
    return log( x ) / log( 2.0f );
}

// population count
uint ones32( uint x )
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

uint floorLog2( uint x )
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

uint ceilLog2( uint x )
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

int numBins( int inputLength, int binSize )
{
    float nf = ceil( static_cast< float >( inputLength ) / binSize );
    return static_cast< int >( nf );

    // benchmarking shows that float version is actually faster
    //return( ( inputLength + binSize - 1 ) / binSize );
}

int numBins( int inputLength, const dim3& binSize )
{
    return numBins( inputLength, binSize.x );
}

dim3 numBins2D( int inputWidth, int inputHeight, int binWidth, int binHeight )
{
    return dim3
    (
        numBins( inputWidth, binWidth ),
        numBins( inputHeight, binHeight ),
        1
    );
}

dim3 numBins2D( const int2& inputSize, const int2& binSize )
{
    return numBins2D( inputSize.x, inputSize.y, binSize.x, binSize.y );
}

dim3 numBins2D( int inputWidth, int inputHeight, const dim3& binSize )
{
    return numBins2D( inputWidth, inputHeight, binSize.x, binSize.y );
}

dim3 numBins2D( const int2& inputSize, const dim3& binSize )
{
    return numBins2D( inputSize.x, inputSize.y, binSize.x, binSize.y );
}

dim3 numBins3D( int inputWidth, int inputHeight, int inputDepth, int binWidth, int binHeight, int binDepth )
{
    return dim3
    (
        numBins( inputWidth, binWidth ),
        numBins( inputHeight, binHeight ),
        numBins( inputDepth, binDepth )
    );
}

dim3 numBins3D( int inputWidth, int inputHeight, int inputDepth, const dim3& binSize )
{
    return numBins3D( inputWidth, inputHeight, inputDepth, binSize.x, binSize.y, binSize.z );
}

dim3 numBins3D( const int3& inputSize, const dim3& binSize )
{
    return numBins3D( inputSize.x, inputSize.y, inputSize.z, binSize );
}

dim3 numBins3D( const int3& inputSize, const int3& binSize )
{
    return numBins3D( inputSize.x, inputSize.y, inputSize.z, binSize.x, binSize.y, binSize.z );
}

int numElementsInBin( int binIndex, int binSize, int n )
{
    // if it's not the last bin, then it's just binSize
    // otherwise, it's n % binSize
    return
    ( ( binIndex + 1 ) * binSize > n ) ?
    ( n % binSize ) : binSize;
}

bool inRectangle( int x, int y, int width, int height )
{
    return inRectangle( x, y, 0, 0, width, height );
}

bool inRectangle( int x, int y, int x0, int y0, int width, int height )
{
    return( x >= x0 && x < x0 + width && y >= y0 && y < y0 + height );
}

bool inRectangle( const int2& xy, int width, int height )
{
    return inRectangle( xy.x, xy.y, width, height );
}

bool inRectangle( const int2& xy, int x0, int y0, int width, int height )
{
    return inRectangle( xy.x, xy.y, x0, y0, width, height );
}

bool inRectangle( const int2& xy, const int2& size )
{
    return inRectangle( xy.x, xy.y, 0, 0, size.x, size.y );
}

bool inRectangle( const int2& xy, const int2& origin, const int2& size )
{
    return inRectangle( xy.x, xy.y, origin.x, origin.y, size.x, size.y );
}

bool inBox( int x, int y, int z, int width, int height, int depth )
{
    return inBox( x, y, z, 0, 0, 0, width, height, depth );
}

bool inBox( int x, int y, int z, int x0, int y0, int z0, int width, int height, int depth )
{
    return
    (
        x >= x0 && x < x0 + width &&
        y >= y0 && y < y0 + height &&
        z >= z0 && z < z0 + depth
    );
}

bool inBox( const int3& xyz, int width, int height, int depth )
{
    return inBox( xyz.x, xyz.y, xyz.z, 0, 0, 0, width, height, depth );
}

bool inBox( const int3& xyz, const int3& size )
{
    return inBox( xyz.x, xyz.y, xyz.z, 0, 0, 0, size.x, size.y, size.z );
}

bool inBox( const int3& xyz, const int3& origin, const int3& size )
{
    return inBox( xyz.x, xyz.y, xyz.z, origin.x, origin.y, origin.z, size.x, size.y, size.z );
}

int clampToRangeExclusive( int x, int origin, int size )
{
    return clamp( x, origin, origin + size - 1 );
}

int clampToRangeExclusive( int x, int size )
{
    return clampToRangeExclusive( x, 0, size );
}

int2 clampToRectangleExclusive( const int2& v, const int2& origin, const int2& size )
{
    return make_int2
    (
        clampToRangeExclusive( v.x, origin.x, size.x ),
        clampToRangeExclusive( v.y, origin.y, size.y )
    );
}

int2 clampToRectangleExclusive( const int2& v, const int2& size )
{
    return make_int2
    (
        clampToRangeExclusive( v.x, 0, size.x ),
        clampToRangeExclusive( v.y, 0, size.y )
    );
}

int3 clampToBoxExclusive( const int3& v, const int3& origin, const int3& size )
{
    return make_int3
    (
        clampToRangeExclusive( v.x, origin.x, size.x ),
        clampToRangeExclusive( v.y, origin.y, size.y ),
        clampToRangeExclusive( v.z, origin.z, size.z )
    );
}

int3 clampToBoxExclusive( const int3& v, const int3& size )
{
    return make_int3
    (
        clampToRangeExclusive( v.x, 0, size.x ),
        clampToRangeExclusive( v.y, 0, size.y ),
        clampToRangeExclusive( v.z, 0, size.z )
    );
}

float3 clampToBox( const float3& v, const float3& origin, const float3& size )
{
    float3 corner = origin + size;

    return make_float3
    (
        clamp( v.x, origin.x, corner.x ),
        clamp( v.y, origin.y, corner.y ),
        clamp( v.z, origin.z, corner.z )
    );
}

#if 0
// TODO: this doesn't compile: wtf
float3 clampToBox( const float3& v, const Box3f& box )
{
    float x = clamp( v.x, box.left(), box.right() );
    float y = clamp( v.y, box.bottom(), box.top() );
    float z = clamp( v.z, box.back(), box.front() );
    return make_float3( x, y, z );
}
#endif

int subscriptToIndex2D( int x, int y, int width )
{
    return y * width + x;
}

int subscriptToIndex2D( int x, int y, const int2& size )
{
    return subscriptToIndex2D( x, y, size.x );
}

int subscriptToIndex2D( const int2& subscript, const int2& size )
{
    return subscriptToIndex2D( subscript.x, subscript.y, size.x );
}

template< typename T >
int subscriptToIndex2DPitched( int x, int y, int rowPitch )
{
    return y * rowPitch / sizeof( T ) + x;
}

template< typename T >
int subscriptToIndex2DPitched( const int2& subscript, int rowPitch )
{
    return subscriptToIndex2DPitched< T >( subscript.x, subscript.y, rowPitch );
}

int subscriptToIndex3D( int x, int y, int z, int width, int height )
{
    return
    (
        z * width * height +
        y * height +
        x
    );
}

int subscriptToIndex3D( int x, int y, int z, const int3& size )
{
    return subscriptToIndex3D( x, y, z, size.x, size.y );
}

int subscriptToIndex3D( const int3& subscript, const int3& size )
{
    return subscriptToIndex3D( subscript.x, subscript.y, subscript.z, size.x, size.y );
}

template< typename T >
int subscriptToIndex3DPitched( int x, int y, int z, int rowPitch, int height )
{
    return z * rowPitch * height / sizeof( T ) + y * rowPitch / sizeof( T ) + x;
}

template< typename T >
int subscriptToIndex3DPitched( const int3& subscript, int rowPitch, int height )
{
    return subscriptToIndex3DPitched< T >( subscript.x, subscript.y, subscript.z, rowPitch, height );
}

int2 indexToSubscript2D( int index, int width )
{
    int y = index / width;
    int x = index - y * width;
    return make_int2( x, y );
}

int2 indexToSubscript2D( int index, const int2& size )
{
    return indexToSubscript2D( index, size.x );
}

int3 indexToSubscript3D( int index, int width, int height )
{
    int wh = width * height;
    int z = index / wh;

    int ky = index - z * wh;
    int y = ky / width;

    int x = ky - y * width;
    return make_int3( x, y, z );
}

int3 indexToSubscript3D( int index, const int3& size )
{
    return indexToSubscript3D( index, size.x, size.y );
}

int floorToInt( float x )
{
    return static_cast< int >( floor( x ) );
}

int2 floorToInt( const float2& v )
{
    return make_int2( floorToInt( v.x ), floorToInt( v.y ) );
}

int3 floorToInt( const float3& v )
{
    return make_int3( floorToInt( v.x ), floorToInt( v.y ), floorToInt( v.z ) );
}

int4 floorToInt( const float4& v )
{
    return make_int4( floorToInt( v.x ), floorToInt( v.y ), floorToInt( v.z ), floorToInt( v.w ) );
}

int ceilToInt( float x )
{
    return static_cast< int >( ceil( x ) );
}

int2 ceilToInt( const float2& v )
{
    return make_int2( ceilToInt( v.x ), ceilToInt( v.y ) );
}

int3 ceilToInt( const float3& v )
{
    return make_int3( ceilToInt( v.x ), ceilToInt( v.y ), ceilToInt( v.z ) );
}

int4 ceilToInt( const float4& v )
{
    return make_int4( ceilToInt( v.x ), ceilToInt( v.y ), ceilToInt( v.z ), ceilToInt( v.w ) );
}

int roundToInt( float x )
{
    return static_cast< int >( x + 0.5f );
}

int2 roundToInt( const float2& v )
{
    return make_int2( roundToInt( v.x ), roundToInt( v.y ) );
}

int3 roundToInt( const float3& v )
{
    return make_int3( roundToInt( v.x ), roundToInt( v.y ), roundToInt( v.z ) );
}

int4 roundToInt( const float4& v )
{
    return make_int4( roundToInt( v.x ), roundToInt( v.y ), roundToInt( v.z ), roundToInt( v.w ) );
}

uint modPowerOfTwo( uint x, uint divisor )
{
    return( x & ( divisor - 1 ) );
}

uint modPowerOfTwoWithPower( uint x, uint p )
{
    return modPowerOfTwo( x, 1 << p );
}

uint8_t floatToUByteNormalized( float f )
{
    return static_cast< uint8_t >( 255 * f );
}

float unsignedByteToFloatNormalized( uint8_t b )
{
    const float rcp = 1.f / 255.f;
    return rcp * b;
}

int8_t floatToByteSignedNormalized( float f )
{
    return static_cast< int8_t >( floor( f * 127 + 0.5f ) );
}

float3 signedByte3ToFloat3( char3 sb )
{
    const float rcp = 1.f / 127.f;

    return make_float3
    (
        rcp * sb.x,
        rcp * sb.y,
        rcp * sb.z
    );
}

float4 signedByte4ToFloat4( char4 sb )
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

uchar4 float3ToUnsignedByte4Normalized( float3 rgb, uint8_t a )
{
    return make_uchar4
    (
        static_cast< uint8_t >( 255 * rgb.x ),
        static_cast< uint8_t >( 255 * rgb.y ),
        static_cast< uint8_t >( 255 * rgb.z ),
        a
    );
}

uchar4 float4ToUnignedByte4( float4 f )
{
    const float s = 255.f;

    return make_uchar4
    (
        static_cast< uint8_t >( s * f.x ),
        static_cast< uint8_t >( s * f.y ),
        static_cast< uint8_t >( s * f.z ),
        static_cast< uint8_t >( s * f.w )
    );
}

char4 float4ToSignedByte4( float4 f )
{
    const float s = 127.f;

    return make_char4
    (
        static_cast< int8_t >( s * f.x ),
        static_cast< int8_t >( s * f.y ),
        static_cast< int8_t >( s * f.z ),
        static_cast< int8_t >( s * f.w )
    );
}

float3 signedByte4ToFloat3( char4 sb )
{
    const float rcp = 1.f / 127.f;

    return make_float3
    (
        rcp * sb.x,
        rcp * sb.y,
        rcp * sb.z
    );
}

float3 unsignedByte4ToFloat3( uchar4 b )
{
    const float rcp = 1.f / 255.f;

    return make_float3
    (
        rcp * b.x,
        rcp * b.y,
        rcp * b.z
    );
}

bool isPowerOfTwo( int x )
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

int rescaleRange( int x, int oldRange, int newRange )
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

float minimum( const float2& v )
{
    return fminf( v.x, v.y );
}

float minimum( const float3& v )
{
    return fminf( v.x, fminf( v.y, v.z ) );
}

float minimum( const float4& v )
{
    return fminf( v.x, fminf( v.y, fminf( v.z, v.w ) ) );
}

float maximum( const float2& v )
{
    return fmaxf( v.x, v.y );
}

float maximum( const float3& v )
{
    return fmaxf( v.x, fmaxf( v.y, v.z ) );
}

float maximum( const float4& v )
{
    return fmaxf( v.x, fmaxf( v.y, fmaxf( v.z, v.w ) ) );
}

__inline__ __host__ __device__
int minimumComponent( const float2& v )
{
    if( v.x <= v.y )
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

__inline__ __host__ __device__
int minimumComponent( const float3& v )
{
    int mc = 0;
    float mv = getComponent( v, 0 );

#ifdef __CUDACC__
#pragma unroll 2
#endif
    for( int i = 1; i < 3; ++i )
    {
        float value = getComponent( v, i );
        if( value < mv )
        {
            mc = i;
            mv = value;
        }
    }

    return mc;
}

__inline__ __host__ __device__
int minimumComponent( const float4& v )
{
    int mc = 0;
    float mv = getComponent( v, 0 );

#ifdef __CUDACC__
#pragma unroll 3
#endif
    for( int i = 1; i < 4; ++i )
    {
        float value = getComponent( v, i );
        if( value < mv )
        {
            mc = i;
            mv = value;
        }
    }

    return mc;
}

__inline__ __host__ __device__
int product( const int3& v )
{
    return v.x * v.y * v.z;
}

#if 0
int roundUpToNearestPowerOfTwo( int x )
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

int roundUpToNearestMultipleOfFour( int x )
{
    return ( x + 3 ) & ~0x3;
}

int roundUpToNearestMultipleOf256( int x )
{
    return ( x + 255 ) & ( ~0xff );
}

template< unsigned B >
ushort signExtend( int8_t x )
{
    short y = x;
    y = y & ( ( 1u << B ) - 1 ); // clear bits above B
    return y;
}

template< unsigned B >
int convertToSignedInt( ushort x )
{
    int r; // result
    const int m = 1u << ( B - 1 );

    int y = x; // sign extend the short into an int
    y = y & ( ( 1u << B ) - 1 );  // (Skip this if bits in x above position b are already zero.)
    r = ( y ^ m ) - m;

    return r;
}

} } } // math, cuda, libcgt