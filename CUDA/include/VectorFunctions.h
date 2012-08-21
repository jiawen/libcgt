#pragma once

#include <common/BasicTypes.h>


// makes an int2 out of a short2
static __inline__ __host__ __device__
int2 make_int2( short2 s )
{
	return make_int2( s.x, s.y );
}

// flips x and y
static __inline__ __host__ __device__
int2 yx( int2 xy )
{
	return make_int2( xy.y, xy.x );	
}

static __inline__ __host__ __device__
float3 xyz( float4 f )
{
	return make_float3( f.x, f.y, f.z );
}

static __inline__ __host__ __device__
float4 homogenized( const float4& f )
{
	float rcpW = 1.0f / f.w;
	return make_float4( rcpW * f.x, rcpW * f.y, rcpW * f.z, 1 );
}

static __inline__ __host__ __device__
float normL1( float2 v )
{
	return abs( v.x ) + abs( v.y );
}

static __inline__ __host__ __device__
float normL1( float3 v )
{
	return abs( v.x ) + abs( v.y ) + abs( v.z );
}

static __inline__ __host__ __device__
uchar4 float4ToUChar4UnsignedNormalized( float4 rgba )
{
	return make_uchar4
	(
		static_cast< ubyte >( 255 * rgba.x ),
		static_cast< ubyte >( 255 * rgba.y ),
		static_cast< ubyte >( 255 * rgba.z ),
		static_cast< ubyte >( 255 * rgba.w )
	);
}

static __inline__ __host__ __device__
uchar4 float3ToUChar4UnsignedNormalized( float3 rgb )
{
	return make_uchar4
	(
		static_cast< ubyte >( 255 * rgb.x ),
		static_cast< ubyte >( 255 * rgb.y ),
		static_cast< ubyte >( 255 * rgb.z ),
		255
	);
}

// ========== operators ==========

// ----- component-wise multiply with conversion -----

static __inline__ __host__ __device__
float3 operator * ( const int3& v1, const float3& v2 )
{
	return make_float3( v1.x * v2.x,
		v1.y * v2.y,
		v1.z * v2.z );
}

static __inline__ __host__ __device__
int3 operator * ( const int3& v1, const uint3& v2 )
{
	return make_int3( v1.x * v2.x,
		v1.y * v2.y,
		v1.z * v2.z );
}

static __inline__ __host__ __device__
float3 operator * ( const float3& v1, const uint3& v2 )
{
	return make_float3( v1.x * v2.x,
		v1.y * v2.y,
		v1.z * v2.z );
}

static __inline__ __host__ __device__
float3 operator * ( const float3& v1, const int3& v2 )
{
	return make_float3(v1.x * v2.x,
		v1.y * v2.y,
		v1.z * v2.z);
}

// ----- component-wise divide with conversion -----

static __inline__ __host__ __device__
float3 operator / ( const float3& v1, const int3& v2 )
{
	return make_float3( v1.x / v2.x,
		v1.y / v2.y,
		v1.z / v2.z );
}

static __inline__ __host__ __device__
float3 operator / ( const float3& v1, const uint3& v2 )
{
	return make_float3( v1.x / v2.x,
		v1.y / v2.y,
		v1.z / v2.z );
}

static __inline__ __host__ __device__
int3 operator / ( const int3& v1, const int3& v2 )
{
	return make_int3( v1.x / v2.x,
		v1.y / v2.y,
		v1.z / v2.z  );
}

static __inline__ __host__ __device__
int3 operator / ( const int3& v1, const uint3& v2 )
{
	return make_int3( v1.x / v2.x,
		v1.y / v2.y,
		v1.z / v2.z  );
}

static __inline__ __host__ __device__
uint3 operator / ( const uint3& v1, const uint3& v2 )
{
	return make_uint3( v1.x / v2.x,
		v1.y / v2.y,
		v1.z / v2.z );
}
