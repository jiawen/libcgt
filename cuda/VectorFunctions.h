#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>

#include "libcgt/core/common/BasicTypes.h"

__inline__ __host__ __device__
uchar4 make_uchar4( uint8_t s )
{
    return make_uchar4( s, s, s, s );
}

__inline__ __host__ __device__
int3 make_int3( const char3& v )
{
    return make_int3( v.x, v.y, v.z );
}

__inline__ __host__ __device__
char3 make_char3( const int3& v )
{
    return make_char3( v.x, v.y, v.z );
}

// makes an int2 out of a short2
__inline__ __host__ __device__
int2 make_int2( const short2& s )
{
    return make_int2( s.x, s.y );
}

// flips x and y
__inline__ __host__ __device__
int2 yx( const int2& xy )
{
    return make_int2( xy.y, xy.x );
}

__inline__ __host__ __device__
float2 xy( const float3& f )
{
    return make_float2( f.x, f.y );
}

__inline__ __host__ __device__
int3 xyz( const int4& v )
{
    return make_int3( v.x, v.y, v.z );
}

__inline__ __host__ __device__
float3 xyz( const float4& v )
{
    return make_float3( v.x, v.y, v.z );
}

__inline__ __host__ __device__
int8_t getComponent( const char3& v, int component )
{
    return reinterpret_cast< const int8_t* >(&v)[component];
}

__inline__ __host__ __device__
void setComponent( char3& v, int component, int8_t value )
{
    reinterpret_cast< int8_t* >( &v )[ component ] = value;
}

__inline__ __host__ __device__
int getComponent( const int3& v, int component )
{
    return reinterpret_cast< const int* >( &v )[ component ];
}

__inline__ __host__ __device__
void setComponent( int3& v, int component, int value )
{
    reinterpret_cast< int* >( &v )[ component ] = value;
}

__inline__ __host__ __device__
int getComponent( const int4& v, int component )
{
    return reinterpret_cast< const int* >( &v )[ component ];
}

__inline__ __host__ __device__
void setComponent( int4& v, int component, int value )
{
    reinterpret_cast< int* >( &v )[ component ] = value;
}

__inline__ __host__ __device__
float getComponent( const float3& v, int component )
{
    return reinterpret_cast< const float* >( &v )[ component ];
}

__inline__ __host__ __device__
void setComponent( float3& v, int component, float value )
{
    reinterpret_cast< float* >( &v )[ component ] = value;
}

__inline__ __host__ __device__
float getComponent( const float4& v, int component )
{
    return reinterpret_cast< const float* >( &v )[ component ];
}

__inline__ __host__ __device__
void setComponent( float4& v, int component, float value )
{
    reinterpret_cast< float* >( &v )[ component ] = value;
}
__inline__ __host__ __device__
float4 homogenized( const float4& f )
{
    float rcpW = 1.0f / f.w;
    return make_float4( rcpW * f.x, rcpW * f.y, rcpW * f.z, 1 );
}

__inline__ __host__ __device__
float lengthSquared( const float2& v )
{
    return v.x * v.x + v.y * v.y;
}

__inline__ __host__ __device__
float lengthSquared( const float3& v )
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

__inline__ __host__ __device__
float normL1( const float2& v )
{
    return abs( v.x ) + abs( v.y );
}

__inline__ __host__ __device__
float normL1( const float3& v )
{
    return abs( v.x ) + abs( v.y ) + abs( v.z );
}

__inline__ __host__ __device__
void print( const int2& v )
{
    printf( "( %d, %d )\n", v.x, v.y );
}

__inline__ __host__ __device__
void print( const int3& v )
{
    printf( "( %d, %d, %d )\n", v.x, v.y, v.z );
}

__inline__ __host__ __device__
void print( const int4& v )
{
    printf( "( %d, %d, %d, %d )\n", v.x, v.y, v.z, v.w );
}

__inline__ __host__ __device__
void print( const float2& v )
{
    printf( "( %.4f, %.4f )\n", v.x, v.y );
}

__inline__ __host__ __device__
void print( const float3& v )
{
    printf( "( %.4f, %.4f, %.4f )\n", v.x, v.y, v.z );
}

static __inline__ __host__ __device__
void print( const float4& v )
{
    printf( "( %.4f, %.4f, %.4f, %.4f )\n", v.x, v.y, v.z, v.w );
}

// ========== operators ==========

// ---- component-wise add with conversion -----

__inline__ __host__ __device__
float3 operator + ( const int3& v0, const float3& v1 )
{
    return make_float3
    (
        v0.x + v1.x,
        v0.y + v1.y,
        v0.z + v1.z
    );
}

__inline__ __host__ __device__
float3 operator + ( const int3& v, float s )
{
    return make_float3
    (
        v.x + s,
        v.y + s,
        v.z + s
    );
}

// ----- component-wise multiply with conversion -----

__inline__ __host__ __device__
float3 operator * ( const char3& v0, const float3& v1 )
{
    return make_float3
    (
        v0.x * v1.x,
        v0.y * v1.y,
        v0.z * v1.z
    );
}


__inline__ __host__ __device__
float3 operator * ( const float3& v0, const char3& v1 )
{
    return v1 * v0;
}

__inline__ __host__ __device__
float3 operator * ( const int3& v0, const float3& v1 )
{
    return make_float3
    (
        v0.x * v1.x,
        v0.y * v1.y,
        v0.z * v1.z
    );
}


__inline__ __host__ __device__
float3 operator * ( const float3& v0, const int3& v1 )
{
    return v1 * v0;
}

__inline__ __host__ __device__
float3 operator * ( const uint3& v0, const float3& v1 )
{
    return make_float3
    (
        v0.x * v1.x,
        v0.y * v1.y,
        v0.z * v1.z
    );
}

__inline__ __host__ __device__
float3 operator * ( const float3& v0, const uint3& v1 )
{
    return v1 * v0;
}

__inline__ __host__ __device__
int3 operator * ( const int3& v0, const uint3& v1 )
{
    return make_int3
    (
        v0.x * v1.x,
        v0.y * v1.y,
        v0.z * v1.z
    );
}

__inline__ __host__ __device__
int3 operator * ( const uint3& v0, const int3& v1 )
{
    return v1 * v0;
}

// ----- component-wise divide with conversion -----

__inline__ __host__ __device__
int3 operator / ( const int3& v0, const int3& v1 )
{
    return make_int3
    (
        v0.x / v1.x,
        v0.y / v1.y,
        v0.z / v1.z
    );
}

__inline__ __host__ __device__
uint3 operator / ( const uint3& v0, const uint3& v1 )
{
    return make_uint3
    (
        v0.x / v1.x,
        v0.y / v1.y,
        v0.z / v1.z
    );
}

__inline__ __host__ __device__
int3 operator / ( const int3& v0, const uint3& v1 )
{
    return make_int3
    (
        v0.x / v1.x,
        v0.y / v1.y,
        v0.z / v1.z
    );
}

__inline__ __host__ __device__
int3 operator / ( const uint3& v0, const int3& v1 )
{
    return make_int3
    (
        v0.x / v1.x,
        v0.y / v1.y,
        v0.z / v1.z
    );
}

__inline__ __host__ __device__
float3 operator / ( const float3& v0, const int3& v1 )
{
    return make_float3
    (
        v0.x / v1.x,
        v0.y / v1.y,
        v0.z / v1.z
    );
}

__inline__ __host__ __device__
float3 operator / ( const float3& v0, const uint3& v1 )
{
    return make_float3
    (
        v0.x / v1.x,
        v0.y / v1.y,
        v0.z / v1.z
    );
}

__inline__ __host__ __device__
int3 operator / ( const int3& v, int s )
{
    return make_int3
    (
        v.x / s,
        v.y / s,
        v.z / s
    );
}

__inline__ __host__ __device__
float3 operator / ( float s, const int3& v )
{
    return make_float3
    (
        s / v.x,
        s / v.y,
        s / v.z
    );
}

__inline__ __host__ __device__
float3 operator / ( float s, const uint3& v )
{
    return make_float3
    (
        s / v.x,
        s / v.y,
        s / v.z
    );
}

// ---- equals -----

__inline__ __host__ __device__
bool operator == ( const int3& v0, const int3& v1 )
{
    return
    (
        v0.x == v1.x &&
        v0.y == v1.y &&
        v0.z == v1.z
    );
}
