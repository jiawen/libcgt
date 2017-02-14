#include "libcgt/cuda/VecmathConversions.h"

#include <cstring>

#include <vector_functions.h>

#include "libcgt/core/vecmath/Matrix3f.h"
#include "libcgt/core/vecmath/Matrix4f.h"
#include "libcgt/core/vecmath/Vector2f.h"
#include "libcgt/core/vecmath/Vector3f.h"
#include "libcgt/core/vecmath/Vector4f.h"
#include "libcgt/core/vecmath/Vector2i.h"
#include "libcgt/core/vecmath/Vector3i.h"
#include "libcgt/core/vecmath/Vector4i.h"

__host__
Vector2f from_float2( const float2& v )
{
    return{ v.x, v.y };
}

__host__
Vector3f from_float3( const float3& v )
{
    return{ v.x, v.y, v.z };
}

__host__
Vector4f from_float4( const float4& v )
{
    return{ v.x, v.y, v.z, v.w };
}

__host__
float2 make_float2( const Vector2f& v )
{
    return make_float2( v.x, v.y );
}

__host__
float3 make_float3( const Vector3f& v )
{
    return make_float3( v.x, v.y, v.z );
}

__host__
float4 make_float4( const Vector4f& v )
{
    return make_float4( v.x, v.y, v.z, v.w );
}

__host__
float3x3 make_float3x3( const Matrix3f& m )
{
    float3x3 output;
    memcpy( &output, m, 9 * sizeof( float ) );
    return output;
}

__host__
float4x4 make_float4x4( const Matrix4f& m )
{
    float4x4 output;
    memcpy( &output, m, 16 * sizeof( float ) );
    return output;
}

__host__
Matrix3f from_float3x3( const float3x3& m )
{
    Matrix3f output;
    memcpy( &output, m.m_elements, 9 * sizeof( float ) );
    return output;
}

__host__
Matrix4f from_float4x4( const float4x4& m )
{
    Matrix4f output;
    memcpy( &output, m.m_elements, 16 * sizeof( float ) );
    return output;
}

__host__
Vector2i from_int2( const int2& v )
{
    return{ v.x, v.y };
}

__host__
Vector3i from_int3( const int3& v )
{
    return{ v.x, v.y, v.z };
}

__host__
Vector4i from_int4( const int4& v )
{
    return{ v.x, v.y, v.z, v.w };
}

__host__
int2 make_int2( const Vector2i& v )
{
    return make_int2( v.x, v.y );
}

__host__
int3 make_int3( const Vector3i& v )
{
    return make_int3( v.x, v.y, v.z );
}

__host__
int4 make_int4( const Vector4i& v )
{
    return make_int4( v.x, v.y, v.z, v.w );
}
