#pragma once

#include <vector_types.h>
#include "float4x4.h"

class Vector2f;
class Vector3f;
class Vector4f;
class Matrix4f;

__host__
Vector2f from_float2( const float2& v );

__host__
Vector3f from_float3( const float3& v );

__host__
Vector4f from_float4( const float4& v );

__host__
Matrix4f from_float4x4( const float4x4& m );

__host__
float2 make_float2( const Vector2f& v );

__host__
float3 make_float3( const Vector3f& v );

__host__
float4 make_float4( const Vector4f& v );

__host__
float4x4 make_float4x4( const Matrix4f& m );