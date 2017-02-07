#pragma once

// TODO: add CUDA and Eigen support directly into libcgt.
// TODO: get rid of float4x4.

#include <vector_types.h>

#include "libcgt/cuda/float3x3.h"
#include "libcgt/cuda/float4x4.h"

class Matrix3f;
class Matrix4f;
class Vector2f;
class Vector3f;
class Vector4f;
class Vector2i;
class Vector3i;
class Vector4i;

__host__
Vector2f from_float2( const float2& v );

__host__
Vector3f from_float3( const float3& v );

__host__
Vector4f from_float4( const float4& v );

__host__
float2 make_float2( const Vector2f& v );

__host__
float3 make_float3( const Vector3f& v );

__host__
float4 make_float4( const Vector4f& v );

__host__
float3x3 make_float3x3( const Matrix3f& m );

__host__
float4x4 make_float4x4( const Matrix4f& m );

__host__
Matrix3f from_float3x3( const float3x3& m );

__host__
Matrix4f from_float4x4( const float4x4& m );

__host__
Vector2i from_int2( const int2& v );

__host__
Vector3i from_int3( const int3& v );

__host__
Vector4i from_int4( const int4& v );

__host__
int2 make_int2( const Vector2i& v );

__host__
int3 make_int3( const Vector3i& v );

__host__
int4 make_int4( const Vector4i& v );
