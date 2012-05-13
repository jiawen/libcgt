#pragma once

#include <vector_types.h>
#include "float4x4.h"

class Vector3f;
class Vector4f;
class Matrix4f;

__host__
Vector3f from_float3( const float3& v );

__host__
Vector4f from_float4( const float4& v );

__host__
float4x4 make_float4x4( const Matrix4f& m );

// convert a Vector4f into a float4
__host__
float4 make_float4( const Vector4f& v );