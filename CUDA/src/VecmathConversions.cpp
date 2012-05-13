#include "VecmathConversions.h"

#include <cstring>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>
#include <vecmath/Matrix4f.h>
#include <vector_functions.h>

__host__
Vector3f from_float3( const float3& v )
{
	return Vector3f( v.x, v.y, v.z );
}

__host__
Vector4f from_float4( const float4& v )
{
	return Vector4f( v.x, v.y, v.z, v.w );
}

__host__
float4 make_float4( const Vector4f& v )
{
	return make_float4( v.x, v.y, v.z, v.w );
}

__host__
float4x4 make_float4x4( const Matrix4f& m )
{
	float4x4 output;
	memcpy( &output, m, 16 * sizeof( float ) );
	return output;
}