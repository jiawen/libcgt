#pragma once

// 4x4 matrix, stored in column major order (FORTRAN / OpenGL style)
struct float4x4
{
	union
	{
		struct
		{
			float m00;
			float m10;
			float m20;
			float m30;

			float m01;
			float m11;
			float m21;
			float m31;

			float m02;
			float m12;
			float m22;
			float m32;

			float m03;
			float m13;
			float m23;
			float m33;
		};
		float m_elements[ 16 ];
	};
};

static __inline__ __host__ __device__
float4 operator * ( const float4x4& m, const float4& v )
{
	float4 b;

	b.x = m.m00 * v.x + m.m01 * v.y + m.m02 * v.z + m.m03 * v.w;
	b.y = m.m10 * v.x + m.m11 * v.y + m.m12 * v.z + m.m13 * v.w;
	b.z = m.m20 * v.x + m.m21 * v.y + m.m22 * v.z + m.m23 * v.w;
	b.w = m.m30 * v.x + m.m31 * v.y + m.m32 * v.z + m.m33 * v.w;

	return b;
}