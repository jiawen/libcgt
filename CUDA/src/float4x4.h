#pragma once

#include <cstdio>
#include <vector_types.h>

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
        struct
        {
            float4 col0;
            float4 col1;
            float4 col2;
            float4 col3;
        };
        float m_elements[ 16 ];
    };
};

__inline__ __host__ __device__
float4 operator * ( const float4x4& m, const float4& v )
{
    return
    {
        m.m00 * v.x + m.m01 * v.y + m.m02 * v.z + m.m03 * v.w,
        m.m10 * v.x + m.m11 * v.y + m.m12 * v.z + m.m13 * v.w,
        m.m20 * v.x + m.m21 * v.y + m.m22 * v.z + m.m23 * v.w,
        m.m30 * v.x + m.m31 * v.y + m.m32 * v.z + m.m33 * v.w
    };
}

__inline__ __host__ __device__
float3 transformPoint( const float4x4& m, const float3& p )
{
    return
    {
        m.m00 * p.x + m.m01 * p.y + m.m02 * p.z + m.m03,
        m.m10 * p.x + m.m11 * p.y + m.m12 * p.z + m.m13,
        m.m20 * p.x + m.m21 * p.y + m.m22 * p.z + m.m23
    };
}

__inline__ __host__ __device__
float3 transformVector( const float4x4& m, const float3& v )
{
    return
    {
        m.m00 * v.x + m.m01 * v.y + m.m02 * v.z,
        m.m10 * v.x + m.m11 * v.y + m.m12 * v.z,
        m.m20 * v.x + m.m21 * v.y + m.m22 * v.z
    };
}

__inline__ __host__ __device__
void print( const float4x4& m )
{
    printf
    (
        "[ %.4f %.4f %.4f %.4f ]\n"
        "[ %.4f %.4f %.4f %.4f ]\n"
        "[ %.4f %.4f %.4f %.4f ]\n"
        "[ %.4f %.4f %.4f %.4f ]\n",
        m.m00, m.m01, m.m02, m.m03,
        m.m10, m.m11, m.m12, m.m13,
        m.m20, m.m21, m.m22, m.m23,
        m.m30, m.m31, m.m32, m.m33
    );
}
