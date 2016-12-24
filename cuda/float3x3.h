#pragma once

#include <cstdio>
#include <vector_types.h>

// 3x3 matrix, stored in column major order (FORTRAN / OpenGL style)
struct float3x3
{
    union
    {
        struct
        {
            float m00;
            float m10;
            float m20;

            float m01;
            float m11;
            float m21;

            float m02;
            float m12;
            float m22;
        };
        struct
        {
            float3 col0;
            float3 col1;
            float3 col2;
        };
        float m_elements[ 9 ];
    };
};

__inline__ __host__ __device__
float3 operator * ( const float3x3& m, const float3& v )
{
    return
    {
        m.m00 * v.x + m.m01 * v.y + m.m02 * v.z,
        m.m10 * v.x + m.m11 * v.y + m.m12 * v.z,
        m.m20 * v.x + m.m21 * v.y + m.m22 * v.z
    };
}

__inline__ __host__ __device__
float3 transformPoint( const float3x3& m, const float2& p )
{
    return
    {
        m.m00 * p.x + m.m01 * p.y + m.m02,
        m.m10 * p.x + m.m11 * p.y + m.m12,
        m.m20 * p.x + m.m21 * p.y + m.m22
    };
}

__inline__ __host__ __device__
float3 transformVector( const float3x3& m, const float2& v )
{
    return
    {
        m.m00 * v.x + m.m01 * v.y,
        m.m10 * v.x + m.m11 * v.y,
        m.m20 * v.x + m.m21 * v.y
    };
}

__inline__ __host__ __device__
void print( const float3x3& m )
{
    printf
    (
        "[ %.4f %.4f %.4f ]\n"
        "[ %.4f %.4f %.4f ]\n"
        "[ %.4f %.4f %.4f ]\n",
        m.m00, m.m01, m.m02,
        m.m10, m.m11, m.m12,
        m.m20, m.m21, m.m22
    );
}
