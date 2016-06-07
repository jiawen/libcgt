#pragma once

#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>

#include "MathUtils.h"

// TODO(jiawen): unify with Box3f.h
namespace libcgt { namespace cuda {

class Box3f
{
public:

    __inline__ __host__ __device__
    Box3f() = default;

    __inline__ __host__ __device__
    explicit Box3f( const float3& size );

    __inline__ __host__ __device__
    explicit Box3f( const int3& size );

    __inline__ __host__ __device__
    float left() const;

    __inline__ __host__ __device__
    float right() const;

    __inline__ __host__ __device__
    float bottom() const;

    __inline__ __host__ __device__
    float top() const;

    __inline__ __host__ __device__
    float back() const;

    __inline__ __host__ __device__
    float front() const;

    __inline__ __host__ __device__
    float3 minimum() const;

    __inline__ __host__ __device__
    float3 maximum() const;

    __inline__ __host__ __device__
    float3 center() const;

    __inline__ __host__ __device__
    void getCorners( float3 corners[8] ) const;

    __inline__ __host__ __device__
    bool contains( float x, float y, float z ) const;

    __inline__ __host__ __device__
    bool contains( const float3& p ) const;

    __inline__ __host__ __device__
    static bool intersect( const Box3f& r0, const Box3f& r1 );

    __inline__ __host__ __device__
    static bool intersect( const Box3f& r0, const Box3f& r1, Box3f& intersection );

    // TODO(VS2015):
    //float3 m_origin = { 0 };
    //float3 m_size = { 0 };

    float3 m_origin = make_float3( 0 );
    float3 m_size = make_float3( 0 );
};

__inline__ __host__ __device__
bool intersectLine( const float3& origin, const float3& direction,
    const Box3f& box,
    float& tNear, float& tFar )
{
    // Compute t to each face.
    float3 rcpDir = 1.0f / direction;

    // Intersect the three "bottom" faces (min of the box).
    float3 tBottom = rcpDir * (box.minimum() - origin);
    // Intersect the three "top" faces (max of the box).
    float3 tTop = rcpDir * (box.maximum() - origin);

    // Find the smallest and largest distances along each axis.
    float3 tMin = fminf( tBottom, tTop );
    float3 tMax = fmaxf( tBottom, tTop );

    // tNear is the largest tMin
    tNear = libcgt::cuda::math::maximum(tMin);

    // tFar is the smallest tMax
    tFar = libcgt::cuda::math::minimum(tMax);

    return tFar > tNear;
}

} } // cuda, libcgt

#include "Box3f.inl"
