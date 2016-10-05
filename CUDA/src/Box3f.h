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
    explicit Box3f( const float3& origin, const float3& size );

    __inline__ __host__ __device__
    explicit Box3f( const int3& origin, const int3& size );

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

    float3 m_origin = {};
    float3 m_size = {};
};

__inline__ __host__ __device__
bool intersectLine( const float3& origin, const float3& direction,
    const Box3f& box,
    float& tNear, float& tFar );

} } // cuda, libcgt

#include "Box3f.inl"
