#pragma once

#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>

#include "libcgt/cuda/Rect2i.h"
#include "libcgt/cuda/MathUtils.h"

namespace libcgt { namespace cuda {

class Rect2f
{
public:

    __inline__ __host__ __device__
    Rect2f();

    __inline__ __host__ __device__
    Rect2f( float left, float bottom, float width, float height );

    __inline__ __host__ __device__
    Rect2f( float width, float height );

    __inline__ __host__ __device__
    Rect2f( const float2& origin, const float2& size );

    __inline__ __host__ __device__
    Rect2f( const float2& size );

    __inline__ __host__ __device__
    float left() const;

    __inline__ __host__ __device__
    float right() const;

    __inline__ __host__ __device__
    float bottom() const;

    __inline__ __host__ __device__
    float top() const;

    __inline__ __host__ __device__
    float2 bottomLeft() const;

    __inline__ __host__ __device__
    float2 bottomRight() const;

    __inline__ __host__ __device__
    float2 topLeft() const;

    __inline__ __host__ __device__
    float2 topRight() const;

    __inline__ __host__ __device__
    float2 origin() const;

    __inline__ __host__ __device__
    float2 size() const;

    __inline__ __host__ __device__
    float area() const;

    __inline__ __host__ __device__
    Rect2i enlargedToInt() const;

    __inline__ __host__ __device__
    static bool intersect( const Rect2f& r0, const Rect2f& r1 );

    __inline__ __host__ __device__
    static bool intersect( const Rect2f& r0, const Rect2f& r1, Rect2f& intersection );

private:

    float2 m_origin;
    float2 m_size;
};

} } // cuda, libcgt

#include "libcgt/cuda/Rect2f.inl"
