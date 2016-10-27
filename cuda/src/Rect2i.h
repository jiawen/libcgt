#pragma once

#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>

namespace libcgt { namespace cuda {

class Rect2i
{
public:

    int2 origin = int2{ 0, 0 };
    int2 size = int2{ 0, 0 };

    __inline__ __device__ __host__
    Rect2i() = default;

    __inline__ __device__ __host__
    Rect2i( const int2& _size );

    __inline__ __device__ __host__
    Rect2i( const int2& _origin, const int2& _size );

    __inline__ __device__ __host__
    int left() const;

    __inline__ __device__ __host__
    int right() const;

    __inline__ __device__ __host__
    int bottom() const;

    __inline__ __device__ __host__
    int top() const;

    __inline__ __device__ __host__
    int2 bottomLeft() const;

    __inline__ __device__ __host__
    int2 bottomRight() const;

    __inline__ __device__ __host__
    int2 topLeft() const;

    __inline__ __device__ __host__
    int2 topRight() const;

    __inline__ __device__ __host__
    int area() const;

};

__inline__ __device__ __host__
Rect2i flipX( const Rect2i& r, int width );

__inline__ __device__ __host__
Rect2i flipY( const Rect2i& r, int height );

// Shrink a rectangle by delta on all four sides.
__inline__ __device__ __host__
Rect2i inset( const Rect2i& r, int delta );

// Shrink a rectangle by xy.x from both left and right, and xy.y from both
// bottom and top.
__inline__ __device__ __host__
Rect2i inset( const Rect2i& r, const int2& xy );

// Returns true if the rectangle "r" contains the point "p".
__inline__ __device__ __host__
bool contains( const Rect2i& r, const int2& p );

} } // cuda, libcgt

#include "Rect2i.inl"
