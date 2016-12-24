#pragma once

#include <common/BasicTypes.h>
#include <common/WrapConstPointerT.h>

template< typename T >
class KernelArray2D
{
public:

    // T --> void*. const T --> const void*.
    using VoidPointer = typename WrapConstPointerT< T, void >::pointer;
    // T --> uint8_t*. const T --> const uint8_t*.
    using UInt8Pointer = typename WrapConstPointerT< T, uint8_t >::pointer;

    __inline__ __device__ __host__
    KernelArray2D() = default;

    __inline__ __device__ __host__
    KernelArray2D( VoidPointer pointer, const int2& size );

    __inline__ __device__ __host__
    KernelArray2D( VoidPointer pointer, const int2& size, size_t pitch );

    __inline__ __device__
    T* pointer() const;

    __inline__ __device__
    T* elementPointer( const int2& xy ) const;

    __inline__ __device__
    T* rowPointer( int y ) const;

    __inline__ __device__
    int width() const;

    __inline__ __device__
    int height() const;

    __inline__ __device__
    size_t pitch() const;

    __inline__ __device__
    int2 size() const;

    __inline__ __device__
    T& operator [] ( const int2& xy ) const;

private:

    UInt8Pointer md_pointer = nullptr;
    int2 m_size = int2{ 0 };
    size_t m_pitch = 0;
};

#include "libcgt/cuda/KernelArray2D.inl"
