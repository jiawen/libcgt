#pragma once

#include <common/BasicTypes.h>
#include <common/WrapConstPointerT.h>

template< typename T >
struct KernelArray1D
{
    // T --> void*. const T --> const void*.
    using VoidPointer = typename WrapConstPointerT< T, void >::pointer;
    // T --> uint8_t*. const T --> const uint8_t*.
    using UInt8Pointer = typename WrapConstPointerT< T, uint8_t >::pointer;

    __inline__ __host__ __device__
    KernelArray1D() = default;

    __inline__ __host__ __device__
    KernelArray1D( VoidPointer pointer, size_t size,
        std::ptrdiff_t stride = sizeof( T ) );

    __inline__ __device__
    size_t width() const;

    __inline__ __device__
    size_t size() const;

    // The number of bytes between elements.
    __inline__ __device__
    std::ptrdiff_t stride() const;

    __inline__ __device__
    T* pointer() const;

    __inline__ __device__
    T* elementPointer( size_t x ) const;

    __inline__ __device__
    T& operator [] ( size_t i ) const;

private:

    UInt8Pointer md_pointer = nullptr;
    size_t m_size = 0;
    std::ptrdiff_t m_stride = 0;
};

#include "KernelArray1D.inl"
