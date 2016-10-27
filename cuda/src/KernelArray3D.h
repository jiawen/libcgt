#pragma once

#include <common/BasicTypes.h>
#include <common/WrapConstPointerT.h>

template< typename T >
class KernelArray3D
{
public:

    // T --> void*. const T --> const void*.
    using VoidPointer = typename WrapConstPointerT< T, void >::pointer;
    // T --> uint8_t*. const T --> const uint8_t*.
    using UInt8Pointer = typename WrapConstPointerT< T, uint8_t >::pointer;

    __inline__ __device__ __host__
    KernelArray3D() = default;

    __inline__ __device__ __host__
    KernelArray3D( cudaPitchedPtr d_pitchedPointer, int depth );

    // wraps a KernelArray3D (with pitchedPointer) around linear device memory
    // (assumes that the memory pointed to by d_linearPointer is tightly packed,
    // if it's not, then the caller should construct a cudaPitchedPtr directly)
    __inline__ __device__ __host__
    KernelArray3D( VoidPointer d_linearPointer, const int3& size );

    __inline__ __device__
    T* pointer() const;

    __inline__ __device__
    T* elementPointer( const int3& xyz ) const;

    __inline__ __device__
    T* rowPointer( const int2& yz ) const;

    __inline__ __device__
    T* slicePointer( int z ) const;

    __inline__ __device__
    int width() const;

    __inline__ __device__
    int height() const;

    __inline__ __device__
    int depth() const;

    __inline__ __device__
    int3 size() const;

    __inline__ __device__
    size_t rowPitch() const;

    __inline__ __device__
    size_t slicePitch() const;

    __inline__ __device__
    T& operator [] ( const int3& xyz ) const;

private:

    UInt8Pointer md_pointer = nullptr;
    int3 m_size = int3{ 0 };
    size_t m_rowPitch;

};

#include "KernelArray3D.inl"
