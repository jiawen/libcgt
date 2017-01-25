#pragma once

#include <cuda_runtime.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include "libcgt/core/common/Array1D.h"
#include "libcgt/core/common/ArrayView.h"
#include "libcgt/cuda/KernelArray1D.h"

// Basic wrapper interface around CUDA global memory.
// Wraps around cudaMalloc() (linear allocation).
template< typename T >
class DeviceArray1D
{
public:

    DeviceArray1D() = default;
    DeviceArray1D( size_t length );
    DeviceArray1D( const DeviceArray1D< T >& copy );
    DeviceArray1D( DeviceArray1D< T >&& move );
    DeviceArray1D< T >& operator = ( const DeviceArray1D< T >& copy );
    DeviceArray1D< T >& operator = ( DeviceArray1D< T >&& move );
    ~DeviceArray1D();

    bool isNull() const;
    bool notNull() const;

    // The number of elements in this DeviceArray1D.
    size_t length() const;

    // The number of elements in this DeviceArray1D, equivalent to length().
    size_t size() const;

    // The number of elements in this DeviceArray1D, equivalent to length().
    size_t numElements() const;

    // The number of bytes allocated for this DeviceArray1D.
    size_t sizeInBytes() const;

    // This only works if T is a CUDA builtin type such as char, int2, or
    // float4 and packed().
    // TODO: consider using std::enable_if or thrust::enable_if.
    cudaResourceDesc resourceDesc() const;

    // Resize this array, destroying its contents.
    // size.x and size.y must be positive.
    cudaError resize( size_t length );

    // Sets this array to 0 (all bytes to 0).
    cudaError clear();

    // Fills this array with the given value.
    void fill( const T& value );

    // Get an element of the array from the device, returning the value and
    // its status in err.
    // WARNING: probably slow as it incurs a cudaMemcpy.
    T get( int x, cudaError& err ) const;

    // Get an element of the array from the device, without a status code.
    // WARNING: this operation is probably slow as it incurs a cudaMemcpy.
    T operator [] ( int x ) const;

    // Sets an element of the array from the host.
    // WARNING: this operation is probably slow as it incurs a cudaMemcpy.
    cudaError set( int x, const T& value );

    const T* pointer() const;
    T* pointer();

    const T* elementPointer( size_t x ) const;
    T* elementPointer( size_t x );

    KernelArray1D< const T > readView() const;
    KernelArray1D< T > writeView();

private:

    size_t m_length = 0;
    uint8_t* m_devicePointer = nullptr;

    cudaError destroy();
};

// Copy data from src.length() to dst[ dstOffset : dstOffset + src.length() ).
// Assumes:
// - src.packed()
// - dst.notNull()
// - src.length() - dstOffset >= dst.length()
// TODO: once we have Array1DWriteView on device pointers, can get rid of
// dstOffset.
template< typename T >
cudaError copy( Array1DReadView< T > src, DeviceArray1D< T >& dst,
    size_t dstOffset = 0 );

// Copy data from src to dst.
// Assumes:
// - dst.packed()
// - src.notNull()
// src and dst must have the same size.
template< typename T >
cudaError copy( const DeviceArray1D< T >& src, Array1DWriteView< T > dst );

// Copy data from src[ srcOffset : dst.length() ) to dst.
// Assumes:
// - dst.packed()
// - src.notNull()
// - src.length() - srcOffset >= dst.length().
// TODO: once we have Array1DReadView on device pointers, can get rid of
// srcOffset.
template< typename T >
cudaError copy( const DeviceArray1D< T >& src, size_t srcOffset,
    Array1DWriteView< T > dst );

// Copy data from src to dst.
// src and dst must have the same size.
template< typename T >
cudaError copy( const DeviceArray1D< T >& src, DeviceArray1D< T >& dst );

#include "DeviceArray1D.inl"
