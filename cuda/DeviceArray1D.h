#pragma once

// CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>

// libcgt
#include <common/Array1D.h>

#include "libcgt/cuda/KernelArray1D.h"
#include "libcgt/cuda/ErrorChecking.h"

// Basic vector interface around CUDA global memory.
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

    size_t length() const;
    size_t sizeInBytes() const;

    // This only works if T is a CUDA builtin type such as char, int2, or
    // float4 (i.e.,
    // TODO: consider using std::enable_if or thrust::enable_if.
    cudaResourceDesc resourceDesc() const;

    // resizes the vector
    // original data is not preserved
    void resize( size_t length );

    // sets the vector to 0 (all bytes to 0)
    void clear();

    // fills this array with value
    void fill( const T& value );

    // get an element of the vector from the device
    // WARNING: probably slow as it incurs a cudaMemcpy
    T get( int x ) const;

    // get an element of the vector from the device
    // WARNING: probably slow as it incurs a cudaMemcpy
    T operator [] ( int x ) const;

    // sets an element of the vector from the host
    // WARNING: probably slow as it incurs a cudaMemcpy
    void set( int x, const T& value );

    // copy from another DeviceArray1D to this
    // this is automatically resized
    void copyFromDevice( const DeviceArray1D< T >& src );

    // copy src.length() elements from host --> device vector
    // starting at dstOffset
    // this vector is *not* resized
    //
    // dstOffset must be >= 0
    // src.length() - dstOffset must be >= dst.length()
    // src must be packed()
    //
    // returns false on failure
    bool copyFromHost( Array1DReadView< T > src, int dstOffset = 0 );

    // copy dst.length() elements from device vector --> host
    // starting from srcOffset
    // srcOffset must be >= 0
    // length() - srcOffset must be >= dst.length()
    // dst must be packed()
    // return false on failure
    bool copyToHost( Array1DWriteView< T > dst, int srcOffset = 0 ) const;

    const T* pointer() const;
    T* pointer();

    const T* elementPointer( int x ) const;
    T* elementPointer( int x );

    KernelArray1D< T > kernelArray1D();

private:

    size_t m_length = 0;
    uint8_t* m_devicePointer = nullptr;

    void destroy();
};

#include "DeviceArray1D.inl"
