#pragma once

// CUDA
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

// libcgt
#include <common/Array1DView.h>

#include "KernelArray1D.h"
#include "ErrorChecking.h"

// Basic vector interface around CUDA global memory.
// Wraps around cudaMalloc() (linear allocation).
template< typename T >
class DeviceArray1D
{
public:

    DeviceArray1D() = default;
    DeviceArray1D( int length );
    DeviceArray1D( const DeviceArray1D< T >& copy );
    DeviceArray1D( DeviceArray1D< T >&& move );
    DeviceArray1D< T >& operator = ( const DeviceArray1D< T >& copy );
    DeviceArray1D< T >& operator = ( DeviceArray1D< T >&& move );
    virtual ~DeviceArray1D();

    bool isNull() const;
    bool notNull() const;

    int length() const;
    size_t sizeInBytes() const;

    // resizes the vector
    // original data is not preserved
    void resize( int length );

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
    bool copyFromHost( Array1DView< const T > src, int dstOffset = 0 );

    // copy dst.length() elements from device vector --> host
    // starting from srcOffset
    // srcOffset must be >= 0
    // length() - srcOffset must be >= dst.length()
    // dst must be packed()
    // return false on failure
    bool copyToHost( Array1DView< T > dst, int srcOffset = 0 ) const;

    const T* pointer() const;
    T* pointer();

    const T* elementPointer( int x ) const;
    T* elementPointer( int x );

    KernelArray1D< T > kernelArray1D();

private:

    int m_length = 0;
    uint8_t* m_devicePointer = nullptr;

    void destroy();
};

#include "DeviceArray1D.inl"
