#pragma once

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <common/Array2D.h>

#include "KernelArray2D.h"

// Basic 2D array interface around CUDA global memory.
// Wraps around cudaMallocPitch() (linear allocation with pitch).
template< typename T >
class DeviceArray2D
{
public:

    DeviceArray2D() = default;
    DeviceArray2D( const Vector2i& size );
    DeviceArray2D( Array2DView< const T > src );
    DeviceArray2D( const DeviceArray2D< T >& copy );
    DeviceArray2D( DeviceArray2D< T >&& move );
    DeviceArray2D< T >& operator = ( Array2DView< const T > src );
    DeviceArray2D< T >& operator = ( const DeviceArray2D< T >& copy );
    DeviceArray2D< T >& operator = ( DeviceArray2D< T >&& move );
    virtual ~DeviceArray2D();

    bool isNull() const;
    bool notNull() const;

    int width() const;
    int height() const;
    Vector2i size() const; // (width, height)
    int numElements() const;

    // The number of bytes between rows
    size_t pitch() const;

    // Total size of the data in bytes (counting alignment)
    size_t sizeInBytes() const;

    // resizes the vector
    // original data is not preserved
    void resize( const Vector2i& size );

    // sets the vector to 0 (all bytes to 0)
    void clear();

    // fills this array with value
    void fill( const T& value );

    // get an element of the array from the device
    // WARNING: probably slow as it incurs a cudaMemcpy
    T get( const Vector2i& subscript ) const;

    // get an element of the array from the device
    // WARNING: probably slow as it incurs a cudaMemcpy
    T operator [] ( const Vector2i& subscript ) const;

    // sets an element of the array from the host
    // WARNING: probably slow as it incurs a cudaMemcpy
    void set( const Vector2i& subscript, const T& value );

    // copy from another DeviceArray2D to this
    // this is automatically resized
    void copyFromDevice( const DeviceArray2D< T >& src );

    // Copy from host array src to this.
    // This is automatically resized.
    bool copyFromHost( Array2DView< const T > src );

    // copy from this to host array dst
    // dst is automatically resized
    void copyToHost( Array2D< T >& dst ) const;

    // copy from cudaArray src to this
    void copyFromArray( cudaArray* src );

    // copy from this to cudaArray dst
    void copyToArray( cudaArray* dst ) const;

    const T* devicePointer() const;
    T* devicePointer();

    KernelArray2D< T > kernelArray() const;

    void load( const char* filename );
    void save( const char* filename ) const;

private:

    Vector2i m_size;
    size_t m_pitch = 0;
    T* m_devicePointer = nullptr;
    size_t m_sizeInBytes = 0;

    // frees the memory if this is not null
    void destroy();

    // Size of one row in bytes (not counting alignment)
    // Used for cudaMemset, which requires both a pitch and the original width
    size_t widthInBytes() const;
};

#include "DeviceArray2D.inl"
