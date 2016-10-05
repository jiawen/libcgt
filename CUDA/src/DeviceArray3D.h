#pragma once

#include <cuda_runtime.h>
//#include <vector_types.h>
#include <helper_cuda.h>

#include <common/Array3D.h>
#include <common/Array3DView.h>
#include <vecmath/Vector3i.h>

#include "KernelArray3D.h"

// Basic 3D array interface around CUDA global memory.
// Wraps around cudaMalloc3D() (linear allocation with pitch).
template< typename T >
class DeviceArray3D
{
public:

    DeviceArray3D() = default;
    DeviceArray3D( const Vector3i& size );
    DeviceArray3D( const DeviceArray3D< T >& copy );
    DeviceArray3D( DeviceArray3D< T >&& move );
    DeviceArray3D< T >& operator = ( const DeviceArray3D< T >& copy );
    DeviceArray3D< T >& operator = ( DeviceArray3D< T >&& move );
    virtual ~DeviceArray3D();

    bool isNull() const;
    bool notNull() const;

    int width() const;
    int height() const;
    int depth() const;
    Vector3i size() const;
    int numElements() const;

    // The number of bytes between rows within any slice.
    size_t rowPitch() const;

    // The number of bytes between slices
    size_t slicePitch() const;

    // Total size of the data in bytes (counting alignment)
    size_t sizeInBytes() const;

    // resizes the array, original data is not preserved
    void resize( int width, int height, int depth );
    void resize( const Vector3i& size );

    // TODO: implement get/set/operator() with Vector3i, vector3
    // TODO: implement constructors for strided, pitched, slicePitched

    // fills this array with the 0 byte pattern
    void clear();

    // fills this array with value
    void fill( const T& value );

    // Get an element of the array from the device.
    // WARNING: probably slow as it incurs a call to cudaMemcpy().
    T get( const Vector3i& xyz ) const;

    // Get an element of the array from the device.
    // WARNING: probably slow as it incurs a call to cudaMemcpy().
    T operator [] ( const Vector3i& xyz ) const;

    // Set an element of the array with data from the host.
    // WARNING: probably slow as it incurs a call to cudaMemcpy().
    void set( const Vector3i& xyz, const T& value );

    // TODO(jiawen): move these into utility functions.
    // Copy from another DeviceArray3D to this
    bool copyFromDevice( const DeviceArray3D< T >& src );

    // copy from host array src to this
    bool copyFromHost( Array3DView< const T > src );

    // copy from this to host array dst
    bool copyToHost( Array3DView< T > dst ) const;

    const cudaPitchedPtr pitchedPointer() const;
    cudaPitchedPtr pitchedPointer();

    KernelArray3D< const T > readView() const;
    KernelArray3D< T > writeView() const;

    void load( const char* filename );

private:

    Vector3i m_size;

    // We allocate memory by first making a cudaExtent, which expects:
    // width in *bytes*, height and depth in elements.
    cudaExtent m_extent = {};

    // We then pass m_extent to cudaMalloc3D() to allocate a 3D block of
    // memory. It returns a cudaPitchedPtr, which only knows about 2D pitch.
    //
    // The returned pitch is the width of a row in bytes.
    // The returned xsize and ysize echo what was passed in:
    //   xsize is the logical width in bytes (not elements).
    //   ysize is the logical height.
    //   depth is not echoed.
    // There is never any space between slices.
    cudaPitchedPtr m_pitchedPointer = {};

    // Frees the memory if this is not null.
    void destroy();
};

#include "DeviceArray3D.inl"
