#pragma once

#include <cuda_runtime.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include "libcgt/core/common/Array3D.h"
#include "libcgt/core/vecmath/Vector3i.h"
#include "libcgt/cuda/KernelArray3D.h"

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
    ~DeviceArray3D();

    bool isNull() const;
    bool notNull() const;

    int width() const;
    int height() const;
    int depth() const;
    Vector3i size() const;
    int numElements() const;

    // The number of bytes between two elements of any row.
    // In CUDA, this is always sizeof( T ).
    size_t elementStrideBytes() const;

    // The number of bytes between rows within any slice.
    size_t rowStrideBytes() const;

    // The number of bytes between slices.
    // In CUDA, this is always height() * rowStride().
    size_t sliceStrideBytes() const;

    // Total size of the data in bytes (counting alignment)
    size_t sizeInBytes() const;

    // Resizes array. Original data is not preserved.
    cudaError resize( const Vector3i& size );

    // TODO: implement constructors for strided, pitched, slicePitched

    // Sets this array to 0 (all bytes to 0).
    cudaError clear();

    // Fills this array with the given value.
    void fill( const T& value );

    // Get an element of the array from the device, returning the value and
    // its status in err.
    // WARNING: probably slow as it incurs a cudaMemcpy.
    T get( const Vector3i& xyz, cudaError& err ) const;

    // Get an element of the array from the device, without a status code.
    // WARNING: this operation is probably slow as it incurs a cudaMemcpy.
    T operator [] ( const Vector3i& xyz ) const;

    // Sets an element of the array from the host.
    // WARNING: this operation is probably slow as it incurs a cudaMemcpy.
    cudaError set( const Vector3i& xyz, const T& value );

    // Get a pointer to the first element.
    const T* pointer() const;
    T* pointer();

    // Get a pointer a particular element.
    const T* elementPointer( const Vector3i& xyz ) const;
    T* elementPointer( const Vector3i& xyz );

    // Returns a pointer to the beginning of the y-th row of the z-th slice
    const T* rowPointer( int y, int z ) const;
    T* rowPointer( int y, int z );

    // Returns a pointer to the beginning of the z-th slice
    const T* slicePointer( int z ) const;
    T* slicePointer( int z );

    const cudaPitchedPtr pitchedPointer() const;
    cudaPitchedPtr pitchedPointer();

    // Create a cudaExtent describing the size of this array.
    // Since this is (pitched) linear memory, it describe width in bytes,
    // but height and depth in elements.
    cudaExtent extent() const;

    // Returns true if there is no space between adjacent rows,
    // i.e., if rowStrideBytes() == width() * elementStrideBytes().
    //
    // Note that null arrays are not packed.
    bool rowsArePacked() const;

    KernelArray3D< const T > readView() const;
    KernelArray3D< T > writeView() const;

    void load( const char* filename );

private:

    Vector3i m_size;

    // cudaMalloc3D() allocates a 3D block of memory and returns a
    // cudaPitchedPtr describing its layout. cudaPitchedPtr can only describe a
    // 2D layout: (width, height, rowStride). This implies that there is never
    // any space between elements, or between slices, *but there may be space
    // between rows*.
    //
    // The m_pitchedPointer.pitch is the width of a row in bytes.
    // The returned xsize and ysize echo what was passed in:
    //   xsize is the logical width in bytes (not elements).
    //   ysize is the logical height.
    //   depth is not echoed.
    cudaPitchedPtr m_pitchedPointer = {};

    // Frees the memory if this is not null.
    cudaError destroy();
};

// Copy data from src to dst.
// src's elements must be packed, but its rows do not have to be.
// src and dst must have the same size.
template< typename T >
cudaError copy( Array3DReadView< T > src, DeviceArray3D< T >& dst );

// Copy data from src to dst.
// dst's elements must be packed, but its rows do not have to be.
// src and dst must have the same size.
template< typename T >
cudaError copy( const DeviceArray3D< T >& src, Array3DWriteView< T > dst );

// Copy data from src to dst.
// src and dst must have the same size.
template< typename T >
cudaError copy( const DeviceArray3D< T >& src, DeviceArray3D< T >& dst );

#include "libcgt/cuda/DeviceArray3D.inl"
