#pragma once

#include <cuda_runtime.h>

#include "libcgt/core/common/Array2D.h"
#include "libcgt/core/common/ArrayView.h"
#include "libcgt/cuda/KernelArray2D.h"

// Basic 2D array interface around CUDA global memory.
// Wraps around cudaMallocPitch() (linear allocation with pitch).
template< typename T >
class DeviceArray2D
{
public:

    DeviceArray2D() = default;
    DeviceArray2D( const Vector2i& size );
    DeviceArray2D( const DeviceArray2D< T >& copy );
    DeviceArray2D( DeviceArray2D< T >&& move );
    DeviceArray2D< T >& operator = ( const DeviceArray2D< T >& copy );
    DeviceArray2D< T >& operator = ( DeviceArray2D< T >&& move );
    ~DeviceArray2D();

    bool isNull() const;
    bool notNull() const;

    int width() const;
    int height() const;
    Vector2i size() const; // (width, height)
    int numElements() const;

    // The space between the start of elements in bytes.
    size_t elementStrideBytes() const;

    // The space between the start of rows in bytes.
    size_t rowStrideBytes() const;

    // { elementStride, rowStride } in bytes.
    Vector2i stride() const;

    // Total size of the data in bytes (counting alignment)
    size_t sizeInBytes() const;

    // This only works if T is a CUDA builtin type such as char, int2, or
    // float4 (i.e.,
    // TODO: consider using std::enable_if or thrust::enable_if.
    //
    // TODO: make this part of DeviceArray2DView once KernelArray2D is renamed.
    cudaResourceDesc resourceDesc() const;

    // Resize this array, destroying its contents.
    // size.x and size.y must be positive.
    cudaError resize( const Vector2i& size );

    // Sets this array to 0 (all bytes to 0).
    cudaError clear();

    // Fills this array with the given value.
    cudaError fill( const T& value );

    // Get an element of the array from the device, returning the value and
    // its status in err.
    // WARNING: probably slow as it incurs a cudaMemcpy.
    T get( const Vector2i& xy, cudaError& err ) const;

    // Get an element of the array from the device, without a status code.
    // WARNING: this operation is probably slow as it incurs a cudaMemcpy.
    T operator [] ( const Vector2i& xy ) const;

    // Sets an element of the array from the host.
    // WARNING: this operation is probably slow as it incurs a cudaMemcpy.
    cudaError set( const Vector2i& xy, const T& value );

    const T* pointer() const;
    T* pointer();

    const T* elementPointer( const Vector2i& xy ) const;
    T* elementPointer( const Vector2i& xy );

    const T* rowPointer( size_t y ) const;
    T* rowPointer( size_t y );

    KernelArray2D< const T > readView() const;
    KernelArray2D< T > writeView();

    void load( const char* filename );

private:

    Vector2i m_size = Vector2i{ 0 };
    // TODO(jiawen): stride should be a Vector2<uint64_t>.
    Vector2i m_stride = Vector2i{ 0 };

    uint8_t* m_devicePointer = nullptr;

    // Frees the device memory held by this array, sets its size and stride
    // to 0, and its pointer to nullptr.
    cudaError destroy();

    // Size of one row in bytes (not counting alignment)
    // Used for cudaMemset, which requires both row stride and the original
    // width.
    size_t widthInBytes() const;
};

// TODO: move these into ArrayUtils along with the rest.
// TODO: consider using Array2DReadView and Array2dWriteView and
// cudaMemcpyDefault, which can infer the direction.

// Copy data from src to dst.
// src's elements must be packed, but its rows do not have to be.
// src and dst must have the same size.
template< typename T >
cudaError copy( Array2DReadView< T > src, DeviceArray2D< T >& dst );

// Copy data from src to dst.
// dst's elements must be packed, but its rows do not have to be.
// src and dst must have the same size.
template< typename T >
cudaError copy( const DeviceArray2D< T >& src, Array2DWriteView< T > dst );

// Copy data from src to dst.
// src and dst must have the same size.
template< typename T >
cudaError copy( const DeviceArray2D< T >& src, DeviceArray2D< T >& dst );

// Copy data from src to dst.
// T must be a type supported by CUDA arrays (textures).
template< typename T >
cudaError copy( cudaArray_t src, DeviceArray2D< T >& dst );

// Copy data from src, starting at position srcXY, to dst.
// x points right, y points down.
// T must be a type supported by CUDA arrays (textures).
template< typename T >
cudaError copy( cudaArray_t src, const Vector2i& srcXY,
    DeviceArray2D< T >& dst );

// Copy data from src to dst.
// T must be a type supported by CUDA arrays (textures).
// TODO: src should use be an Array2DReadView to allow cropping.
// TODO: dst should be a DeviceOpaqueArray2D to allow size validation.
template< typename T >
cudaError copy( const DeviceArray2D< T >& src, cudaArray_t dst,
    const Vector2i& dstXY = Vector2i{ 0 } );

#include "libcgt/cuda/DeviceArray2D.inl"
