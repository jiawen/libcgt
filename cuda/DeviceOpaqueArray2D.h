#pragma once

#include <cassert>

#include <cuda_runtime.h>

#include "libcgt/core/common/ArrayView.h"
#include "libcgt/core/geometry/RectangleUtils.h"
#include "libcgt/core/vecmath/Rect2i.h"

// TODO: create an Interop version.
// TODO: support CUDA surfaces: bindSurface()
// TODO: use cudaMalloc3DArray() for an ND array.

// Wrapper around CUDA "array" memory.
// Allocates 2D memory using cudaMallocArray(), which can only be used as
// textures or surfaces.
//
// Since T must be a CUDA element type (signed or unsigned 8, 16, or 32 bit
// integers, or float), all instances are explicitly instantiated.
template< typename T >
class DeviceOpaqueArray2D
{
public:

    DeviceOpaqueArray2D() = default;
    DeviceOpaqueArray2D( const Vector2i& size );
    DeviceOpaqueArray2D( const DeviceOpaqueArray2D< T >& copy );
    DeviceOpaqueArray2D( DeviceOpaqueArray2D< T >&& move );
    DeviceOpaqueArray2D< T >& operator = (
        const DeviceOpaqueArray2D< T >& copy );
    DeviceOpaqueArray2D< T >& operator = ( DeviceOpaqueArray2D< T >&& move );
    ~DeviceOpaqueArray2D();

    bool isNull() const;
    bool notNull() const;

    const cudaChannelFormatDesc& channelFormatDescription() const;
    const cudaResourceDesc& resourceDescription() const;

    int width() const;
    int height() const;
    Vector2i size() const;
    int numElements() const;

    // TODO: get(), set(), operator [], implemented using cudaMemcpy.

    // TODO: bindTexture()

    const cudaArray_t deviceArray() const;
    cudaArray_t deviceArray();

private:

    cudaError destroy();
    cudaError resize( const Vector2i& size );

    Vector2i m_size = Vector2i{ 0 };
    cudaChannelFormatDesc m_cfd = {};
    cudaResourceDesc m_resourceDesc = {};
    size_t m_sizeInBytes = 0;
    cudaArray_t m_deviceArray = nullptr;

};

// Copy data from src to dst, starting at dstOffset.
// Assumes:
// - src.elementsArePacked() (rows can have space between them).
// - src.size() == dst.size() - dstOffset.
template< typename T >
cudaError copy( Array2DReadView< T > src, DeviceOpaqueArray2D< T >& dst,
    const Vector2i& dstOffset = Vector2i{ 0 } );

// Copy data from src to dst.
// Assumes:
// - dst.elementsArePacked() (rows can have space between them).
// - src.size() == dst.size().
template< typename T >
cudaError copy( const DeviceOpaqueArray2D< T >& src,
    Array2DWriteView< T > dst );

// Copy a rectangular subset of data, starting at srcOffset, to dst.
// Assumes:
// - dst.elementsArePacked() (rows can have space between them).
// - src.size() - srcOffset == dst.size().
template< typename T >
cudaError copy( const DeviceOpaqueArray2D< T >& src, const Vector2i& srcOffset,
    Array2DWriteView< T > dst );

// Copy all of src to dst, starting at dstOffset.
// Assumes:
// - src.size() == dst.size() - dstOffset.
template< typename T >
cudaError copy( const DeviceOpaqueArray2D< T >& src,
    DeviceOpaqueArray2D< T >& dst, const Vector2i& dstOffset = Vector2i{ 0 } );

// Copy a rectangular subset of src, starting from srcOffset to the end, to
// dst, starting at dstOffset.
// Assumes:
// - src.size() - srcOffset == dst.size() - dstOffset.
template< typename T >
cudaError copy( const DeviceOpaqueArray2D< T >& src, const Vector2i& srcOffset,
    DeviceOpaqueArray2D< T >& dst, const Vector2i& dstOffset = Vector2i{ 0 } );

// Copy a rectangular subset of src to dst, starting at dstOffset.
// Assumes:
// - src.size() - srcOffset == dst.size() - dstOffset.
template< typename T >
cudaError copy( const DeviceOpaqueArray2D< T >& src, const Rect2i& srcRect,
    DeviceOpaqueArray2D< T >& dst, const Vector2i& dstOffset = Vector2i{ 0 } );
