#pragma once

#include <cassert>

#include <cuda_runtime.h>

#include <common/ArrayView.h>

// TODO: create an Interop version.
// TODO: support CUDA surfaces: bindSurface()
// TODO: use cudaMalloc3DArray() for an ND array.

// Wrapper around CUDA "array" memory.
// Allocates 2D memory using cudaMallocArray(), which can only be used as
// textures or surfaces.
//
// T should be a CUDA element type such uchar, schar2, or float4.
template< typename T >
class DeviceOpaqueArray2D
{
public:

    DeviceOpaqueArray2D() = default;
    DeviceOpaqueArray2D( const Vector2i& size );
    ~DeviceOpaqueArray2D();
    // TODO: move constructors and assignment operator.

    bool isNull() const;
    bool notNull() const;

    const cudaChannelFormatDesc& channelFormatDescription() const;
    const cudaResourceDesc& resourceDescription() const;

    int width() const;
    int height() const;
    Vector2i size() const;
    int numElements() const;

    // TODO: resize()?
    // TODO: bindTexture()

    // For the copy to succeed, sizes must be exact:
    //   src.size() == size() - dstOffset
    // In addition, src.elementsArePacked() or src.packed() must be true.
    bool copyFromHost( Array2DReadView< T > src,
        const Vector2i& dstOffset = Vector2i{ 0 } );

    // For the copy to succeed, sizes must be exact:
    //   size() - dstOffset == dst.size()
    // In addition, dst.elementsArePacked() or dst.packed() must be true.
    bool copyToHost( Array2DWriteView< T > dst,
        const Vector2i& srcOffset = Vector2i{ 0 } ) const;

    const cudaArray* deviceArray() const;
    cudaArray* deviceArray();

private:

    Vector2i m_size = Vector2i{ 0 };
    cudaChannelFormatDesc m_cfd = {};
    cudaResourceDesc m_resourceDesc = {};
    size_t m_sizeInBytes = 0;
    cudaArray* m_deviceArray = nullptr;

};

#include "DeviceOpaqueArray2D.inl"
