#include "libcgt/cuda/DeviceOpaqueArray2D.h"

template< typename T >
DeviceOpaqueArray2D< T >::DeviceOpaqueArray2D( const Vector2i& size )
{
    resize( size );
}

template< typename T >
DeviceOpaqueArray2D< T >::DeviceOpaqueArray2D(
    const DeviceOpaqueArray2D< T >& copy )
{
    resize( copy.size() );
    ::copy( copy, *this );
}

template< typename T >
DeviceOpaqueArray2D< T >::DeviceOpaqueArray2D(
    DeviceOpaqueArray2D< T >&& move )
{
    m_size = move.m_size;
    m_cfd = move.m_cfd;
    m_resourceDesc = move.m_resourceDesc;
    m_sizeInBytes = move.m_sizeInBytes;
    m_deviceArray = move.m_deviceArray;

    move.m_size = Vector2i{ 0 };
    move.m_cfd = {};
    move.m_resourceDesc = {};
    move.m_sizeInBytes = 0;
    move.m_deviceArray = nullptr;
}

template< typename T >
DeviceOpaqueArray2D< T >& DeviceOpaqueArray2D< T >::operator =
( const DeviceOpaqueArray2D< T >& copy )
{
    if( this != &copy )
    {
        resize( copy.size() );
        ::copy( copy, *this );
    }
    return *this;
}

template< typename T >
DeviceOpaqueArray2D< T >& DeviceOpaqueArray2D< T >::operator =
( DeviceOpaqueArray2D< T >&& move )
{
    if( this != &move )
    {
        destroy();

        m_size = move.m_size;
        m_cfd = move.m_cfd;
        m_resourceDesc = move.m_resourceDesc;
        m_sizeInBytes = move.m_sizeInBytes;
        m_deviceArray = move.m_deviceArray;

        move.m_size = Vector2i{ 0 };
        move.m_cfd = {};
        move.m_resourceDesc = {};
        move.m_sizeInBytes = 0;
        move.m_deviceArray = nullptr;
    }
    return *this;
}

template< typename T >
DeviceOpaqueArray2D< T >::~DeviceOpaqueArray2D()
{
    destroy();
}

template< typename T >
bool DeviceOpaqueArray2D< T >::isNull() const
{
    return m_deviceArray == nullptr;
}

template< typename T >
bool DeviceOpaqueArray2D< T >::notNull() const
{
    return !isNull();
}

template< typename T >
const cudaChannelFormatDesc&
DeviceOpaqueArray2D< T >::channelFormatDescription() const
{
    return m_cfd;
}

template< typename T >
const cudaResourceDesc& DeviceOpaqueArray2D< T >::resourceDescription() const
{
    return m_resourceDesc;
}

template< typename T >
int DeviceOpaqueArray2D< T >::width() const
{
    return m_size.x;
}

template< typename T >
int DeviceOpaqueArray2D< T >::height() const
{
    return m_size.y;
}

template< typename T >
Vector2i DeviceOpaqueArray2D< T >::size() const
{
    return m_size;
}

template< typename T >
int DeviceOpaqueArray2D< T >::numElements() const
{
    return m_size.x * m_size.y;
}

template< typename T >
const cudaArray_t DeviceOpaqueArray2D< T >::deviceArray() const
{
    return m_deviceArray;
}

template< typename T >
cudaArray_t DeviceOpaqueArray2D< T >::deviceArray()
{
    return m_deviceArray;
}

template< typename T >
cudaError DeviceOpaqueArray2D< T >::destroy()
{
    cudaError err = cudaSuccess;
    if( notNull() )
    {
        err = cudaFreeArray( m_deviceArray );
        m_deviceArray = nullptr;
    }
    m_size = Vector2i{ 0 };
    m_resourceDesc = {};
    m_cfd = {};
    m_sizeInBytes = 0;

    return err;
}

template< typename T >
cudaError DeviceOpaqueArray2D< T >::resize( const Vector2i& size )
{
    if( size == m_size )
    {
        return cudaSuccess;
    }
    // Explicitly allow resize( 0, 0 ) for invoking constructor from a null
    // right hand side.
    if( size.x < 0 || size.y < 0 )
    {
        return cudaErrorInvalidValue;
    }

    cudaError err = destroy();
    if( err == cudaSuccess )
    {
        if( size.x > 0 && size.y > 0 )
        {
            cudaChannelFormatDesc cfd = cudaCreateChannelDesc< T >();
            err = cudaMallocArray( &m_deviceArray, &cfd,
                size.x * sizeof( T ), size.y );
            if( err == cudaSuccess )
            {
                m_size = size;
                m_sizeInBytes = size.x * size.y * sizeof( T );
                m_cfd = cfd;
                m_resourceDesc.resType = cudaResourceTypeArray;
                m_resourceDesc.res.array.array = m_deviceArray;
            }
        }
    }
    return err;
}

template< typename T >
cudaError copy( Array2DReadView< T > src, DeviceOpaqueArray2D< T >& dst,
    const Vector2i& dstOffset )
{
    // TODO: Vector< 2, size_t >
    if( src.size() != dst.size() - dstOffset )
    {
        return cudaErrorInvalidValue;
    }

    if( src.packed() )
    {
        return cudaMemcpyToArray( dst.deviceArray(),
            dstOffset.x, dstOffset.y,
            src.pointer(), src.numElements() * sizeof( T ),
            cudaMemcpyHostToDevice );
    }
    else if( src.elementsArePacked() )
    {
        return cudaMemcpy2DToArray( dst.deviceArray(),
            dstOffset.x, dstOffset.y,
            src.pointer(), src.rowStrideBytes(),
            src.width() * sizeof( T ), src.height(),
            cudaMemcpyHostToDevice );
    }
    else
    {
        return cudaErrorInvalidValue;
    }
}

template< typename T >
cudaError copy( const DeviceOpaqueArray2D< T >& src,
    Array2DWriteView< T > dst )
{
    return copy( src, { 0, 0 }, dst );
}

template< typename T >
cudaError copy( const DeviceOpaqueArray2D< T >& src, const Vector2i& srcOffset,
    Array2DWriteView< T > dst )
{
    // TODO: Vector< 2, size_t >
    if( src.size() - srcOffset != dst.size() )
    {
        return cudaErrorInvalidValue;
    }

    if( dst.packed() )
    {
        return cudaMemcpyFromArray( dst.pointer(),
            src.deviceArray(),
            srcOffset.x, srcOffset.y,
            dst.numElements() * sizeof( T ),
            cudaMemcpyDeviceToHost
        );
    }
    else if( dst.elementsArePacked() )
    {
        return cudaMemcpy2DFromArray( dst.pointer(),
            dst.rowStrideBytes(),
            src.deviceArray(),
            srcOffset.x, srcOffset.y,
            dst.width() * sizeof( T ), dst.height(),
            cudaMemcpyDeviceToHost
        );
    }
    else
    {
        return cudaErrorInvalidValue;
    }
}

template< typename T >
cudaError copy( const DeviceOpaqueArray2D< T >& src,
    DeviceOpaqueArray2D< T >& dst, const Vector2i& dstOffset )
{
    return copy( src, Rect2i{ src.size() }, dst, dstOffset );
}

template< typename T >
cudaError copy( const DeviceOpaqueArray2D< T >& src, const Vector2i& srcOffset,
    DeviceOpaqueArray2D< T >& dst, const Vector2i& dstOffset )
{
    Rect2i srcRect{ srcOffset, src.size() - srcOffset };
    return copy( src, srcRect, dst, dstOffset );
}

template< typename T >
cudaError copy( const DeviceOpaqueArray2D< T >& src, const Rect2i& srcRect,
    DeviceOpaqueArray2D< T >& dst, const Vector2i& dstOffset )
{
    if( !srcRect.isStandard() ||
        !libcgt::core::geometry::contains( Rect2i{ src.size() }, srcRect ) )
    {
        return cudaErrorInvalidValue;
    }
    if( dst.size().x - dstOffset.x < srcRect.size.x ||
        dst.size().y - dstOffset.y < srcRect.size.y )
    {
        return cudaErrorInvalidValue;
    }

    return cudaMemcpyArrayToArray( dst.deviceArray(),
        dstOffset.x, dstOffset.y,
        src.deviceArray(),
        srcRect.origin.x, srcRect.origin.y,
        srcRect.size.x * srcRect.size.y * sizeof( T ),
        cudaMemcpyDeviceToDevice );

    // TODO: figure out why the following doesn't work: random rows get lost.
#if 0
    return cudaMemcpy2DArrayToArray( dst.deviceArray(),
        dstOffset.x, dstOffset.y,
        src.deviceArray(),
        srcRect.origin.x, srcRect.origin.y,
        srcRect.size.x * sizeof( T ), srcRect.size.y,
        cudaMemcpyDeviceToDevice );
#endif
}

template class DeviceOpaqueArray2D< char >;
template class DeviceOpaqueArray2D< char2 >;
template class DeviceOpaqueArray2D< char3 >;
template class DeviceOpaqueArray2D< char4 >;
template class DeviceOpaqueArray2D< unsigned char >;
template class DeviceOpaqueArray2D< uchar2 >;
template class DeviceOpaqueArray2D< uchar3 >;
template class DeviceOpaqueArray2D< uchar4 >;
template class DeviceOpaqueArray2D< short >;
template class DeviceOpaqueArray2D< short2 >;
template class DeviceOpaqueArray2D< short3 >;
template class DeviceOpaqueArray2D< short4 >;
template class DeviceOpaqueArray2D< ushort1 >;
template class DeviceOpaqueArray2D< ushort2 >;
template class DeviceOpaqueArray2D< ushort3 >;
template class DeviceOpaqueArray2D< ushort4 >;
template class DeviceOpaqueArray2D< int >;
template class DeviceOpaqueArray2D< int2 >;
template class DeviceOpaqueArray2D< int3 >;
template class DeviceOpaqueArray2D< int4 >;
template class DeviceOpaqueArray2D< unsigned int >;
template class DeviceOpaqueArray2D< uint2 >;
template class DeviceOpaqueArray2D< uint3 >;
template class DeviceOpaqueArray2D< uint4 >;
template class DeviceOpaqueArray2D< float >;
template class DeviceOpaqueArray2D< float2 >;
template class DeviceOpaqueArray2D< float3 >;
template class DeviceOpaqueArray2D< float4 >;

// ----- Host to device -----

template
cudaError copy( Array2DReadView< char > src, DeviceOpaqueArray2D< char >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< char2 > src, DeviceOpaqueArray2D< char2 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< char3 > src, DeviceOpaqueArray2D< char3 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< char4 > src, DeviceOpaqueArray2D< char4 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< unsigned char > src, DeviceOpaqueArray2D< unsigned char >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< uchar2 > src, DeviceOpaqueArray2D< uchar2 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< uchar3 > src, DeviceOpaqueArray2D< uchar3 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< uchar4 > src, DeviceOpaqueArray2D< uchar4 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< short > src, DeviceOpaqueArray2D< short >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< short2 > src, DeviceOpaqueArray2D< short2 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< short3 > src, DeviceOpaqueArray2D< short3 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< short4 > src, DeviceOpaqueArray2D< short4 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< unsigned short > src, DeviceOpaqueArray2D< unsigned short >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< ushort2 > src, DeviceOpaqueArray2D< ushort2 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< ushort3 > src, DeviceOpaqueArray2D< ushort3 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< ushort4 > src, DeviceOpaqueArray2D< ushort4 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< int > src, DeviceOpaqueArray2D< int >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< int2 > src, DeviceOpaqueArray2D< int2 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< int3 > src, DeviceOpaqueArray2D< int3 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< int4 > src, DeviceOpaqueArray2D< int4 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< unsigned int > src, DeviceOpaqueArray2D< unsigned int >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< uint2 > src, DeviceOpaqueArray2D< uint2 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< uint3 > src, DeviceOpaqueArray2D< uint3 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< uint4 > src, DeviceOpaqueArray2D< uint4 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< float > src, DeviceOpaqueArray2D< float >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< float2 > src, DeviceOpaqueArray2D< float2 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< float3 > src, DeviceOpaqueArray2D< float3 >& dst,
    const Vector2i& dstOffset );

template
cudaError copy( Array2DReadView< float4 > src, DeviceOpaqueArray2D< float4 >& dst,
    const Vector2i& dstOffset );

// ----- Device to host -----

template
cudaError copy( const DeviceOpaqueArray2D< char >& src, Array2DWriteView< char > dst );

template
cudaError copy( const DeviceOpaqueArray2D< char2 >& src, Array2DWriteView< char2 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< char3 >& src, Array2DWriteView< char3 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< char4 >& src, Array2DWriteView< char4 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< unsigned char >& src, Array2DWriteView< unsigned char > dst );

template
cudaError copy( const DeviceOpaqueArray2D< uchar2 >& src, Array2DWriteView< uchar2 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< uchar3 >& src, Array2DWriteView< uchar3 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< uchar4 >& src, Array2DWriteView< uchar4 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< short >& src, Array2DWriteView< short > dst );

template
cudaError copy( const DeviceOpaqueArray2D< short2 >& src, Array2DWriteView< short2 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< short3 >& src, Array2DWriteView< short3 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< short4 >& src, Array2DWriteView< short4 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< unsigned short >& src, Array2DWriteView< unsigned short > dst );

template
cudaError copy( const DeviceOpaqueArray2D< ushort2 >& src, Array2DWriteView< ushort2 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< ushort3 >& src, Array2DWriteView< ushort3 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< ushort4 >& src, Array2DWriteView< ushort4 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< int >& src, Array2DWriteView< int > dst );

template
cudaError copy( const DeviceOpaqueArray2D< int2 >& src, Array2DWriteView< int2 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< int3 >& src, Array2DWriteView< int3 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< int4 >& src, Array2DWriteView< int4 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< unsigned int >& src, Array2DWriteView< unsigned int > dst );

template
cudaError copy( const DeviceOpaqueArray2D< uint2 >& src, Array2DWriteView< uint2 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< uint3 >& src, Array2DWriteView< uint3 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< uint4 >& src, Array2DWriteView< uint4 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< float >& src, Array2DWriteView< float > dst );

template
cudaError copy( const DeviceOpaqueArray2D< float2 >& src, Array2DWriteView< float2 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< float3 >& src, Array2DWriteView< float3 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< float4 >& src, Array2DWriteView< float4 > dst );

// ----- Device subset to host  -----

template
cudaError copy( const DeviceOpaqueArray2D< char >& src, const Vector2i& srcOffset, Array2DWriteView< char > dst );

template
cudaError copy( const DeviceOpaqueArray2D< char2 >& src, const Vector2i& srcOffset, Array2DWriteView< char2 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< char3 >& src, const Vector2i& srcOffset, Array2DWriteView< char3 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< char4 >& src, const Vector2i& srcOffset, Array2DWriteView< char4 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< unsigned char >& src, const Vector2i& srcOffset, Array2DWriteView< unsigned char > dst );

template
cudaError copy( const DeviceOpaqueArray2D< uchar2 >& src, const Vector2i& srcOffset, Array2DWriteView< uchar2 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< uchar3 >& src, const Vector2i& srcOffset, Array2DWriteView< uchar3 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< uchar4 >& src, const Vector2i& srcOffset, Array2DWriteView< uchar4 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< short >& src, const Vector2i& srcOffset, Array2DWriteView< short > dst );

template
cudaError copy( const DeviceOpaqueArray2D< short2 >& src, const Vector2i& srcOffset, Array2DWriteView< short2 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< short3 >& src, const Vector2i& srcOffset, Array2DWriteView< short3 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< short4 >& src, const Vector2i& srcOffset, Array2DWriteView< short4 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< unsigned short >& src, const Vector2i& srcOffset, Array2DWriteView< unsigned short > dst );

template
cudaError copy( const DeviceOpaqueArray2D< ushort2 >& src, const Vector2i& srcOffset, Array2DWriteView< ushort2 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< ushort3 >& src, const Vector2i& srcOffset, Array2DWriteView< ushort3 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< ushort4 >& src, const Vector2i& srcOffset, Array2DWriteView< ushort4 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< int >& src, const Vector2i& srcOffset, Array2DWriteView< int > dst );

template
cudaError copy( const DeviceOpaqueArray2D< int2 >& src, const Vector2i& srcOffset, Array2DWriteView< int2 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< int3 >& src, const Vector2i& srcOffset, Array2DWriteView< int3 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< int4 >& src, const Vector2i& srcOffset, Array2DWriteView< int4 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< unsigned int >& src, const Vector2i& srcOffset, Array2DWriteView< unsigned int > dst );

template
cudaError copy( const DeviceOpaqueArray2D< uint2 >& src, const Vector2i& srcOffset, Array2DWriteView< uint2 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< uint3 >& src, const Vector2i& srcOffset, Array2DWriteView< uint3 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< uint4 >& src, const Vector2i& srcOffset, Array2DWriteView< uint4 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< float >& src, const Vector2i& srcOffset, Array2DWriteView< float > dst );

template
cudaError copy( const DeviceOpaqueArray2D< float2 >& src, const Vector2i& srcOffset, Array2DWriteView< float2 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< float3 >& src, const Vector2i& srcOffset, Array2DWriteView< float3 > dst );

template
cudaError copy( const DeviceOpaqueArray2D< float4 >& src, const Vector2i& srcOffset, Array2DWriteView< float4 > dst );

// ----- Device to device, with destination offset -----

template
cudaError copy( const DeviceOpaqueArray2D< char >& src, DeviceOpaqueArray2D< char >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< char2 >& src, DeviceOpaqueArray2D< char2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< char3 >& src, DeviceOpaqueArray2D< char3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< char4 >& src, DeviceOpaqueArray2D< char4 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< unsigned char >& src, DeviceOpaqueArray2D< unsigned char >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< uchar2 >& src, DeviceOpaqueArray2D< uchar2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< uchar3 >& src, DeviceOpaqueArray2D< uchar3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< uchar4 >& src, DeviceOpaqueArray2D< uchar4 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< short >& src, DeviceOpaqueArray2D< short >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< short2 >& src, DeviceOpaqueArray2D< short2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< short3 >& src, DeviceOpaqueArray2D< short3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< short4 >& src, DeviceOpaqueArray2D< short4 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< unsigned short >& src, DeviceOpaqueArray2D< unsigned short >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< ushort2 >& src, DeviceOpaqueArray2D< ushort2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< ushort3 >& src, DeviceOpaqueArray2D< ushort3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< ushort4 >& src, DeviceOpaqueArray2D< ushort4 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< int >& src, DeviceOpaqueArray2D< int >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< int2 >& src, DeviceOpaqueArray2D< int2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< int3 >& src, DeviceOpaqueArray2D< int3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< int4 >& src, DeviceOpaqueArray2D< int4 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< unsigned int >& src, DeviceOpaqueArray2D< unsigned int >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< uint2 >& src, DeviceOpaqueArray2D< uint2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< uint3 >& src, DeviceOpaqueArray2D< uint3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< uint4 >& src, DeviceOpaqueArray2D< uint4 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< float >& src, DeviceOpaqueArray2D< float >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< float2 >& src, DeviceOpaqueArray2D< float2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< float3 >& src, DeviceOpaqueArray2D< float3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< float4 >& src, DeviceOpaqueArray2D< float4 >& dst, const Vector2i& dstOffset );

// ----- Device to device, with source and destination offset -----

template
cudaError copy( const DeviceOpaqueArray2D< char >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< char >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< char2 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< char2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< char3 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< char3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< char4 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< char4 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< unsigned char >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< unsigned char >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< uchar2 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< uchar2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< uchar3 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< uchar3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< uchar4 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< uchar4 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< short >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< short >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< short2 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< short2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< short3 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< short3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< short4 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< short4 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< unsigned short >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< unsigned short >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< ushort2 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< ushort2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< ushort3 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< ushort3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< ushort4 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< ushort4 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< int >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< int >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< int2 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< int2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< int3 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< int3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< int4 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< int4 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< unsigned int >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< unsigned int >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< uint2 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< uint2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< uint3 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< uint3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< uint4 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< uint4 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< float >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< float >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< float2 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< float2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< float3 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< float3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< float4 >& src, const Vector2i& srcOffset, DeviceOpaqueArray2D< float4 >& dst, const Vector2i& dstOffset );

// ----- Device to device, with source rect and destination offset -----

template
cudaError copy( const DeviceOpaqueArray2D< char >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< char >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< char2 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< char2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< char3 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< char3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< char4 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< char4 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< unsigned char >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< unsigned char >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< uchar2 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< uchar2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< uchar3 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< uchar3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< uchar4 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< uchar4 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< short >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< short >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< short2 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< short2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< short3 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< short3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< short4 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< short4 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< unsigned short >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< unsigned short >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< ushort2 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< ushort2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< ushort3 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< ushort3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< ushort4 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< ushort4 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< int >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< int >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< int2 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< int2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< int3 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< int3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< int4 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< int4 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< unsigned int >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< unsigned int >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< uint2 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< uint2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< uint3 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< uint3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< uint4 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< uint4 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< float >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< float >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< float2 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< float2 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< float3 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< float3 >& dst, const Vector2i& dstOffset );

template
cudaError copy( const DeviceOpaqueArray2D< float4 >& src, const Rect2i& srcRect, DeviceOpaqueArray2D< float4 >& dst, const Vector2i& dstOffset );
