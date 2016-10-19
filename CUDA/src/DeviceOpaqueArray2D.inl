template< typename T >
DeviceOpaqueArray2D< T >::DeviceOpaqueArray2D( const Vector2i& size ) :
    m_size( size ),
    m_sizeInBytes( size.x * size.y * sizeof( T ) ),
    m_cfd( cudaCreateChannelDesc< T >() )
{
    assert( size.x > 0 );
    assert( size.y > 0 );

    cudaError_t err = cudaMallocArray( &m_deviceArray, &m_cfd,
        size.x, size.y );
    if( err == cudaSuccess )
    {
        m_resourceDesc.resType = cudaResourceTypeArray;
        m_resourceDesc.res.array.array = m_deviceArray;
    }
    else
    {
        m_size = Vector2i{ 0 };
        m_cfd = { 0 };
        m_sizeInBytes = 0;
        m_deviceArray = nullptr;
    }
}

template< typename T >
DeviceOpaqueArray2D< T >::~DeviceOpaqueArray2D()
{
    if( m_deviceArray != nullptr )
    {
        cudaFreeArray( m_deviceArray );
        m_deviceArray = nullptr;
    }
    m_size = Vector2i{ 0 };
    m_resourceDesc = {};
    m_cfd = {};
    m_sizeInBytes = 0;
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
bool DeviceOpaqueArray2D< T >::copyFromHost( Array2DView< const T > src,
    const Vector2i& dstOffset )
{
    // TODO: Vector< 2, size_t >
    if( src.size() != size() - dstOffset() )
    {
        return false;
    }

    if( src.packed() )
    {
        cudaError_t err = cudaMemcpyToArray( m_deviceArray,
            dstOffset.x, dstOffset.y,
            src, src.numElements(),
            cudaMemcpyHostToDevice
        );
        return( err == cudaSuccess );
    }
    else if( src.elementsArePacked() )
    {
        cudaError_t err = cudaMemcpy2DToArray( m_deviceArray,
            dstOffset.x, dstOffset.y,
            src, src.rowStrideBytes(),
            src.width(), src.height(),
            cudaMemcpyHostToDevice
        );
        return( err == cudaSuccess );
    }
    else
    {
        return false;
    }
}

template< typename T >
bool DeviceOpaqueArray2D< T >::copyToHost( Array2DView< T > dst,
    const Vector2i& srcOffset ) const
{
    // TODO: Vector< 2, size_t >
    if( dst.size() != size() - srcOffset )
    {
        return false;
    }

    if( dst.packed() )
    {
        cudaError_t err = cudaMemcpyFromArray( dst,
            m_deviceArray,
            srcOffset.x, srcOffset.y,
            dst.numElements(),
            cudaMemcpyDeviceToHost
        );
        return( err == cudaSuccess );
    }
    else if( dst.elementsArePacked() )
    {
        cudaError_t err = cudaMemcpy2DFromArray( dst,
            dst.rowStrideBytes(),
            m_deviceArray,
            srcOffset.x, srcOffset.y,
            dst.width(), dst.height(),
            cudaMemcpyDeviceToHost
        );
        return( err == cudaSuccess );
    }
    else
    {
        return false;
    }
}

template< typename T >
const cudaArray* DeviceOpaqueArray2D< T >::deviceArray() const
{
    return m_deviceArray;
}

template< typename T >
cudaArray* DeviceOpaqueArray2D< T >::deviceArray()
{
    return m_deviceArray;
}
