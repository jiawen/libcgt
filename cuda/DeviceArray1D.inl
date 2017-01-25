template< typename T >
DeviceArray1D< T >::DeviceArray1D( size_t length )
{
    resize( length );
}

template< typename T >
DeviceArray1D< T >::DeviceArray1D( const DeviceArray1D< T >& copy )
{
    resize( copy.size() );
    ::copy( copy, *this );
}

template< typename T >
DeviceArray1D< T >::DeviceArray1D( DeviceArray1D< T >&& move )
{
    m_length = move.m_length;
    m_devicePointer = move.m_devicePointer;

    move.m_length = 0;
    move.m_devicePointer = nullptr;
}

template< typename T >
DeviceArray1D< T >& DeviceArray1D< T >::operator = (
    const DeviceArray1D< T >& copy )
{
    if( this != &copy )
    {
        resize( copy.size() );
        ::copy( copy, *this );
    }
    return *this;
}

template< typename T >
DeviceArray1D< T >& DeviceArray1D< T >::operator = (
    DeviceArray1D< T >&& move )
{
    if( this != &move )
    {
        destroy();

        m_length = move.m_length;
        m_devicePointer = move.m_devicePointer;

        move.m_length = 0;
        move.m_devicePointer = nullptr;
    }
    return *this;
}

template< typename T >
// virtual
DeviceArray1D< T >::~DeviceArray1D()
{
    destroy();
}

template< typename T >
bool DeviceArray1D< T >::isNull() const
{
    return( m_devicePointer == nullptr );
}

template< typename T >
bool DeviceArray1D< T >::notNull() const
{
    return( m_devicePointer != nullptr );
}

template< typename T >
size_t DeviceArray1D< T >::length() const
{
    return m_length;
}

template< typename T >
size_t DeviceArray1D< T >::size() const
{
    return m_length;
}

template< typename T >
size_t DeviceArray1D< T >::numElements() const
{
    return m_length;
}

template< typename T >
size_t DeviceArray1D< T >::sizeInBytes() const
{
    return m_length * sizeof( T );
}

template< typename T >
cudaResourceDesc DeviceArray1D< T >::resourceDesc() const
{
    cudaResourceDesc desc = {};
    desc.resType = cudaResourceTypeLinear;
    desc.res.linear.devPtr = m_devicePointer;
    desc.res.linear.sizeInBytes = sizeInBytes();
    desc.res.linear.desc = cudaCreateChannelDesc< T >();
    return desc;
}

template< typename T >
cudaError DeviceArray1D< T >::resize( size_t length )
{
    if( m_length == length )
    {
        return cudaSuccess;
    }

    cudaError err = destroy();
    if( err == cudaSuccess )
    {
        if( length > 0 )
        {
            err = cudaMalloc(
                reinterpret_cast< void** >( &m_devicePointer ),
                length * sizeof( T ) );
            if( err == cudaSuccess )
            {
                m_length = length;
            }
        }
    }
    return err;
}

template< typename T >
cudaError DeviceArray1D< T >::clear()
{
    return cudaMemset( m_devicePointer, 0, sizeInBytes() );
}

template< typename T >
void DeviceArray1D< T >::fill( const T& value )
{
    T* begin = elementPointer( 0 );
    T* end = elementPointer( length() );
    thrust::fill( thrust::device, begin, end, value );
}

template< typename T >
T DeviceArray1D< T >::get( int x, cudaError& err ) const
{
    T output;
    err = cudaMemcpy
    (
        &output, elementPointer( x ), sizeof( T ),
        cudaMemcpyDeviceToHost
    );
    return output;
}

template< typename T >
T DeviceArray1D< T >::operator [] ( int x ) const
{
    cudaError err;
    return get( x, err );
}

template< typename T >
cudaError DeviceArray1D< T >::set( int x, const T& value )
{
    return cudaMemcpy( elementPointer( x ), &value, sizeof( T ),
        cudaMemcpyHostToDevice );
}

template< typename T >
const T* DeviceArray1D< T >::pointer() const
{
    return reinterpret_cast< const T* >( m_devicePointer );
}

template< typename T >
T* DeviceArray1D< T >::pointer()
{
    return reinterpret_cast< T* >( m_devicePointer );
}

template< typename T >
const T* DeviceArray1D< T >::elementPointer( size_t x ) const
{
    return reinterpret_cast< const T* >( m_devicePointer + x * sizeof( T ) );
}

template< typename T >
T* DeviceArray1D< T >::elementPointer( size_t x )
{
    return reinterpret_cast< T* >( m_devicePointer + x * sizeof( T ) );
}

template< typename T >
KernelArray1D< const T > DeviceArray1D< T >::readView() const
{
    return KernelArray1D< const T >( pointer(), m_length );
}

template< typename T >
KernelArray1D< T > DeviceArray1D< T >::writeView()
{
    return KernelArray1D< T >( pointer(), m_length );
}

template< typename T >
cudaError DeviceArray1D< T >::destroy()
{
    cudaError err = cudaSuccess;
    if( notNull() )
    {
        err = cudaFree( m_devicePointer );
        m_devicePointer = nullptr;
    }

    m_length = 0;

    return err;
}

template< typename T >
cudaError copy( Array1DReadView< T > src, DeviceArray1D< T >& dst,
    size_t dstOffset )
{
    if( dst.isNull() ||
        !( src.packed() ) ||
        dst.length() - dstOffset < src.size() )
    {
        return cudaErrorInvalidValue;
    }

    const T* srcPointer = src.pointer();
    T* dstPointer = dst.elementPointer( dstOffset );
    size_t srcSizeBytes = src.size() * sizeof( T );

    return cudaMemcpy( dstPointer, srcPointer, srcSizeBytes,
        cudaMemcpyHostToDevice );
}

template< typename T >
cudaError copy( const DeviceArray1D< T >& src, Array1DWriteView< T > dst )
{
    return copy( src, 0, dst );
}

template< typename T >
cudaError copy( const DeviceArray1D< T >& src, size_t srcOffset,
    Array1DWriteView< T > dst )
{
    if( src.isNull() ||
        !( dst.packed() ) ||
        src.length() - srcOffset < dst.size() )
    {
        return cudaErrorInvalidValue;
    }

    const T* srcPointer = src.elementPointer( srcOffset );
    T* dstPointer = dst.pointer();
    size_t dstSizeBytes = dst.size() * sizeof( T );

    return cudaMemcpy( dstPointer, srcPointer, dstSizeBytes,
        cudaMemcpyDeviceToHost );
}

// Copy data from src to dst.
// src and dst must have the same size.
template< typename T >
cudaError copy( const DeviceArray1D< T >& src, DeviceArray1D< T >& dst )
{
    if( src.isNull() ||
        !( dst.elementsArePacked() ) ||
        src.size() != dst.size() )
    {
        return cudaErrorInvalidValue;
    }

    return cudaMemcpy( dst.pointer(), src.pointer(), src.sizeInBytes(),
        cudaMemcpyDeviceToDevice );
}
