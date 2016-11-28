template< typename T >
DeviceArray2D< T >::DeviceArray2D( const Vector2i& size )
{
    resize( size );
}

template< typename T >
DeviceArray2D< T >::DeviceArray2D( const DeviceArray2D< T >& copy )
{
    copyFromDevice( copy );
}

template< typename T >
DeviceArray2D< T >::DeviceArray2D( DeviceArray2D< T >&& move )
{
    m_size = move.m_size;
    m_stride = move.m_stride;
    m_devicePointer = move.m_devicePointer;

    move.m_size = Vector2i{ 0 };
    move.m_stride = Vector2i{ 0 };
    move.m_devicePointer = nullptr;
}

template< typename T >
DeviceArray2D< T >& DeviceArray2D< T >::operator = (
    const DeviceArray2D< T >& copy )
{
    if( this != &copy )
    {
        copyFromDevice( copy );
    }
    return *this;
}

template< typename T >
DeviceArray2D< T >& DeviceArray2D< T >::operator = (
    DeviceArray2D< T >&& move )
{
    if( this != &move )
    {
        destroy();

        m_size = move.m_size;
        m_stride = move.m_stride;
        m_devicePointer = move.m_devicePointer;

        move.m_size = Vector2i{ 0 };
        move.m_stride = Vector2i{ 0 };
        move.m_devicePointer = nullptr;
    }
    return *this;
}

template< typename T >
DeviceArray2D< T >::~DeviceArray2D()
{
    destroy();
}

template< typename T >
bool DeviceArray2D< T >::isNull() const
{
    return( m_devicePointer == nullptr );
}

template< typename T >
bool DeviceArray2D< T >::notNull() const
{
    return( m_devicePointer != nullptr );
}

template< typename T >
int DeviceArray2D< T >::width() const
{
    return m_size.x;
}

template< typename T >
int DeviceArray2D< T >::height() const
{
    return m_size.y;
}

template< typename T >
Vector2i DeviceArray2D< T >::size() const
{
    return m_size;
}

template< typename T >
int DeviceArray2D< T >::numElements() const
{
    return m_size.x * m_size.y;
}

template< typename T >
size_t DeviceArray2D< T >::elementStrideBytes() const
{
    return m_stride.x;
}

template< typename T >
size_t DeviceArray2D< T >::rowStrideBytes() const
{
    return m_stride.y;
}

template< typename T >
Vector2i DeviceArray2D< T >::stride() const
{
    return m_stride;
}

template< typename T >
size_t DeviceArray2D< T >::sizeInBytes() const
{
    return m_stride.y * m_size.y;
}

template< typename T >
cudaResourceDesc DeviceArray2D< T >::resourceDesc() const
{
    cudaResourceDesc desc = {};
    desc.resType = cudaResourceTypePitch2D;
    desc.res.pitch2D.devPtr = m_devicePointer;
    desc.res.pitch2D.width = m_size.x;
    desc.res.pitch2D.height = m_size.y;
    desc.res.pitch2D.pitchInBytes = m_stride.y;
    desc.res.pitch2D.desc = cudaCreateChannelDesc< T >();
    return desc;
}

template< typename T >
void DeviceArray2D< T >::resize( const Vector2i& size )
{
    if( size == m_size )
    {
        return;
    }

    destroy();

    if( size.x > 0 && size.y > 0 )
    {
        m_size = size;
        size_t pitch;

        checkCudaErrors
        (
            cudaMallocPitch
            (
                reinterpret_cast< void** >( &m_devicePointer ),
                &pitch,
                m_size.x * sizeof( T ),
                m_size.y
            )
        );

        m_stride = { sizeof( T ), static_cast< int >( pitch ) };
    }
}

template< typename T >
void DeviceArray2D< T >::clear()
{
    checkCudaErrors
    (
        cudaMemset2D( pointer(), rowStrideBytes(), 0, widthInBytes(), height() )
    );
}

template< typename T >
void DeviceArray2D< T >::fill( const T& value )
{
    // TODO(jiawen): get rid of Array2D and use thrust::fill.
    Array2D< T > h_array( width(), height(), value );
    copyFromHost( h_array );
}

template< typename T >
T DeviceArray2D< T >::get( const Vector2i& xy ) const
{
    T output;
    const T* p = elementPointer( xy );
    checkCudaErrors
    (
        cudaMemcpy( &output, p, sizeof( T ), cudaMemcpyDeviceToHost )
    );
    return output;
}

template< typename T >
T DeviceArray2D< T >::operator [] ( const Vector2i& xy ) const
{
    return get( xy );
}

template< typename T >
void DeviceArray2D< T >::set( const Vector2i& xy, const T& value )
{
    T* p = elementPointer( xy );
    checkCudaErrors
    (
        cudaMemcpy( p, &value, sizeof( T ), cudaMemcpyHostToDevice )
    );
}

template< typename T >
void DeviceArray2D< T >::copyFromDevice( const DeviceArray2D< T >& src )
{
    if( isNull() || src.isNull() )
    {
        return;
    }

    resize( src.size() );
    checkCudaErrors
    (
        cudaMemcpy2D
        (
            m_devicePointer,
            rowStrideBytes(), src.m_devicePointer, src.m_stride.y,
            src.widthInBytes(), src.height(),
            cudaMemcpyDeviceToDevice
        )
    );
}

template< typename T >
bool DeviceArray2D< T >::copyFromHost( Array2DReadView< T > src )
{
    if( src.isNull() )
    {
        return false;
    }
    if( !( src.elementsArePacked() ) )
    {
        return false;
    }

    // TODO: check if resize ok, then check if memcpy is ok.
    resize( src.size() );
    checkCudaErrors
    (
        cudaMemcpy2D
        (
            pointer(), rowStrideBytes(),
            src.pointer(), src.rowStrideBytes(),
            src.width() * sizeof( T ), src.height(),
            cudaMemcpyHostToDevice
        )
    );

    return true;
}

template< typename T >
bool DeviceArray2D< T >::copyToHost( Array2DWriteView< T > dst ) const
{
    if( isNull() || dst.isNull() || m_size != dst.size() )
    {
        return false;
    }

    checkCudaErrors
    (
        cudaMemcpy2D
        (
            dst.pointer(), dst.width() * sizeof( T ),
            pointer(), m_stride.y,
            widthInBytes(), height(),
            cudaMemcpyDeviceToHost
        )
    );

    return true;
}

template< typename T >
void DeviceArray2D< T >::copyFromArray( cudaArray* src )
{
    checkCudaErrors
    (
        cudaMemcpy2DFromArray
        (
            pointer(), rowStrideBytes(),
            src,
            0, 0,
            widthInBytes(), height(),
            cudaMemcpyDeviceToDevice
        )
    );
}

template< typename T >
void DeviceArray2D< T >::copyToArray( cudaArray* dst ) const
{
    checkCudaErrors
    (
        cudaMemcpy2DToArray
        (
            dst,
            0, 0,
            pointer(), rowStrideBytes(),
            widthInBytes(), height(),
            cudaMemcpyDeviceToDevice
        )
    );
}

template< typename T >
const T* DeviceArray2D< T >::pointer() const
{
    return reinterpret_cast< const T* >( m_devicePointer );
}

template< typename T >
T* DeviceArray2D< T >::pointer()
{
    return reinterpret_cast< T* >( m_devicePointer );
}

template< typename T >
const T* DeviceArray2D< T >::elementPointer( const Vector2i& xy ) const
{
    return reinterpret_cast< const T* >(
        m_devicePointer + Vector2i::dot( xy, m_stride ) );
}

template< typename T >
T* DeviceArray2D< T >::elementPointer( const Vector2i& xy )
{
    return reinterpret_cast< T* >(
        m_devicePointer + Vector2i::dot( xy, m_stride ) );
}

template< typename T >
const T* DeviceArray2D< T >::rowPointer( size_t y ) const
{
    return elementPointer( { 0, static_cast< int >( y ) } );
}

template< typename T >
T* DeviceArray2D< T >::rowPointer( size_t y )
{
    return elementPointer( { 0, static_cast< int >( y ) } );
}

template< typename T >
KernelArray2D< const T > DeviceArray2D< T >::readView()
{
    return KernelArray2D< const T >(
        pointer(), { m_size.x, m_size.y }, m_stride.y );
}

template< typename T >
KernelArray2D< T > DeviceArray2D< T >::writeView()
{
    return KernelArray2D< T >( pointer(), { m_size.x, m_size.y }, m_stride.y );
}

template< typename T >
void DeviceArray2D< T >::load( const char* filename )
{
    Array2D< T > h_arr( filename );
    if( !( h_arr.isNull() ) )
    {
        resize( h_arr.size() );
        copyFromHost( h_arr );
    }
}

template< typename T >
void DeviceArray2D< T >::destroy()
{
    if( notNull() )
    {
        checkCudaErrors( cudaFree( m_devicePointer ) );
        m_devicePointer = nullptr;
    }

    m_size = Vector2i{ 0 };
    m_stride = Vector2i{ 0 };
}

template< typename T >
size_t DeviceArray2D< T >::widthInBytes() const
{
    return m_size.x * sizeof( T );
}
