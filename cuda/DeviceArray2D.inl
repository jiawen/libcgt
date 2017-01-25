template< typename T >
DeviceArray2D< T >::DeviceArray2D( const Vector2i& size )
{
    resize( size );
}

template< typename T >
DeviceArray2D< T >::DeviceArray2D( const DeviceArray2D< T >& copy )
{
    resize( copy.size() );
    ::copy( copy, *this );
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
        resize( copy.size() );
        ::copy( copy, *this );
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
cudaError DeviceArray2D< T >::resize( const Vector2i& size )
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
            size_t pitch;
            err = cudaMallocPitch(
                reinterpret_cast< void** >( &m_devicePointer ),
                &pitch,
                size.x * sizeof( T ),
                size.y );
            if( err == cudaSuccess )
            {
                m_size = size;
                m_stride = { sizeof( T ), static_cast< int >( pitch ) };
            }
        }
    }
    return err;
}

template< typename T >
cudaError DeviceArray2D< T >::clear()
{
    return cudaMemset2D( pointer(), rowStrideBytes(), 0 /* value */,
        widthInBytes(), height() );
}

template< typename T >
cudaError DeviceArray2D< T >::fill( const T& value )
{
    // TODO(jiawen): get rid of Array2D and use thrust::fill.
    Array2D< T > h_array( width(), height(), value );
    return copy( h_array.readView(), *this );
}

template< typename T >
T DeviceArray2D< T >::get( const Vector2i& xy, cudaError& err ) const
{
    T output;
    const T* p = elementPointer( xy );
    err = cudaMemcpy( &output, p, sizeof( T ), cudaMemcpyDeviceToHost );
    return output;
}

template< typename T >
T DeviceArray2D< T >::operator [] ( const Vector2i& xy ) const
{
    cudaError err;
    return get( xy, err );
}

template< typename T >
cudaError DeviceArray2D< T >::set( const Vector2i& xy, const T& value )
{
    T* p = elementPointer( xy );
    return cudaMemcpy( p, &value, sizeof( T ), cudaMemcpyHostToDevice );
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
KernelArray2D< const T > DeviceArray2D< T >::readView() const
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
        copy( h_arr, *this );
    }
}

template< typename T >
cudaError DeviceArray2D< T >::destroy()
{
    cudaError err = cudaSuccess;
    if( notNull() )
    {
        err = cudaFree( m_devicePointer );
        m_devicePointer = nullptr;
    }

    m_size = Vector2i{ 0 };
    m_stride = Vector2i{ 0 };
    return err;
}

template< typename T >
size_t DeviceArray2D< T >::widthInBytes() const
{
    return m_size.x * sizeof( T );
}


template< typename T >
cudaError copy( const DeviceArray2D< T >& src, DeviceArray2D< T >& dst )
{
    if( src.isNull() || dst.isNull() || src.size() != dst.size() )
    {
        return cudaErrorInvalidValue;
    }

    return cudaMemcpy2D
    (
        dst.pointer(), dst.rowStrideBytes(),
        src.m_devicePointer, src.rowStrideBytes(),
        src.width() * sizeof( T ), src.height(),
        cudaMemcpyDeviceToDevice
    );
}

template< typename T >
cudaError copy( Array2DReadView< T > src, DeviceArray2D< T >& dst )
{
    if( !( src.elementsArePacked() ) || src.size() != dst.size() )
    {
        return cudaErrorInvalidValue;
    }

    return cudaMemcpy2D
    (
        dst.pointer(), dst.rowStrideBytes(),
        src.pointer(), src.rowStrideBytes(),
        src.width() * sizeof( T ), src.height(),
        cudaMemcpyHostToDevice
    );
}

template< typename T >
cudaError copy( const DeviceArray2D< T >& src, Array2DWriteView< T > dst )
{
    if( src.isNull() ||
        !( dst.elementsArePacked() ) ||
        src.size() != dst.size() )
    {
        return cudaErrorInvalidValue;
    }

    return cudaMemcpy2D
    (
        dst.pointer(), dst.width() * sizeof( T ),
        src.pointer(), src.rowStrideBytes(),
        src.width() * sizeof( T ), src.height(),
        cudaMemcpyDeviceToHost
    );
}

template< typename T >
cudaError copy( cudaArray_t src, DeviceArray2D< T >& dst )
{
    return copy( src, { 0, 0 }, dst );
}

template< typename T >
cudaError copy( cudaArray_t src, const Vector2i& srcXY,
    DeviceArray2D< T >& dst )
{
    return cudaMemcpy2DFromArray
    (
        dst.pointer(), dst.rowStrideBytes(),
        src,
        srcXY.x, srcXY.y,
        dst.width() * sizeof( T ), dst.height(),
        cudaMemcpyDeviceToDevice
    );
}

template< typename T >
cudaError copy( const DeviceArray2D< T >& src, cudaArray_t dst,
    const Vector2i& dstXY )
{
    return cudaMemcpy2DToArray
    (
        dst,
        dstXY.x, dstXY.y,
        src.pointer(), src.rowStrideBytes(),
        src.width() * sizeof( T ), src.height(),
        cudaMemcpyDeviceToDevice
    );
}
