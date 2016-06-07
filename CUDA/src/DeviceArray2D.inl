template< typename T >
DeviceArray2D< T >::DeviceArray2D( const Vector2i& size )
{
    resize( size );
}

template< typename T >
DeviceArray2D< T >::DeviceArray2D( Array2DView< const T > src )
{
    copyFromHost( src );
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
    m_pitch = move.m_pitch;
    m_sizeInBytes = move.m_sizeInBytes;
    m_devicePointer = move.m_devicePointer;

    move.m_size = Vector2i{ 0 };
    move.m_pitch = 0;
    move.m_sizeInBytes = 0;
    move.m_devicePointer = nullptr;
}

template< typename T >
DeviceArray2D< T >& DeviceArray2D< T >::operator = ( Array2DView< const T > src )
{
    copyFromHost( src );
    return *this;
}

template< typename T >
DeviceArray2D< T >& DeviceArray2D< T >::operator = ( const DeviceArray2D< T >& copy )
{
    if( this != &copy )
    {
        copyFromDevice( copy );
    }
    return *this;
}

template< typename T >
DeviceArray2D< T >& DeviceArray2D< T >::operator = ( DeviceArray2D< T >&& move )
{
    if( this != &move )
    {
        destroy();

        m_size = move.m_size;
        m_pitch = move.m_pitch;
        m_sizeInBytes = move.m_sizeInBytes;
        m_devicePointer = move.m_devicePointer;

        move.m_size = Vector2i{ 0 };
        move.m_pitch = 0;
        move.m_sizeInBytes = 0;
        move.m_devicePointer = nullptr;
    }
    return *this;
}

template< typename T >
// virtual
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
size_t DeviceArray2D< T >::pitch() const
{
    return m_pitch;
}

template< typename T >
size_t DeviceArray2D< T >::sizeInBytes() const
{
    return m_sizeInBytes;
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

        checkCudaErrors
        (
            cudaMallocPitch
            (
                reinterpret_cast< void** >( &m_devicePointer ),
                &m_pitch,
                m_size.x * sizeof( T ),
                m_size.y
            )
        );

        m_sizeInBytes = m_pitch * m_size.y;
    }
}

template< typename T >
void DeviceArray2D< T >::clear()
{
    checkCudaErrors( cudaMemset2D( devicePointer(), pitch(), 0, widthInBytes(), height() ) );
}

template< typename T >
void DeviceArray2D< T >::fill( const T& value )
{
    Array2D< T > h_array( width(), height(), value );
    copyFromHost( h_array );
}

template< typename T >
T DeviceArray2D< T >::get( const Vector2i& subscript ) const
{
    T output;

    const uint8_t* sourcePointer = reinterpret_cast< const uint8_t* >( devicePointer() );
    const uint8_t* rowPointer = sourcePointer + subscript.y * pitch();
    const uint8_t* elementPointer = rowPointer + subscript.x * sizeof( T );

    checkCudaErrors( cudaMemcpy( &output, elementPointer, sizeof( T ), cudaMemcpyDeviceToHost ) );

    return output;
}

template< typename T >
T DeviceArray2D< T >::operator [] ( const Vector2i& subscript ) const
{
    return get( subscript );
}

template< typename T >
void DeviceArray2D< T >::set( const Vector2i& subscript, const T& value )
{
    uint8_t* destinationPointer = reinterpret_cast< uint8_t* >( devicePointer() );
    uint8_t* rowPointer = destinationPointer + subscript.y * pitch();
    uint8_t* elementPointer = rowPointer + subscript.x * sizeof( T );

    checkCudaErrors( cudaMemcpy( elementPointer, &value, sizeof( T ), cudaMemcpyHostToDevice ) );
}

template< typename T >
void DeviceArray2D< T >::copyFromDevice( const DeviceArray2D< T >& src )
{
    if( isNull() || src.isNull() )
    {
        return;
    }

    resize( src.size() );
    checkCudaErrors( cudaMemcpy2D( m_devicePointer, m_pitch, src.m_devicePointer, src.m_pitch,
        src.widthInBytes(), src.height(), cudaMemcpyDeviceToDevice ) );
}

template< typename T >
bool DeviceArray2D< T >::copyFromHost( Array2DView< const T > src )
{
    if( src.isNull() )
    {
        return false;
    }
    if( !( src.elementsArePacked() ) )
    {
        return false;
    }


    resize( src.size() );
    checkCudaErrors
    (
        cudaMemcpy2D
        (
            devicePointer(), pitch(),
            src, src.rowStrideBytes(),
            src.width() * sizeof( T ), src.height(),
            cudaMemcpyHostToDevice
        )
    );

    return true;
}

template< typename T >
void DeviceArray2D< T >::copyToHost( Array2D< T >& dst ) const
{
    if( isNull() )
    {
        return;
    }

    dst.resize( size() );
    checkCudaErrors
    (
        cudaMemcpy2D
        (
            dst, dst.width() * sizeof( T ),
            devicePointer(), pitch(),
            widthInBytes(), height(),
            cudaMemcpyDeviceToHost
        )
    );
}

template< typename T >
void DeviceArray2D< T >::copyFromArray( cudaArray* src )
{
    checkCudaErrors
    (
        cudaMemcpy2DFromArray
        (
            devicePointer(), pitch(),
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
            devicePointer(), pitch(),
            widthInBytes(), height(),
            cudaMemcpyDeviceToDevice
        )
    );
}

template< typename T >
const T* DeviceArray2D< T >::devicePointer() const
{
    return m_devicePointer;
}

template< typename T >
T* DeviceArray2D< T >::devicePointer()
{
    return m_devicePointer;
}

template< typename T >
KernelArray2D< T > DeviceArray2D< T >::kernelArray() const
{
    return KernelArray2D< T >( m_devicePointer, m_size.x, m_size.y, m_pitch );
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
void DeviceArray2D< T >::save( const char* filename ) const
{
    Array2D< T > h_arr( width(), height() );
    copyToHost( h_arr );
    h_arr.save( filename );
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
    m_pitch = 0;
    m_sizeInBytes = 0;
}

template< typename T >
size_t DeviceArray2D< T >::widthInBytes() const
{
    return m_size.x * sizeof( T );
}
