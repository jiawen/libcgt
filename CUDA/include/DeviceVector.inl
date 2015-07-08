template< typename T >
DeviceVector< T >::DeviceVector() :

    m_sizeInBytes( 0 ),
    m_length( -1 ),
    m_devicePointer( nullptr )

{

}


template< typename T >
DeviceVector< T >::DeviceVector( int length ) :

    m_sizeInBytes( 0 ),
    m_length( -1 ),
    m_devicePointer( nullptr )

{
    resize( length );
}

template< typename T >
DeviceVector< T >::DeviceVector( const std::vector< T >& src ) :

    m_sizeInBytes( 0 ),
    m_length( -1 ),
    m_devicePointer( nullptr )

{
    copyFromHost( src );
}

template< typename T >
DeviceVector< T >::DeviceVector( const DeviceVector< T >& copy ) :

    m_sizeInBytes( 0 ),
    m_length( -1 ),
    m_devicePointer( nullptr )

{
    copyFromDevice( copy );
}

template< typename T >
DeviceVector< T >::DeviceVector( DeviceVector< T >&& move )
{
    m_sizeInBytes = move.m_sizeInBytes;
    m_length = move.m_length;
    m_devicePointer = move.m_devicePointer;

    move.m_sizeInBytes = 0;
    move.m_length = -1;
    move.m_devicePointer = nullptr;
}

template< typename T >
DeviceVector< T >& DeviceVector< T >::operator = ( const DeviceVector< T >& copy )
{
    if( this != &copy )
    {
        copyFromDevice( copy );
    }
    return *this;
}

template< typename T >
DeviceVector< T >& DeviceVector< T >::operator = ( DeviceVector< T >&& move )
{
    if( this != &move )
    {
        destroy();

        m_sizeInBytes = move.m_sizeInBytes;
        m_length = move.m_length;
        m_devicePointer = move.m_devicePointer;

        move.m_sizeInBytes = 0;
        move.m_length = -1;
        move.m_devicePointer = nullptr;
    }
    return *this;
}

template< typename T >
// virtual
DeviceVector< T >::~DeviceVector()
{
    destroy();
}

template< typename T >
bool DeviceVector< T >::isNull() const
{
    return( m_devicePointer == nullptr );
}

template< typename T >
bool DeviceVector< T >::notNull() const
{
    return( m_devicePointer != nullptr );
}

template< typename T >
int DeviceVector< T >::length() const
{
    return m_length;
}

template< typename T >
size_t DeviceVector< T >::sizeInBytes() const
{
    return m_sizeInBytes;
}

template< typename T >
void DeviceVector< T >::resize( int length )
{
    if( m_length == length )
    {
        return;
    }

    destroy();

    m_length = length;
    m_sizeInBytes = length * sizeof( T );

    checkCudaErrors( cudaMalloc( reinterpret_cast< void** >( &m_devicePointer ), m_sizeInBytes ) );
}


template< typename T >
void DeviceVector< T >::clear()
{
    checkCudaErrors( cudaMemset( m_devicePointer, 0, m_sizeInBytes ) );
}

template< typename T >
void DeviceVector< T >::fill( const T& value )
{
    std::vector< T > h_array( length(), value );
    copyFromHost( h_array );
}

template< typename T >
T DeviceVector< T >::get( int index ) const
{
    T output;
    checkCudaErrors( cudaMemcpy( &output, m_devicePointer + index, sizeof( T ), cudaMemcpyDeviceToHost ) );
    return output;
}

template< typename T >
T DeviceVector< T >::operator [] ( int index ) const
{
    return get( index );
}

template< typename T >
void DeviceVector< T >::set( int index, const T& value )
{
    checkCudaErrors( cudaMemcpy( m_devicePointer + index, &value, sizeof( T ), cudaMemcpyHostToDevice ) );
}

template< typename T >
void DeviceVector< T >::copyFromDevice( const DeviceVector< T >& src )
{
    resize( src.length() );
    checkCudaErrors( cudaMemcpy( m_devicePointer, src.m_devicePointer, src.m_sizeInBytes, cudaMemcpyDeviceToDevice ) );
}

template< typename T >
void DeviceVector< T >::copyFromHost( const std::vector< T >& src )
{
    resize( static_cast< int >( src.size() ) );
    checkCudaErrors( cudaMemcpy( m_devicePointer, src.data(), m_sizeInBytes, cudaMemcpyHostToDevice ) );
}

template< typename T >
bool DeviceVector< T >::copyFromHost( const Array1DView< T >& src, int dstOffset )
{
    if( dstOffset < 0 )
    {
        return false;
    }
    if( m_devicePointer == nullptr ||
        !( src.packed() ) ||
        length() - dstOffset < src.length() )
    {
        return false;
    }

    const T* srcPointer = src.pointer();
    T* dstPointer = m_devicePointer + dstOffset;
    size_t countInBytes = src.length() * sizeof( T );

    cudaError_t err = cudaMemcpy( dstPointer, srcPointer, countInBytes, cudaMemcpyHostToDevice );
    return( err == cudaSuccess );
}

template< typename T >
void DeviceVector< T >::copyToHost( std::vector< T >& dst ) const
{
    if( isNull() )
    {
        return;
    }

    // because STL does some stupid copying even when the lengths are the same
    if( dst.size() != length() )
    {
        dst.resize( length() );
    }

    T* dstPointer = dst.data();
    checkCudaErrors( cudaMemcpy( dstPointer, m_devicePointer, m_sizeInBytes, cudaMemcpyDeviceToHost ) );
}

template< typename T >
bool DeviceVector< T >::copyToHost( Array1DView< T >& dst, int srcOffset ) const
{
    if( srcOffset < 0 )
    {
        return false;
    }
    if( dst.pointer() == nullptr ||
        !( dst.packed() ) ||
        length() - srcOffset < dst.size() )
    {
        return false;
    }

    T* dstPointer = dst.pointer();
    T* srcPointer = m_devicePointer + srcOffset;
    size_t countInBytes = dst.size() * sizeof(T);

    cudaError_t err = cudaMemcpy( dstPointer, srcPointer, countInBytes, cudaMemcpyDeviceToHost );
    return( err == cudaSuccess );
}

template< typename T >
const T* DeviceVector< T >::devicePointer() const
{
    return m_devicePointer;
}

template< typename T >
T* DeviceVector< T >::devicePointer()
{
    return m_devicePointer;
}

template< typename T >
KernelVector< T > DeviceVector< T >::kernelVector() const
{
    return KernelVector< T >( m_devicePointer, m_length );
}

template< typename T >
void DeviceVector< T >::destroy()
{
    if( notNull() )
    {
        checkCudaErrors( cudaFree( m_devicePointer ) );
        m_devicePointer = nullptr;
    }

    m_sizeInBytes = 0;
    m_length = -1;
    m_devicePointer = nullptr;
}
