#include <common/Array1D.h>

template< typename T >
DeviceArray1D< T >::DeviceArray1D( int length )
{
    resize( length );
}

template< typename T >
DeviceArray1D< T >::DeviceArray1D( const DeviceArray1D< T >& copy )
{
    copyFromDevice( copy );
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
DeviceArray1D< T >& DeviceArray1D< T >::operator = ( const DeviceArray1D< T >& copy )
{
    if( this != &copy )
    {
        copyFromDevice( copy );
    }
    return *this;
}

template< typename T >
DeviceArray1D< T >& DeviceArray1D< T >::operator = ( DeviceArray1D< T >&& move )
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
int DeviceArray1D< T >::length() const
{
    return m_length;
}

template< typename T >
size_t DeviceArray1D< T >::sizeInBytes() const
{
    return m_length * sizeof( T );
}

template< typename T >
void DeviceArray1D< T >::resize( int length )
{
    if( m_length == length )
    {
        return;
    }

    destroy();

    m_length = length;

    checkCudaErrors
    (
        cudaMalloc
        (
            reinterpret_cast< void** >( &m_devicePointer ),
            length * sizeof( T )
        )
    );
}


template< typename T >
void DeviceArray1D< T >::clear()
{
    checkCudaErrors( cudaMemset( m_devicePointer, 0, sizeInBytes() ) );
}

template< typename T >
void DeviceArray1D< T >::fill( const T& value )
{
    // TODO(jiawen): use thrust::fill().
    Array1D< T > h_array( length(), value );
    copyFromHost( h_array );
}

template< typename T >
T DeviceArray1D< T >::get( int x ) const
{
    T output;
    checkCudaErrors
    (
        cudaMemcpy
        (
            &output, elementPointer( x ), sizeof( T ),
            cudaMemcpyDeviceToHost
        )
    );
    return output;
}

template< typename T >
T DeviceArray1D< T >::operator [] ( int x ) const
{
    return get( x );
}

template< typename T >
void DeviceArray1D< T >::set( int x, const T& value )
{
    checkCudaErrors
    (
        cudaMemcpy
        (
            elementPointer( x ), &value, sizeof( T ),
            cudaMemcpyHostToDevice
        )
    );
}

template< typename T >
void DeviceArray1D< T >::copyFromDevice( const DeviceArray1D< T >& src )
{
    resize( src.length() );
    checkCudaErrors
    (
        cudaMemcpy
        (
            m_devicePointer, src.m_devicePointer,
            src.sizeInBytes(), cudaMemcpyDeviceToDevice
        )
    );
}

template< typename T >
bool DeviceArray1D< T >::copyFromHost( Array1DView< const T > src, int dstOffset )
{
    if( dstOffset < 0 )
    {
        return false;
    }
    if( m_devicePointer == nullptr ||
        !( src.packed() ) ||
        length() - dstOffset < src.size() )
    {
        return false;
    }

    const T* srcPointer = src.pointer();
    T* dstPointer = elementPointer( dstOffset );
    size_t srcSizeBytes = src.size() * sizeof( T );

    cudaError_t err = cudaMemcpy( dstPointer, srcPointer, srcSizeBytes,
        cudaMemcpyHostToDevice );
    return( err == cudaSuccess );
}

template< typename T >
bool DeviceArray1D< T >::copyToHost( Array1DView< T > dst, int srcOffset ) const
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
    const T* srcPointer = elementPointer( srcOffset );
    size_t dstSizeBytes = dst.size() * sizeof(T);

    cudaError_t err = cudaMemcpy( dstPointer, srcPointer, dstSizeBytes,
        cudaMemcpyDeviceToHost );
    return( err == cudaSuccess );
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
const T* DeviceArray1D< T >::elementPointer( int x ) const
{
    return reinterpret_cast< const T* >( m_devicePointer + x * sizeof( T ) );
}

template< typename T >
T* DeviceArray1D< T >::elementPointer( int x )
{
    return reinterpret_cast< T* >( m_devicePointer + x * sizeof( T ) );
}

template< typename T >
KernelArray1D< T > DeviceArray1D< T >::kernelArray1D()
{
    return KernelArray1D< T >( pointer(), m_length );
}

template< typename T >
void DeviceArray1D< T >::destroy()
{
    if( notNull() )
    {
        checkCudaErrors( cudaFree( m_devicePointer ) );
        m_devicePointer = nullptr;
    }

    m_length = 0;
}
