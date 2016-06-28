#include <thrust/fill.h>
#include <thrust/execution_policy.h>

template< typename T >
DeviceArray3D< T >::DeviceArray3D( const Vector3i& size )
{
    resize( size );
}

template< typename T >
DeviceArray3D< T >::DeviceArray3D( const DeviceArray3D< T >& copy )
{
    copyFromDevice( copy );
}

template< typename T >
DeviceArray3D< T >::DeviceArray3D( DeviceArray3D< T >&& move )
{
    destroy();

    m_size = move.m_size;
    m_pitchedPointer = move.m_pitchedPointer;
    m_extent = move.m_extent;

    move.m_size = Vector3i{ 0 };
    move.m_pitchedPointer = make_cudaPitchedPtr( nullptr, 0, 0, 0 );
    move.m_extent = make_cudaExtent( 0, 0, 0 );
}

template< typename T >
DeviceArray3D< T >& DeviceArray3D< T >::operator = ( const DeviceArray3D< T >& copy )
{
    if( this != &copy )
    {
        copyFromDevice( copy );
    }
    return *this;
}

template< typename T >
DeviceArray3D< T >& DeviceArray3D< T >::operator = ( DeviceArray3D< T >&& move )
{
    if( this != &move )
    {
        destroy();

        m_size = move.m_size;
        m_pitchedPointer = move.m_pitchedPointer;
        m_extent = move.m_extent;

        move.m_size = Vector3i{ 0 };
        move.m_pitchedPointer = make_cudaPitchedPtr( nullptr, 0, 0, 0 );
        move.m_extent = make_cudaExtent( 0, 0, 0 );
    }
    return *this;
}

template< typename T >
// virtual
DeviceArray3D< T >::~DeviceArray3D()
{
    destroy();
}

template< typename T >
bool DeviceArray3D< T >::isNull() const
{
    return( m_pitchedPointer.ptr == NULL );
}

template< typename T >
bool DeviceArray3D< T >::notNull() const
{
    return( m_pitchedPointer.ptr != NULL );
}

template< typename T >
int DeviceArray3D< T >::width() const
{
    return m_size.x;
}

template< typename T >
int DeviceArray3D< T >::height() const
{
    return m_size.y;
}

template< typename T >
int DeviceArray3D< T >::depth() const
{
    return m_size.z;
}

template< typename T >
Vector3i DeviceArray3D< T >::size() const
{
    return m_size;
}

template< typename T >
int DeviceArray3D< T >::numElements() const
{
    return m_size.x * m_size.y * m_size.z;
}

template< typename T >
size_t DeviceArray3D< T >::rowPitch() const
{
    return m_pitchedPointer.pitch;
}

template< typename T >
size_t DeviceArray3D< T >::slicePitch() const
{
    return m_pitchedPointer.pitch * m_size.y;
}

template< typename T >
size_t DeviceArray3D< T >::sizeInBytes() const
{
    return slicePitch() * m_size.z;
}

template< typename T >
void DeviceArray3D< T >::resize( const Vector3i& size )
{
    if( size == m_size )
    {
        return;
    }

    destroy();

    if( size.x > 0 && size.y > 0 && size.z > 0 )
    {
        m_size = size;
        m_extent = make_cudaExtent( size.x * sizeof( T ), size.y, size.z );

        checkCudaErrors
        (
            cudaMalloc3D( &m_pitchedPointer, m_extent )
        );
    }
}

template< typename T >
void DeviceArray3D< T >::clear()
{
    checkCudaErrors( cudaMemset3D( m_pitchedPointer, 0, m_extent ) );
}

template< typename T >
void DeviceArray3D< T >::fill( const T& value )
{
    // TODO(jiawen): write a function for this.
    if( m_pitchedPointer.xsize == m_size.x * sizeof( T ) )
    {
        T* begin = reinterpret_cast< T* >( m_pitchedPointer.ptr );
        T* end = begin + numElements();
        thrust::fill( thrust::device, begin, end, value );
    }
    else
    {
        // TODO(jiawen): optimize this with thrust::fill().
        Array3D< T > h_array( m_size, value );
        copyFromHost( h_array );
    }
}

template< typename T >
T DeviceArray3D< T >::get( const Vector3i& xyz ) const
{
    T output;

    const uint8_t* sourcePointer = reinterpret_cast< const uint8_t* >( m_pitchedPointer.ptr );
    const uint8_t* slicePointer = sourcePointer + xyz.z * slicePitch();
    const uint8_t* rowPointer = slicePointer + xyz.y * rowPitch();
    const uint8_t* elementPointer = rowPointer + xyz.x * sizeof( T );

    checkCudaErrors( cudaMemcpy( &output, elementPointer, sizeof( T ), cudaMemcpyDeviceToHost ) );

    return output;
}

template< typename T >
T DeviceArray3D< T >::operator [] ( const Vector3i& xyz ) const
{
    return get( xyz );
}

template< typename T >
void DeviceArray3D< T >::set( const Vector3i& xyz, const T& value )
{
    uint8_t* destinationPointer = reinterpret_cast< uint8_t* >( m_pitchedPointer.ptr );
    uint8_t* slicePointer = destinationPointer + xyz.z * slicePitch();
    uint8_t* rowPointer = slicePointer + xyz.y * rowPitch();
    uint8_t* elementPointer = rowPointer + xyz.x * sizeof( T );

    checkCudaErrors( cudaMemcpy( elementPointer, &value, sizeof( T ), cudaMemcpyHostToDevice ) );
}

template< typename T >
bool DeviceArray3D< T >::copyFromDevice( const DeviceArray3D< T >& src )
{
    if( isNull() || src.isNull() || m_size != src.size() )
    {
        return false;
    }

    cudaMemcpy3DParms params;

    params.kind = cudaMemcpyDeviceToDevice;

    params.srcPtr = src.m_pitchedPointer;
    params.srcArray = nullptr; // we're not copying a CUDA array
    params.srcPos = make_cudaPos( 0, 0, 0 );

    params.dstPtr = m_pitchedPointer;
    params.dstArray = nullptr; // we're not copying a CUDA array
    params.dstPos = make_cudaPos( 0, 0, 0 );

    params.extent = src.m_extent;

    checkCudaErrors( cudaMemcpy3D( &params ) );
    return true;
}

template< typename T >
bool DeviceArray3D< T >::copyFromHost( Array3DView< const T > src )
{
    if( isNull() || src.isNull() || m_size != src.size() )
    {
        return false;
    }

    cudaMemcpy3DParms params;

    params.kind = cudaMemcpyHostToDevice;

    // Since the source (on the host) is not pitched
    // make a pitchedPointer for it

    // using const_cast since CUDA is stupid and wants a void* instead of
    // const void* for src.
    T* srcPointer = const_cast< T* >( src.pointer() );
    params.srcPtr = make_cudaPitchedPtr( srcPointer, src.width() * sizeof( T ), src.width(), src.height() );
    params.srcArray = NULL; // we're not copying a CUDA array
    params.srcPos = make_cudaPos( 0, 0, 0 );

    params.dstPtr = m_pitchedPointer;
    params.dstArray = NULL; // we're not copying a CUDA array
    params.dstPos = make_cudaPos( 0, 0, 0 );

    params.extent = m_extent;

    checkCudaErrors( cudaMemcpy3D( &params ) );
    return true;
}

template< typename T >
bool DeviceArray3D< T >::copyToHost( Array3DView< T > dst ) const
{
    if( isNull() || dst.isNull() || m_size != dst.size() )
    {
        return false;
    }

    cudaMemcpy3DParms params;

    params.kind = cudaMemcpyDeviceToHost;

    params.srcPtr = m_pitchedPointer;
    params.srcArray = NULL; // we're not copying a CUDA array
    params.srcPos = make_cudaPos( 0, 0, 0 );

    // Since the destination (on the host) is not pitched
    // make a pitchedPointer for it
    params.dstPtr = make_cudaPitchedPtr( dst, dst.width() * sizeof( T ), dst.width(), dst.height() );
    params.dstArray = NULL; // we're not copying a CUDA array
    params.dstPos = make_cudaPos( 0, 0, 0 );

    params.extent = m_extent;

    // TODO(jiawen): check for error, return it.
    checkCudaErrors( cudaMemcpy3D( &params ) );
    return true;
}

template< typename T >
const cudaPitchedPtr DeviceArray3D< T >::pitchedPointer() const
{
    return m_pitchedPointer;
}

template< typename T >
cudaPitchedPtr DeviceArray3D< T >::pitchedPointer()
{
    return m_pitchedPointer;
}

template< typename T >
KernelArray3D< const T > DeviceArray3D< T >::readView() const
{
    return KernelArray3D< const T >( m_pitchedPointer, m_size.z );
}

template< typename T >
KernelArray3D< T > DeviceArray3D< T >::writeView() const
{
    return KernelArray3D< T >( m_pitchedPointer, m_size.z );
}

template< typename T >
void DeviceArray3D< T >::load( const char* filename )
{
    Array3D< T > h_arr( filename );
    if( !( h_arr.isNull() ) )
    {
        resize( h_arr.width(), h_arr.height(), h_arr.depth() );
        copyFromHost( h_arr );
    }
}

template< typename T >
void DeviceArray3D< T >::destroy()
{
    if( notNull() )
    {
        checkCudaErrors( cudaFree( m_pitchedPointer.ptr ) );
        m_pitchedPointer = { 0 };
    }

    m_size = Vector3i{ 0 };
    m_extent = { 0 };
}
