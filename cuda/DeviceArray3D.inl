template< typename T >
DeviceArray3D< T >::DeviceArray3D( const Vector3i& size )
{
    resize( size );
}

template< typename T >
DeviceArray3D< T >::DeviceArray3D( const DeviceArray3D< T >& copy )
{
    resize( copy.size() );
    ::copy( copy, *this );
}

template< typename T >
DeviceArray3D< T >::DeviceArray3D( DeviceArray3D< T >&& move )
{
    destroy();

    m_size = move.m_size;
    m_pitchedPointer = move.m_pitchedPointer;

    move.m_size = Vector3i{ 0 };
    move.m_pitchedPointer = {};
}

template< typename T >
DeviceArray3D< T >& DeviceArray3D< T >::operator = ( const DeviceArray3D< T >& copy )
{
    if( this != &copy )
    {
        resize( copy.size() );
        ::copy( copy, *this );
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

        move.m_size = Vector3i{ 0 };
        move.m_pitchedPointer = {};
    }
    return *this;
}

template< typename T >
DeviceArray3D< T >::~DeviceArray3D()
{
    destroy();
}

template< typename T >
bool DeviceArray3D< T >::isNull() const
{
    return( m_pitchedPointer.ptr == nullptr );
}

template< typename T >
bool DeviceArray3D< T >::notNull() const
{
    return( m_pitchedPointer.ptr != nullptr );
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
size_t DeviceArray3D< T >::elementStrideBytes() const
{
    return sizeof( T );
}

template< typename T >
size_t DeviceArray3D< T >::rowStrideBytes() const
{
    return m_pitchedPointer.pitch;
}

template< typename T >
size_t DeviceArray3D< T >::sliceStrideBytes() const
{
    return m_pitchedPointer.pitch * m_size.y;
}

template< typename T >
size_t DeviceArray3D< T >::sizeInBytes() const
{
    return slicePitch() * m_size.z;
}

template< typename T >
cudaError DeviceArray3D< T >::resize( const Vector3i& size )
{
    if( size == m_size )
    {
        return cudaSuccess;
    }
    // Explicitly allow resize( 0, 0, 0 ) for invoking constructor from a null
    // right hand side.
    if( size.x < 0 && size.y < 0 && size.z < 0 )
    {
        return cudaErrorInvalidValue;
    }

    cudaError err = destroy();
    if( err == cudaSuccess )
    {
        if( size.x > 0 && size.y > 0 && size.z > 0 )
        {
            // Allocate memory by first making a cudaExtent, which expects
            // width in *bytes*, but height and depth in elements.
            cudaExtent newExt
            {
                static_cast< size_t >( size.x * sizeof( T ) ),
                static_cast< size_t >( size.y ),
                static_cast< size_t >( size.z )
            };
            err = cudaMalloc3D( &m_pitchedPointer, newExt );
            if( err == cudaSuccess )
            {
                m_size = size;
            }
        }
    }
    return err;
}

template< typename T >
cudaError DeviceArray3D< T >::clear()
{
    return cudaMemset3D( m_pitchedPointer, 0, extent() );
}

template< typename T >
void DeviceArray3D< T >::fill( const T& value )
{
    if( rowsArePacked() )
    {
        T* begin = slicePointer( 0 );
        T* end = slicePointer( depth() );
        thrust::fill( thrust::device, begin, end, value );
    }
    else
    {
        // TODO(jiawen): optimize this with thrust::fill().
        Array3D< T > h_array( m_size, value );
        copy( h_array.readView(), *this );
    }
}

template< typename T >
T DeviceArray3D< T >::get( const Vector3i& xyz, cudaError& err ) const
{
    T output;
    err = cudaMemcpy( &output, elementPointer( xyz ), sizeof( T ),
        cudaMemcpyDeviceToHost );
    return output;
}

template< typename T >
T DeviceArray3D< T >::operator [] ( const Vector3i& xyz ) const
{
    cudaError err;
    return get( xyz, err );
}

template< typename T >
cudaError DeviceArray3D< T >::set( const Vector3i& xyz, const T& value )
{
    return cudaMemcpy( elementPointer( xyz ), &value, sizeof( T ),
        cudaMemcpyHostToDevice );
}

template< typename T >
const T* DeviceArray3D< T >::pointer() const
{
    return reinterpret_cast< const T* >( m_pitchedPointer.ptr );
}

template< typename T >
T* DeviceArray3D< T >::pointer()
{
    return reinterpret_cast< T* >( m_pitchedPointer.ptr );
}

template< typename T >
const T* DeviceArray3D< T >::elementPointer( const Vector3i& xyz ) const
{
    const uint8_t* p = reinterpret_cast< const uint8_t* >( pointer() );
    return reinterpret_cast< const T* >(
        p +
        xyz.x * elementStrideBytes() +
        xyz.y * rowStrideBytes() +
        xyz.x * sliceStrideBytes() );
}

template< typename T >
T* DeviceArray3D< T >::elementPointer( const Vector3i& xyz )
{
    uint8_t* p = reinterpret_cast< uint8_t* >( pointer() );
    return reinterpret_cast< T* >(
        p +
        xyz.x * elementStrideBytes() +
        xyz.y * rowStrideBytes() +
        xyz.z * sliceStrideBytes() );
}

template< typename T >
const T* DeviceArray3D< T >::rowPointer( int y, int z ) const
{
    return elementPointer( { 0, y, z } );
}

template< typename T >
T* DeviceArray3D< T >::rowPointer( int y, int z )
{
    return elementPointer( { 0, y, z } );
}

template< typename T >
const T* DeviceArray3D< T >::slicePointer( int z ) const
{
    return elementPointer( { 0, 0, z } );
}

template< typename T >
T* DeviceArray3D< T >::slicePointer( int z )
{
    return elementPointer( { 0, 0, z } );
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
cudaExtent DeviceArray3D< T >::extent() const
{
    return
    {
        static_cast< size_t >( width() * sizeof( T ) ),
        static_cast< size_t >( height() ),
        static_cast< size_t >( depth() )
    };
}

template< typename T >
bool DeviceArray3D< T >::rowsArePacked() const
{
    return rowStrideBytes() == ( m_size.x * elementStrideBytes() );
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
        copy( h_arr.readView(), *this );
    }
}

template< typename T >
cudaError DeviceArray3D< T >::destroy()
{
    cudaError err = cudaSuccess;
    if( notNull() )
    {
        err = cudaFree( m_pitchedPointer.ptr );
        m_pitchedPointer = { 0 };
    }

    m_size = Vector3i{ 0 };
    return err;
}

template< typename T >
cudaError copy( Array3DReadView< T > src, DeviceArray3D< T >& dst )
{
    if( src.isNull() || dst.isNull() || src.size() != src.size() )
    {
        return cudaErrorInvalidValue;
    }

    cudaMemcpy3DParms params;

    params.kind = cudaMemcpyHostToDevice;

    // Make a cudaPitchedPtr for the source host pointer.
    // Using const_cast since CUDA is stupid and wants a void* instead of
    // const void* for src.
    T* srcPointer = const_cast< T* >( src.pointer() );
    params.srcPtr = make_cudaPitchedPtr( srcPointer,
        src.rowStrideBytes(), src.width() * sizeof( T ), src.height() );
    params.srcArray = nullptr; // We're not copying a CUDA array.
    params.srcPos = {};

    params.dstPtr = dst.pitchedPointer();
    params.dstArray = nullptr; // We're not copying a CUDA array.
    params.dstPos = {};

    params.extent = dst.extent();

    return cudaMemcpy3D( &params );
}

template< typename T >
cudaError copy( const DeviceArray3D< T >& src, Array3DWriteView< T > dst )
{
    if( src.isNull() || dst.isNull() || src.size() != dst.size() )
    {
        return cudaErrorInvalidValue;
    }

    cudaMemcpy3DParms params;

    params.kind = cudaMemcpyDeviceToHost;

    params.srcPtr = src.pitchedPointer();
    params.srcArray = nullptr; // We're not copying a CUDA array.
    params.srcPos = {};

    // Make a cudaPitchedPtr for the destination host pointer.
    params.dstPtr = make_cudaPitchedPtr( dst.pointer(),
        dst.rowStrideBytes(), dst.width() * sizeof( T ), dst.height() );
    params.dstArray = nullptr; // We're not copying a CUDA array.
    params.dstPos = {};

    params.extent = src.extent();

    return cudaMemcpy3D( &params );
}

template< typename T >
cudaError copy( const DeviceArray3D< T >& src, DeviceArray3D< T >& dst )
{
    if( src.isNull() || dst.isNull() || src.size() != src.size() )
    {
        return cudaErrorInvalidValue;
    }

    cudaMemcpy3DParms params;

    params.kind = cudaMemcpyDeviceToDevice;

    params.srcPtr = src.pitchedPointer();
    params.srcArray = nullptr; // We're not copying a CUDA array.
    params.srcPos = {};

    params.dstPtr = dst.pitchedPointer();
    params.dstArray = nullptr; // We're not copying a CUDA array.
    params.dstPos = {};

    params.extent = src.extent();

    return cudaMemcpy3D( &params );
}
