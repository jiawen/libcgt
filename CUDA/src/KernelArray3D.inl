// A cudaPitchedPtr contains
// width: logical width of the array in elements
// xsize = elementSize * width, in bytes
// ysize = height of the array in elements
// pitch = roundUpToAlignment( xsize ), in bytes
template< typename T >
__inline__ __device__ __host__
KernelArray3D< T >::KernelArray3D(
    cudaPitchedPtr d_pitchedPointer, int depth ) :
    md_pointer( reinterpret_cast< typename KernelArray3D::UInt8Pointer >(
        d_pitchedPointer.ptr ) ),
    m_rowPitch( d_pitchedPointer.pitch )
{
    m_size.x = static_cast< int >( d_pitchedPointer.xsize / sizeof( T ) );
    m_size.y = static_cast< int >( d_pitchedPointer.ysize );
    m_size.z = depth;
}

template< typename T >
__inline__ __device__ __host__
KernelArray3D< T >::KernelArray3D(
    typename KernelArray3D< T >::VoidPointer d_linearPointer,
    const int3& size ) :
    md_pointer( reinterpret_cast< typename KernelArray3D::UInt8Pointer >(
        d_linearPointer ) ),
    m_size( size ),
    m_rowPitch( size.x * sizeof( T ) )
{

}

template< typename T >
__inline__  __device__
T* KernelArray3D< T >::pointer() const
{
    return reinterpret_cast< T* >( md_pointer );
}

template< typename T >
__inline__ __device__
T* KernelArray3D< T >::elementPointer( const int3& xyz ) const
{
    const int elementStride = sizeof( T );
    uint8_t* p = md_pointer +
        xyz.x * elementStride + xyz.y * m_rowPitch + xyz.z * slicePitch();
    return reinterpret_cast< T* >( p );
}

template< typename T >
__inline__ __device__
T* KernelArray3D< T >::rowPointer( const int2& yz ) const
{
    int y = yz.x;
    int z = yz.y;

    uint8_t* p = md_pointer + y * m_rowPitch + z * slicePitch();
    return reinterpret_cast< T* >( p );
}

template< typename T >
__inline__ __device__
T* KernelArray3D< T >::slicePointer( int z ) const
{
    uint8_t* p = md_pointer + z * slicePitch();
    return reinterpret_cast< T* >( p );
}

template< typename T >
__inline__ __device__
int KernelArray3D< T >::width() const
{
    return m_size.x;
}

template< typename T >
__inline__ __device__
int KernelArray3D< T >::height() const
{
    return m_size.y;
}

template< typename T >
__inline__ __device__
int KernelArray3D< T >::depth() const
{
    return m_size.z;
}

template< typename T >
__inline__ __device__
int3 KernelArray3D< T >::size() const
{
    return m_size;
}

template< typename T >
__inline__ __device__
size_t KernelArray3D< T >::rowPitch() const
{
    return m_rowPitch;
}

template< typename T >
__inline__ __device__
size_t KernelArray3D< T >::slicePitch() const
{
    return m_rowPitch * m_size.y;
}

template< typename T >
__inline__ __device__
T& KernelArray3D< T >::operator [] ( const int3& xyz ) const
{
    return *elementPointer( xyz );
}
