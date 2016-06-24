template< typename T >
__inline__  __device__
KernelArray2D< T >::KernelArray2D(
typename KernelArray2D< T >::VoidPointer pointer, const int2& size ) :
    md_pointer( reinterpret_cast< typename KernelArray2D< T >::UInt8Pointer >(
        pointer ) ),
    m_size( size ),
    m_pitch( size.x * sizeof( T ) )
{

}
template< typename T >
__inline__  __device__
KernelArray2D< T >::KernelArray2D(
    typename KernelArray2D< T >::VoidPointer pointer, const int2& size,
    size_t pitch ) :
    md_pointer( reinterpret_cast< typename KernelArray2D< T >::UInt8Pointer >(
         pointer ) ),
    m_size( size ),
    m_pitch( pitch )
{

}

template< typename T >
__inline__  __device__
T* KernelArray2D< T >::pointer() const
{
    return reinterpret_cast< T* >( md_pointer );
}

template< typename T >
__inline__  __device__
T* KernelArray2D< T >::elementPointer( const int2& xy ) const
{
    const int elementStride = sizeof( T );
    return reinterpret_cast< T* >( md_pointer +
        xy.y * m_pitch + xy.x * elementStride );
}

template< typename T >
__inline__  __device__
T* KernelArray2D< T >::rowPointer( int y ) const
{
    return reinterpret_cast< T* >( md_pointer + y * m_pitch );
}

template< typename T >
__inline__  __device__
int KernelArray2D< T >::width() const
{
    return m_size.x;
}

template< typename T >
__inline__  __device__
int KernelArray2D< T >::height() const
{
    return m_size.y;
}

template< typename T >
__inline__  __device__
size_t KernelArray2D< T >::pitch() const
{
    return m_pitch;
}

template< typename T >
__inline__  __device__
int2 KernelArray2D< T >::size() const
{
    return m_size;
}

template< typename T >
__inline__  __device__
T& KernelArray2D< T >::operator [] ( const int2& xy ) const
{
    return *elementPointer( xy );
}
