template< typename T >
Array3DView< T >::Array3DView() :
    m_size( Vector3i{ 0, 0, 0 } ),
    m_strides( Vector3i{ 0, 0, 0 } ),
    m_pPointer( nullptr )
{

}

template< typename T >
Array3DView< T >::Array3DView( void* pPointer, const Vector3i& size ) :
    m_size( size ),
    m_strides( Vector3i{
        static_cast< int >( sizeof( T ) ),
        static_cast< int >( size.x * sizeof( T ) ),
        static_cast< int >( size.x * size.y * sizeof( T ) ) } ),
    m_pPointer( reinterpret_cast< typename WrapConstPointerT< T, uint8_t >::pointer >( pPointer ) )
{

}

template< typename T >
Array3DView< T >::Array3DView( void* pPointer,
    const Vector3i& size, const Vector3i& strides ) :
    m_size( size ),
    m_strides( strides ),
    m_pPointer( reinterpret_cast< typename WrapConstPointerT< T, uint8_t >::pointer >( pPointer ) )
{

}

template< typename T >
bool Array3DView< T >::isNull() const
{
    return( m_pPointer == nullptr );
}

template< typename T >
bool Array3DView< T >::notNull() const
{
    return( m_pPointer != nullptr );
}

template< typename T >
Array3DView< T >::operator const T* () const
{
    return m_pPointer;
}

template< typename T >
Array3DView< T >::operator T* ()
{
    return m_pPointer;
}

template< typename T >
const T* Array3DView< T >::pointer() const
{
    return m_pPointer;
}

template< typename T >
T* Array3DView< T >::pointer()
{
    return m_pPointer;
}

template< typename T >
T* Array3DView< T >::elementPointer( const Vector3i& xyz )
{
    return reinterpret_cast< T* >( &( m_pPointer[ Vector3i::dot( xyz, m_strides ) ] ) );
}

template< typename T >
T* Array3DView< T >::rowPointer( const Vector2i& yz )
{
    return elementPointer( Vector3i( 0, yz ) );
}

template< typename T >
T* Array3DView< T >::slicePointer( int z )
{
    return elementPointer( { 0, 0, z } );
}

template< typename T >
T& Array3DView< T >::operator [] ( int k )
{
    int x;
    int y;
    int z;
    Indexing::indexToSubscript3D( k, m_size.x, m_size.y, x, y, z );
    return ( *this )[ { x, y, z } ];
}

template< typename T >
T& Array3DView< T >::operator [] ( const Vector3i& xyz )
{
    return *elementPointer( xyz );
}

template< typename T >
int Array3DView< T >::width() const
{
    return m_size.x;
}

template< typename T >
int Array3DView< T >::height() const
{
    return m_size.y;
}

template< typename T >
int Array3DView< T >::depth() const
{
    return m_size.z;
}

template< typename T >
Vector3i Array3DView< T >::size() const
{
    return m_size;
}

template< typename T >
int Array3DView< T >::numElements() const
{
    return m_size.x * m_size.y * m_size.z;
}

template< typename T >
int Array3DView< T >::elementStrideBytes() const
{
    return m_strides.x;
}

template< typename T >
int Array3DView< T >::rowStrideBytes() const
{
    return m_strides.y;
}

template< typename T >
int Array3DView< T >::sliceStrideBytes() const
{
    return m_strides.z;
}

template< typename T >
Vector3i Array3DView< T >::stride() const
{
    return m_strides;
}

template< typename T >
bool Array3DView< T >::elementsArePacked() const
{
    return elementStrideBytes() == sizeof( T );
}

template< typename T >
bool Array3DView< T >::rowsArePacked() const
{
    return rowStrideBytes() == ( width() * elementStrideBytes() );
}

template< typename T >
bool Array3DView< T >::slicesArePacked() const
{
    return sliceStrideBytes() == ( height() * rowStrideBytes() );
}

template< typename T >
bool Array3DView< T >::packed() const
{
    return elementsArePacked() && rowsArePacked() && slicesArePacked();
}
