template< typename T >
Array3DView< T >::Array3DView( Array3DView< T >::VoidPointer pointer,
    const Vector3i& size ) :
    m_size( size ),
    m_stride( Vector3i{
        static_cast< int >( sizeof( T ) ),
        static_cast< int >( size.x * sizeof( T ) ),
        static_cast< int >( size.x * size.y * sizeof( T ) ) } ),
    m_pointer( reinterpret_cast< typename Array3DView< T >::UInt8Pointer >(
        pointer ) )
{

}

template< typename T >
Array3DView< T >::Array3DView( Array3DView< T >::VoidPointer pointer,
    const Vector3i& size, const Vector3i& strides ) :
    m_size( size ),
    m_stride( strides ),
    m_pointer( reinterpret_cast< typename Array3DView< T >::UInt8Pointer >(
        pointer ) )
{

}

template< typename T >
bool Array3DView< T >::isNull() const
{
    return( m_pointer == nullptr );
}

template< typename T >
bool Array3DView< T >::notNull() const
{
    return( m_pointer != nullptr );
}

template< typename T >
template< typename U, typename >
Array3DView< T >::operator const T* () const
{
    return reinterpret_cast< const T* >( m_pointer );
}

template< typename T >
Array3DView< T >::operator T* ()
{
    return reinterpret_cast< T* >( m_pointer );
}

template< typename T >
template< typename U, typename >
const T* Array3DView< T >::pointer() const
{
    return reinterpret_cast< const T* >( m_pointer );
}

template< typename T >
T* Array3DView< T >::pointer()
{
    return reinterpret_cast< T* >( m_pointer );
}

template< typename T >
T* Array3DView< T >::elementPointer( const Vector3i& xyz )
{
    return reinterpret_cast< T* >( &( m_pointer[ Vector3i::dot( xyz, m_stride ) ] ) );
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
    return m_stride.x;
}

template< typename T >
int Array3DView< T >::rowStrideBytes() const
{
    return m_stride.y;
}

template< typename T >
int Array3DView< T >::sliceStrideBytes() const
{
    return m_stride.z;
}

template< typename T >
Vector3i Array3DView< T >::stride() const
{
    return m_stride;
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

template< typename T >
template< typename U, typename >
Array3DView< T >::operator Array3DView< const T >() const
{
    return Array3DView< const T >( m_pointer, m_size, m_stride );
}

template< typename T >
Array1DView< T > Array3DView< T >::xySlice( int x, int y )
{
    return Array1DView< T >( elementPointer( { x, y, 0 } ), m_size.z );
}

template< typename T >
Array1DView< T > Array3DView< T >::yzSlice( int y, int z )
{
    return Array1DView< T >( elementPointer( { 0, y, z } ), m_size.x );
}

template< typename T >
Array1DView< T > Array3DView< T >::xzSlice( int x, int z )
{
    return Array1DView< T >( elementPointer( { x, 0, z } ), m_size.y );
}

template< typename T >
Array2DView< T > Array3DView< T >::xSlice( int x )
{
    return Array2DView< T >( elementPointer( { x, 0, 0 } ), m_size.yz,
        m_stride.yz );
}

template< typename T >
Array2DView< T > Array3DView< T >::ySlice( int y )
{
    return Array2DView< T >( rowPointer( { y, 0 } ), m_size.xz(), m_stride.xz() );
}

template< typename T >
Array2DView< T > Array3DView< T >::zSlice( int z )
{
    return Array2DView< T >( slicePointer( z ), m_size.xy, m_stride.xy );
}
