template< typename T >
Array2DView< T >::Array2DView( void* pointer, const Vector2i& size ) :
    m_size( size ),
    m_stride( Vector2i{ static_cast< int >( sizeof( T ) ), static_cast< int >( size.x * sizeof( T ) ) } ),
    m_pointer( reinterpret_cast< typename WrapConstPointerT< T, uint8_t >::pointer >( pointer ) )
{

}

template< typename T >
Array2DView< T >::Array2DView( void* pointer, const Vector2i& size, const Vector2i& strides ) :
    m_size( size ),
    m_stride( strides ),
    m_pointer( reinterpret_cast< typename WrapConstPointerT< T, uint8_t >::pointer >( pointer ) )
{

}

template< typename T >
bool Array2DView< T >::isNull( ) const
{
    return( m_pointer == nullptr );
}

template< typename T >
bool Array2DView< T >::notNull( ) const
{
    return( m_pointer != nullptr );
}

template< typename T >
Array2DView< T >::operator const T* () const
{
    return reinterpret_cast< T* >( m_pointer );
}

template< typename T >
Array2DView< T >::operator T* ()
{
    return reinterpret_cast< T* >( m_pointer );
}

template< typename T >
const T* Array2DView< T >::pointer() const
{
    return reinterpret_cast< T* >( m_pointer );
}

template< typename T >
T* Array2DView< T >::pointer()
{
    return reinterpret_cast< T* >( m_pointer );
}

template< typename T >
T* Array2DView< T >::elementPointer( const Vector2i& xy )
{
    return reinterpret_cast< T* >( &( m_pointer[ Vector2i::dot( xy, m_stride ) ] ) );
}

template< typename T >
T* Array2DView< T >::rowPointer( int y )
{
    return reinterpret_cast< T* >( &( m_pointer[ y * m_stride.y ] ) );
}

template< typename T >
T& Array2DView< T >::operator [] ( int k )
{
    int x;
    int y;
    Indexing::indexToSubscript2D( k, m_size.x, x, y );
    return ( *this )[ { x, y } ];
}

template< typename T >
T& Array2DView< T >::operator [] ( const Vector2i& xy )
{
    return *( elementPointer( xy ) );
}

template< typename T >
int Array2DView< T >::width() const
{
    return m_size.x;
}

template< typename T >
int Array2DView< T >::height() const
{
    return m_size.y;
}

template< typename T >
Vector2i Array2DView< T >::size() const
{
    return m_size;
}

template< typename T >
int Array2DView< T >::numElements() const
{
    return m_size.x * m_size.y;
}

template< typename T >
int Array2DView< T >::elementStrideBytes() const
{
    return m_stride.x;
}

template< typename T >
int Array2DView< T >::rowStrideBytes() const
{
    return m_stride.y;
}

template< typename T >
Vector2i Array2DView< T >::stride() const
{
    return m_stride;
}

template< typename T >
bool Array2DView< T >::elementsArePacked() const
{
    return m_stride.x == sizeof( T );
}

template< typename T >
bool Array2DView< T >::rowsArePacked() const
{
    return m_stride.y == ( m_size.x * m_stride.x );
}

template< typename T >
bool Array2DView< T >::packed() const
{
    return elementsArePacked() && rowsArePacked();
}

template< typename T >
template< typename U, typename >
Array2DView< T >::operator Array2DView< const T >() const
{
    return Array2DView< const T >( m_pointer, m_size, m_stride );
}

template< typename T >
Array1DView< T > Array2DView< T >::row( int y )
{
    return Array1DView< T >( rowPointer( y ), m_size.x, m_stride.x );
}

template< typename T >
Array1DView< T > Array2DView< T >::column( int x )
{
    return Array1DView< T >( elementPointer( { x, 0 } ), m_size.y, m_stride.y );
}
