template< typename T >
Array1DReadView< T >::Array1DReadView( const void* pointer, size_t size ) :
    m_size( size ),
    m_stride( sizeof( T ) ),
    m_pointer( reinterpret_cast< const uint8_t* >( pointer ) )
{

}

template< typename T >
Array1DReadView< T >::Array1DReadView( const void* pointer,
    size_t size, std::ptrdiff_t stride ) :
    m_size( size ),
    m_stride( stride ),
    m_pointer( reinterpret_cast< const uint8_t* >( pointer ) )
{

}

template< typename T >
bool Array1DReadView< T >::isNull() const
{
    return( m_pointer == nullptr );
}

template< typename T >
bool Array1DReadView< T >::notNull() const
{
    return( m_pointer != nullptr );
}

template< typename T >
void Array1DReadView< T >::setNull()
{
    m_size = 0;
    m_stride = 0;
    m_pointer = nullptr;
}

template< typename T >
Array1DReadView< T >::operator const T* () const
{
    return reinterpret_cast< const T* >( m_pointer );
}

template< typename T >
const T* Array1DReadView< T >::pointer() const
{
    return reinterpret_cast< const T* >( m_pointer );
}

template< typename T >
const T* Array1DReadView< T >::elementPointer( size_t x ) const
{
    return reinterpret_cast< const T* >(
        &( m_pointer[ x * elementStrideBytes() ] ) );
}

template< typename T >
const T& Array1DReadView< T >::operator [] ( size_t x ) const
{
    return *( elementPointer( x ) );
}

template< typename T >
size_t Array1DReadView< T >::width() const
{
    return m_size;
}

template< typename T >
size_t Array1DReadView< T >::size() const
{
    return m_size;
}

template< typename T >
size_t Array1DReadView< T >::numElements() const
{
    return m_size;
}

template< typename T >
std::ptrdiff_t Array1DReadView< T >::elementStrideBytes() const
{
    return m_stride;
}

template< typename T >
std::ptrdiff_t Array1DReadView< T >::stride() const
{
    return elementStrideBytes();
}

template< typename T >
bool Array1DReadView< T >::elementsArePacked() const
{
    return m_stride == sizeof( T );
}

template< typename T >
bool Array1DReadView< T >::packed() const
{
    return elementsArePacked();
}

template< typename T >
Array1DWriteView< T >::Array1DWriteView( void* pointer,
    size_t size ) :
    Array1DReadView< T >( pointer, size ),
    m_write_pointer( reinterpret_cast< uint8_t* >( pointer ) )
{

}

template< typename T >
Array1DWriteView< T >::Array1DWriteView( void* pointer,
    size_t size, std::ptrdiff_t stride ) :
    Array1DReadView< T >( pointer, size, stride ),
    m_write_pointer( reinterpret_cast< uint8_t* >( pointer ) )
{

}

template< typename T >
Array1DWriteView< T >::operator T* () const
{
    return reinterpret_cast< T* >( m_write_pointer );
}

template< typename T >
T* Array1DWriteView< T >::pointer() const
{
    return reinterpret_cast< T* >( m_write_pointer );
}

template< typename T >
T* Array1DWriteView< T >::elementPointer( size_t x ) const
{
    return reinterpret_cast< T* >(
        &( m_write_pointer[ x * this->elementStrideBytes() ] ) );
}

template< typename T >
T& Array1DWriteView< T >::operator [] ( size_t x ) const
{
    return *( elementPointer( x ) );
}

template< typename T >
Array2DReadView< T >::Array2DReadView( const void* pointer,
    const Vector2i& size ) :
    m_size( size ),
    m_stride
    (
        static_cast< int >( sizeof( T ) ),
        static_cast< int >( size.x * sizeof( T ) )
    ),
    m_pointer( reinterpret_cast< const uint8_t* >( pointer ) )
{

}

template< typename T >
Array2DReadView< T >::Array2DReadView( const void* pointer,
    const Vector2i& size, const Vector2i& stride ) :
    m_size( size ),
    m_stride( stride ),
    m_pointer( reinterpret_cast< const uint8_t* >( pointer ) )
{

}

template< typename T >
bool Array2DReadView< T >::isNull() const
{
    return( m_pointer == nullptr );
}

template< typename T >
bool Array2DReadView< T >::notNull() const
{
    return( m_pointer != nullptr );
}

template< typename T >
void Array2DReadView< T >::setNull()
{
    m_size = { 0, 0 };
    m_stride = { 0, 0 };
    m_pointer = nullptr;
}

template< typename T >
Array2DReadView< T >::operator const T* () const
{
    return reinterpret_cast< const T* >( m_pointer );
}

template< typename T >
const T* Array2DReadView< T >::pointer() const
{
    return reinterpret_cast< const T* >( m_pointer );
}

template< typename T >
const T* Array2DReadView< T >::elementPointer( const Vector2i& xy ) const
{
    return reinterpret_cast< const T* >(
        &( m_pointer[ Vector2i::dot( xy, m_stride ) ] ) );
}

template< typename T >
const T* Array2DReadView< T >::rowPointer( int y ) const
{
    return reinterpret_cast< const T* >( &( m_pointer[ y * m_stride.y ] ) );
}

template< typename T >
const T& Array2DReadView< T >::operator [] ( int k ) const
{
    int x;
    int y;
    Indexing::indexToSubscript2D( k, m_size.x, x, y );
    return ( *this )[ { x, y } ];
}

template< typename T >
const T& Array2DReadView< T >::operator [] ( const Vector2i& xy ) const
{
    return *( elementPointer( xy ) );
}

template< typename T >
int Array2DReadView< T >::width() const
{
    return m_size.x;
}

template< typename T >
int Array2DReadView< T >::height() const
{
    return m_size.y;
}

template< typename T >
Vector2i Array2DReadView< T >::size() const
{
    return m_size;
}

template< typename T >
int Array2DReadView< T >::numElements() const
{
    return m_size.x * m_size.y;
}

template< typename T >
int Array2DReadView< T >::elementStrideBytes() const
{
    return m_stride.x;
}

template< typename T >
int Array2DReadView< T >::rowStrideBytes() const
{
    return m_stride.y;
}

template< typename T >
Vector2i Array2DReadView< T >::stride() const
{
    return m_stride;
}

template< typename T >
bool Array2DReadView< T >::elementsArePacked() const
{
    return m_stride.x == sizeof( T );
}

template< typename T >
bool Array2DReadView< T >::rowsArePacked() const
{
    return m_stride.y == ( m_size.x * m_stride.x );
}

template< typename T >
bool Array2DReadView< T >::packed() const
{
    return elementsArePacked() && rowsArePacked();
}

template< typename T >
Array1DReadView< T > Array2DReadView< T >::row( int y )
{
    return Array1DReadView< T >( rowPointer( y ), m_size.x, m_stride.x );
}

template< typename T >
Array1DReadView< T > Array2DReadView< T >::column( int x )
{
    return Array1DReadView< T >( elementPointer( { x, 0 } ),
        m_size.y, m_stride.y );
}

template< typename T >
Array2DWriteView< T >::Array2DWriteView( void* pointer,
    const Vector2i& size ) :
    Array2DReadView< T >( pointer, size ),
    m_write_pointer( reinterpret_cast< uint8_t* >( pointer ) )
{

}

template< typename T >
Array2DWriteView< T >::Array2DWriteView( void* pointer,
    const Vector2i& size, const Vector2i& stride ) :
    Array2DReadView< T >( pointer, size, stride ),
    m_write_pointer( reinterpret_cast< uint8_t* >( pointer ) )
{

}

template< typename T >
Array2DWriteView< T >::operator T* () const
{
    return reinterpret_cast< T* >( m_write_pointer );
}

template< typename T >
T* Array2DWriteView< T >::pointer() const
{
    return reinterpret_cast< T* >( m_write_pointer );
}

template< typename T >
T* Array2DWriteView< T >::elementPointer( const Vector2i& xy ) const
{
    return reinterpret_cast< T* >(
        &( m_write_pointer[ Vector2i::dot( xy, this->stride() ) ] ) );
}

template< typename T >
T* Array2DWriteView< T >::rowPointer( int y ) const
{
    return reinterpret_cast< T* >(
        &( m_write_pointer[ y * this->stride().y ] ) );
}

template< typename T >
T& Array2DWriteView< T >::operator [] ( int k ) const
{
    int x;
    int y;
    Indexing::indexToSubscript2D( k, this->size().x, x, y );
    return ( *this )[ { x, y } ];
}

template< typename T >
T& Array2DWriteView< T >::operator [] ( const Vector2i& xy ) const
{
    return *( elementPointer( xy ) );
}

template< typename T >
Array1DWriteView< T > Array2DWriteView< T >::row( int y )
{
    return Array1DWriteView< T >( rowPointer( y ),
        this->size().x, this->stride().x );
}

template< typename T >
Array1DWriteView< T > Array2DWriteView< T >::column( int x )
{
    return Array1DWriteView< T >( elementPointer( { x, 0 } ),
        this->size().y, this->stride().y );
}
