#include <cassert>

template< typename T >
Array2D< T >::Array2D( void* pointer, const Vector2i& size,
    Allocator* allocator ) :
    Array2D
    (
        pointer,
        size,
        {
            static_cast< int >( sizeof( T ) ),
            static_cast< int >( size.x * sizeof( T ) )
        },
        allocator
    )
{

}

template< typename T >
Array2D< T >::Array2D( void* pointer, const Vector2i& size,
    const Vector2i& stride, Allocator* allocator ) :
    m_size( size ),
    m_stride( stride ),
    m_data( reinterpret_cast< uint8_t* >( pointer ) ),
    m_allocator( allocator )
{

}

template< typename T >
Array2D< T >::Array2D( const Vector2i& size, const T& fillValue,
    Allocator* allocator ) :
    Array2D
    (
        size,
        {
            static_cast< int >( sizeof( T ) ),
            static_cast< int >( size.x * sizeof( T ) )
        },
        fillValue,
        allocator
    )
{
}

template< typename T >
Array2D< T >::Array2D( const Vector2i& size, const Vector2i& stride,
    const T& fillValue, Allocator* allocator ) :
    m_allocator( allocator )
{
    assert( size.x >= 0 );
    assert( size.y >= 0 );
    assert( stride.x >= sizeof( T ) );
    assert( stride.y >= size.x * sizeof( T ) );
    resize( size, stride );
    fill( fillValue );
}

template< typename T >
Array2D< T >::Array2D( const Array2D< T >& copy )
{
    resize( copy.m_size, copy.m_stride );
    if( copy.notNull() )
    {
        memcpy( m_data, copy.m_data, m_stride.y * m_size.y );
    }
}

template< typename T >
Array2D< T >::Array2D( Array2D< T >&& move )
{
    invalidate();
    m_size = move.m_size;
    m_stride = move.m_stride;
    m_data = move.m_data;
    m_allocator = move.m_allocator;

    move.m_size = { 0, 0 };
    move.m_stride = { 0, 0 };
    move.m_data = nullptr;
}

template< typename T >
Array2D< T >& Array2D< T >::operator = ( const Array2D< T >& copy )
{
    if( this != &copy )
    {
        resize( copy.m_size, copy.m_stride );
        if( copy.notNull() )
        {
            memcpy( m_data, copy.m_data, m_stride.y * m_size.y );
        }
    }
    return *this;
}

template< typename T >
Array2D< T >& Array2D< T >::operator = ( Array2D< T >&& move )
{
    if( this != &move )
    {
        invalidate();
        m_size = move.m_size;
        m_stride = move.m_stride;
        m_data = move.m_data;
        m_allocator = move.m_allocator;

        move.m_size = { 0, 0 };
        move.m_stride = { 0, 0 };
        move.m_data = nullptr;
    }
    return *this;
}

// virtual
template< typename T >
Array2D< T >::~Array2D()
{
    invalidate();
}

template< typename T >
bool Array2D< T >::isNull() const
{
    return( m_data == nullptr );
}

template< typename T >
bool Array2D< T >::notNull() const
{
    return( m_data != nullptr );
}

template< typename T >
std::pair< Array2DWriteView< T >, Allocator* > Array2D< T >::relinquish()
{
    Array2DWriteView< T > view = writeView();

    m_data = nullptr;
    m_size = { 0, 0 };
    m_stride = { 0, 0 };

    return std::make_pair( view, m_allocator );
}

template< typename T >
void Array2D< T >::invalidate()
{
    if( m_data != nullptr )
    {
        m_allocator->deallocate( m_data, m_size.y * m_stride.y );
        m_data = nullptr;
        m_size = { 0, 0 };
        m_stride = { 0, 0 };
    }
}

template< typename T >
size_t Array2D< T >::width() const
{
    return m_size.x;
}

template< typename T >
size_t Array2D< T >::height() const
{
    return m_size.y;
}

template< typename T >
Vector2i Array2D< T >::size() const
{
    return m_size;
}

template< typename T >
size_t Array2D< T >::numElements() const
{
    return m_size.x * m_size.y;
}

template< typename T >
size_t Array2D< T >::elementStrideBytes() const
{
    return m_stride.x;
}

template< typename T >
size_t Array2D< T >::rowStrideBytes() const
{
    return m_stride.y;
}

template< typename T >
Vector2i Array2D< T >::stride() const
{
    return m_stride;
}

template< typename T >
void Array2D< T >::fill( const T& fillValue )
{
    size_t ne = numElements();
    for( size_t k = 0; k < ne; ++k )
    {
        ( *this )[ k ] = fillValue;
    }
}

template< typename T >
void Array2D< T >::resize( const Vector2i& size )
{
    resize
    (
        size,
        {
            static_cast< int >( sizeof( T ) ),
            static_cast< int >( size.x * sizeof( T ) )
        }
    );
}

template< typename T >
void Array2D< T >::resize( const Vector2i& size, const Vector2i& stride )
{
    // If we request an invalid size then invalidate this.
    if( size.x == 0 || size.y == 0 ||
        stride.x < sizeof( T ) || stride.y < size.x * sizeof( T ) )
    {
        invalidate();
    }
    // Otherwise, it's a valid size.
    else
    {
        // Check if the total amount of memory is different.
        // If so, reallocate. Otherwise, can reuse it and just change the shape.
        if( size.y * stride.y != m_size.y * m_stride.y )
        {
            invalidate();
            m_data = reinterpret_cast< uint8_t* >
            (
                m_allocator->allocate( size.y * stride.y )
            );
        }

        m_size = size;
        m_stride = stride;
    }
}

template< typename T >
const T* Array2D< T >::pointer() const
{
    return reinterpret_cast< T* >( m_data );
}

template< typename T >
T* Array2D< T >::pointer()
{
    return reinterpret_cast< T* >( m_data );
}

template< typename T >
const T* Array2D< T >::elementPointer( const Vector2i& xy ) const
{
    return reinterpret_cast< const T* >(
        &( m_data[ Vector2i::dot( xy, m_stride ) ] ) );
}

template< typename T >
T* Array2D< T >::elementPointer( const Vector2i& xy )
{
    return reinterpret_cast< T* >( &( m_data[ Vector2i::dot( xy, m_stride ) ] ) );
}

template< typename T >
const T* Array2D< T >::rowPointer( size_t y ) const
{
    return elementPointer( { 0, static_cast< int >( y ) } );
}

template< typename T >
T* Array2D< T >::rowPointer( size_t y )
{
    return elementPointer( { 0, static_cast< int >( y ) } );
}

template< typename T >
Array2DReadView< T > Array2D< T >::readView() const
{
    return Array2DReadView< T >( m_data, m_size, m_stride );
}

template< typename T >
Array2DWriteView< T > Array2D< T >::writeView()
{
    return Array2DWriteView< T >( m_data, m_size, m_stride );
}

template< typename T >
Array2D< T >::operator Array2DReadView< T >() const
{
    return Array2DReadView< T >( m_data, m_size, m_stride );
}

template< typename T >
Array2D< T >::operator Array2DWriteView< T >()
{
    return Array2DWriteView< T >( m_data, m_size, m_stride );
}

template< typename T >
Array2D< T >::operator const T* () const
{
    return reinterpret_cast< const T* >( m_data );
}

template< typename T >
Array2D< T >::operator T* ()
{
    return reinterpret_cast< T* >( m_data );
}

template< typename T >
const T& Array2D< T >::operator [] ( size_t k ) const
{
    int x;
    int y;
    Indexing::indexToSubscript2D(
        static_cast< int >( k ), static_cast< int >( m_size.x ), x, y );
    return ( *this )[ { x, y } ];
}

template< typename T >
T& Array2D< T >::operator [] ( size_t k )
{
    int x;
    int y;
    Indexing::indexToSubscript2D(
        static_cast< int >( k ), m_size.x, x, y );
    return ( *this )[ { x, y } ];
}

template< typename T >
const T& Array2D< T >::operator [] ( const Vector2i& xy ) const
{
    return *( elementPointer( xy ) );
}

template< typename T >
T& Array2D< T >::operator [] ( const Vector2i& xy )
{
    return *( elementPointer( xy ) );
}
