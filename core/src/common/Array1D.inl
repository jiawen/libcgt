#include <algorithm>
#include <cassert>

template< typename T >
Array1D< T >::Array1D( std::initializer_list< T > values, Allocator* allocator ) :
    m_allocator( allocator )
{
    resize( values.size() );
    std::copy( values.begin(), values.end(), reinterpret_cast< T* >( m_data ) );
}

template< typename T >
Array1D< T >::Array1D( size_t size, size_t stride, const T& fillValue,
    Allocator* allocator ) :
    m_allocator( allocator )
{
    assert( stride >= sizeof( T ) );
    resize( size, stride );
    fill( fillValue );
}

template< typename T >
Array1D< T >::Array1D( void* pointer, size_t size, size_t stride, Allocator* allocator ) :
    m_data( pointer ),
    m_size( size ),
    m_stride( stride ),
    m_allocator( allocator )
{

}

template< typename T >
Array1D< T >::Array1D( const Array1D< T >& copy )
{
    resize( copy.m_size, copy.m_stride );
    if( copy.notNull() )
    {
        memcpy( m_data, copy.m_data, m_stride * m_size );
    }
}

template< typename T >
Array1D< T >::Array1D( Array1D< T >&& move )
{
    invalidate();
    m_size = std::move( move.m_size );
    m_stride = std::move( move.m_stride );
    m_data = std::move( move.m_data );
    m_allocator = std::move( move.m_allocator );
}

template< typename T >
Array1D< T >& Array1D< T >::operator = ( const Array1D< T >& copy )
{
    if( this != &copy )
    {
        resize( copy.m_size, copy.m_stride );
        if( copy.notNull() )
        {
            memcpy( m_data, copy.m_data, m_stride * m_size );
        }
    }
    return *this;
}

template< typename T >
Array1D< T >& Array1D< T >::operator = ( Array1D< T >&& move )
{
    if( this != &move )
    {
        invalidate();
        m_size = std::move( move.m_size );
        m_stride = std::move( move.m_stride );
        m_data = std::move( move.m_data );
        m_allocator = std::move( move.m_allocator );
    }
    return *this;
}

// virtual
template< typename T >
Array1D< T >::~Array1D()
{
    invalidate();
}

template< typename T >
bool Array1D< T >::isNull() const
{
    return( m_data == nullptr );
}

template< typename T >
bool Array1D< T >::notNull() const
{
    return( m_data != nullptr );
}

template< typename T >
std::pair< Array1DView< T >, Allocator* > Array1D< T >::relinquish()
{
    Array1DView< T > view = readWriteView();

    m_data = nullptr;
    m_size = 0;
    m_stride = 0;

    return std::make_pair( view, m_allocator );
}

template< typename T >
void Array1D< T >::invalidate()
{
    if( m_data != nullptr )
    {
        m_allocator->deallocate( m_data, m_size * m_stride );
        m_data = nullptr;
        m_size = 0;
        m_stride = 0;
    }
}

template< typename T >
size_t Array1D< T >::width() const
{
    return m_size;
}

template< typename T >
size_t Array1D< T >::size() const
{
    return m_size;
}

template< typename T >
size_t Array1D< T >::numElements() const
{
    return m_size;
}

template< typename T >
size_t Array1D< T >::elementStrideBytes() const
{
    return m_stride;
}

template< typename T >
size_t Array1D< T >::stride() const
{
    return m_stride;
}

template< typename T >
void Array1D< T >::fill( const T& fillValue )
{
    size_t ne = size();
    for( size_t k = 0; k < ne; ++k )
    {
        ( *this )[ k ] = fillValue;
    }
}

template< typename T >
void Array1D< T >::resize( size_t size )
{
    resize( size, sizeof( T ) );
}

template< typename T >
void Array1D< T >::resize( size_t size, size_t stride )
{
    // If we request an invalid size then invalidate this.
    if( size == 0 || stride == 0 || stride < sizeof( T ) )
    {
        invalidate();
    }
    // Otherwise, it's a valid size.
    else
    {
        // Check if the total amount of memory is different.
        // If so, reallocate. Otherwise, can reuse it and just change the shape.
        if( size * stride != m_size * m_stride )
        {
            invalidate();
            m_data = reinterpret_cast< uint8_t* >( m_allocator->allocate( size * stride ) );
        }

        m_size = size;
        m_stride = stride;
    }
}

template< typename T >
const T* Array1D< T >::pointer() const
{
    return reinterpret_cast< T* >( m_data );
}

template< typename T >
T* Array1D< T >::pointer()
{
    return reinterpret_cast< T* >( m_data );
}

template< typename T >
const T* Array1D< T >::elementPointer( size_t x ) const
{
    return reinterpret_cast< const T* >( &( m_data[ x * m_stride ] ) );
}

template< typename T >
T* Array1D< T >::elementPointer( size_t x )
{
    return reinterpret_cast< T* >( &( m_data[ x * m_stride ] ) );
}

template< typename T >
Array1DView< const T > Array1D< T >::readOnlyView() const
{
    return Array1DView< const T >( m_data, m_size, m_stride );
}

template< typename T >
Array1DView< T > Array1D< T >::readWriteView() const
{
    return Array1DView< T >( m_data, m_size, m_stride );
}

template< typename T >
Array1D< T >::operator Array1DView< const T >() const
{
    return readOnlyView();
}

template< typename T >
Array1D< T >::operator Array1DView< T >()
{
    return readWriteView();
}

template< typename T >
Array1D< T >::operator const T* () const
{
    return reinterpret_cast< const T* >( m_data );
}

template< typename T >
Array1D< T >::operator T* ()
{
    return reinterpret_cast< T* >( m_data );
}

template< typename T >
const T& Array1D< T >::operator [] ( size_t k ) const
{
    return *( elementPointer( k ) );
}

template< typename T >
T& Array1D< T >::operator [] ( size_t k )
{
    return *( elementPointer( k ) );
}
