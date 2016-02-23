#include <algorithm>

template< typename T >
Array1D< T >::Array1D( std::initializer_list< T > values )
{
    resize( values.size() );
    std::copy( values.begin(), values.end(), reinterpret_cast< T* >( m_array.get() ) );
}

template< typename T >
Array1D< T >::Array1D( const char* filename )
{
    load( filename );
}

template< typename T >
Array1D< T >::Array1D( size_t size, size_t stride, const T& fillValue )
{
    resize( size, stride );
    fill( fillValue );
}

template< typename T >
Array1D< T >::Array1D( std::unique_ptr< void > pointer, size_t size, size_t stride ) :
    m_size( size ),
    m_stride( stride ),
    m_array( reinterpret_cast< uint8_t* >( std::move( pointer ).release() ) )
{

}

template< typename T >
Array1D< T >::Array1D( const Array1D< T >& copy )
{
    resize( copy.m_size, copy.m_stride );
    if( copy.notNull() )
    {
        memcpy( m_array, copy.m_array, m_stride * m_size );
    }
}

template< typename T >
Array1D< T >::Array1D( Array1D< T >&& move )
{
    m_size = std::move( move.m_size );
    m_stride = std::move( move.m_strides );
    m_array = std::move( move.m_array );
}

template< typename T >
Array1D< T >& Array1D< T >::operator = ( const Array1D< T >& copy )
{
    if( this != &copy )
    {
        resize( copy.m_size, copy.m_stride );
        if( copy.notNull() )
        {
            memcpy( m_array, copy.m_array, m_stride * m_size );
        }
    }
    return *this;
}

template< typename T >
Array1D< T >& Array1D< T >::operator = ( Array1D< T >&& move )
{
    if( this != &move )
    {
        m_size = std::move( move.m_size );
        m_stride = std::move( move.m_stride );
        m_array = std::move( move.m_array );
    }
    return *this;
}

template< typename T >
bool Array1D< T >::isNull() const
{
    return( m_array == nullptr );
}

template< typename T >
bool Array1D< T >::notNull() const
{
    return( m_array != nullptr );
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
    if( size <= 0 || stride <= 0 )
    {
        m_size = 0;
        m_stride = 0;
        m_array = nullptr;
    }
    // Otherwise, it's a valid size.
    else
    {
        // Check if the total number of memory is the same.
        // If it is, can reuse memory.
        if( size * stride != m_size * m_stride )
        {
            m_array = std::unique_ptr< uint8_t[] >( new uint8_t[ size * stride ] );
        }

        // if the number of elements is the same, the dimensions may be different
        m_size = size;
        m_stride = stride;
    }
}

template< typename T >
const T* Array1D< T >::pointer() const
{
    return reinterpret_cast< T* >( m_array.get() );
}

template< typename T >
T* Array1D< T >::pointer()
{
    return reinterpret_cast< T* >( m_array.get() );
}

template< typename T >
const T* Array1D< T >::elementPointer( int x ) const
{
    return reinterpret_cast< const T* >( &( m_array[ x * m_stride ] ) );
}

template< typename T >
T* Array1D< T >::elementPointer( int x )
{
    return reinterpret_cast< T* >( &( m_array[ x * m_stride ] ) );
}

template< typename T >
Array1D< T >::operator Array1DView< const T >() const
{
    return Array1DView< const T >( m_array.get(), m_size, m_stride );
}

template< typename T >
Array1D< T >::operator Array1DView< T >()
{
    return Array1DView< T >( m_array.get(), m_size, m_stride );
}

template< typename T >
Array1D< T >::operator const T* () const
{
    return reinterpret_cast< const T* >( m_array.get() );
}

template< typename T >
Array1D< T >::operator T* ()
{
    return reinterpret_cast< T* >( m_array.get() );
}

template< typename T >
const T& Array1D< T >::operator [] ( int k ) const
{
    return *( elementPointer( k ) );
}

template< typename T >
T& Array1D< T >::operator [] ( int k )
{
    return *( elementPointer( k ) );
}

template< typename T >
bool Array1D< T >::load( const char* filename )
{
    FILE* fp = fopen( filename, "rb" );
    if( fp == nullptr )
    {
        return false;
    }

    bool succeeded = load( fp );

    // close file
    int fcloseRetVal = fclose( fp );
    if( fcloseRetVal != 0 )
    {
        return false;
    }

    return succeeded;
}

template< typename T >
bool Array1D< T >::load( FILE* fp )
{
    size_t width;
    size_t stride;
    size_t elementsRead;

    elementsRead = fread( &width, sizeof( size_t ), 1, fp );
    if( elementsRead != 1 )
    {
        return false;
    }

    elementsRead = fread( &stride, sizeof( size_t ), 1, fp );
    if( elementsRead != 1 )
    {
        return false;
    }

    size_t nBytes = stride * width;
    std::unique_ptr< uint8_t[] > pBuffer( new uint8_t[ nBytes ] );

    // read elements
    elementsRead = fread( pBuffer.get(), 1, nBytes, fp );
    if( elementsRead != nBytes )
    {
        return false;
    }

    // read succeeded, swap contents
    m_size = width;
    m_stride = stride;
    m_array = pBuffer;

    return true;
}

template< typename T >
bool Array1D< T >::save( const char* filename ) const
{
    FILE* fp = fopen( filename, "wb" );
    if( fp == nullptr )
    {
        return false;
    }

    bool succeeded = save( fp );
    fclose( fp );
    return succeeded;
}

template< typename T >
bool Array1D< T >::save( FILE* fp ) const
{
    // TODO: error checking

    fwrite( &m_size, sizeof( size_t ), 1, fp );
    fwrite( &m_stride, sizeof( size_t ), 1, fp );
    fwrite( m_array.get(), 1, m_size * m_stride, fp );
    fclose( fp );

    return true;
}
