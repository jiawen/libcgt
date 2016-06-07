template< typename T >
Array3D< T >::Array3D( void* pointer, const Vector3i& size,
    Allocator* allocator ) :
    Array3D
    (
        pointer,
        size,
        {
            static_cast< int >( sizeof( T ) ),
            static_cast< int >( size.x * sizeof( T ) ),
            static_cast< int >( size.x * size.y * sizeof( T ) )
        },
        allocator
    )
{

}

template< typename T >
Array3D< T >::Array3D( void* pointer, const Vector3i& size,
    const Vector3i& stride, Allocator* allocator ) :
    m_size( size ),
    m_stride( stride ),
    m_data( reinterpret_cast< uint8_t* >( pointer ) ),
    m_allocator( allocator )
{

}

template< typename T >
Array3D< T >::Array3D( const Vector3i& size, const T& fillValue,
    Allocator* allocator ) :
    Array3D
    (
        size,
        {
            static_cast< int >( sizeof( T ) ),
            static_cast< int >( size.x * sizeof( T ) ),
            static_cast< int >( size.x * size.y * sizeof( T ) )
        },
        fillValue,
        allocator
    )
{
}

template< typename T >
Array3D< T >::Array3D( const Vector3i& size, const Vector3i& stride,
    const T& fillValue, Allocator* allocator ) :
    m_allocator( allocator )
{
    assert( size.x >= 0 );
    assert( size.y >= 0 );
    assert( size.z >= 0 );
    assert( stride.x >= sizeof( T ) );
    assert( stride.y >= size.x * sizeof( T ) );
    assert( stride.z >= size.x * size.y * sizeof( T ) );
    resize( size, stride );
    fill( fillValue );
}

template< typename T >
Array3D< T >::Array3D( const Array3D< T >& copy )
{
    resize( copy.m_size, copy.m_stride );
    if( copy.notNull( ) )
    {
        memcpy( m_data, copy.m_data, m_stride.z * m_size.z );
    }
}

template< typename T >
Array3D< T >::Array3D( Array3D< T >&& move )
{
    invalidate();
    m_size = std::move( move.m_size );
    m_stride = std::move( move.m_stride );
    m_data = std::move( move.m_data );
    m_allocator = std::move( move.m_allocator );

    move.m_size = { 0, 0, 0 };
    move.m_stride = { 0, 0, 0 };
    move.m_data = nullptr;
}

template< typename T >
Array3D< T >& Array3D< T >::operator = ( const Array3D< T >& copy )
{
    if( this != &copy )
    {
        resize( copy.m_size, copy.m_stride );
        if( copy.notNull() )
        {
            memcpy( m_data, copy.m_data, m_stride.z * m_size.z );
        }
    }
    return *this;
}

template< typename T >
Array3D< T >& Array3D< T >::operator = ( Array3D< T >&& move )
{
    if( this != &move )
    {
        invalidate();
        m_size = std::move( move.m_size );
        m_stride = std::move( move.m_stride );
        m_data = std::move( move.m_data );
        m_allocator = std::move( move.m_allocator );

        move.m_size = { 0, 0, 0 };
        move.m_stride = { 0, 0, 0 };
        move.m_data = nullptr;
    }
    return *this;
}

// virtual
template< typename T >
Array3D< T >::~Array3D()
{
    invalidate();
}

template< typename T >
bool Array3D< T >::isNull() const
{
    return( m_data == nullptr );
}

template< typename T >
bool Array3D< T >::notNull() const
{
    return( m_data != nullptr );
}

template< typename T >
std::pair< Array3DView< T >, Allocator* > Array3D< T >::relinquish()
{
    Array3DView< T > view = writeView();

    m_data = nullptr;
    m_size = { 0, 0, 0 };
    m_stride = { 0, 0, 0 };

    return std::make_pair( view, m_allocator );
}

template< typename T >
void Array3D< T >::invalidate()
{
    if( m_data != nullptr )
    {
        m_allocator->deallocate( m_data, m_size.z * m_stride.z );
        m_data = nullptr;
        m_size = Vector3i{ 0 };
        m_stride = Vector3i{ 0 };
    }
}

template< typename T >
int Array3D< T >::width() const
{
    return m_size.x;
}

template< typename T >
int Array3D< T >::height() const
{
    return m_size.y;
}

template< typename T >
int Array3D< T >::depth() const
{
    return m_size.z;
}

template< typename T >
Vector3i Array3D< T >::size() const
{
    return m_size;
}

template< typename T >
int Array3D< T >::numElements() const
{
    return m_size.x * m_size.y * m_size.z;
}

template< typename T >
int Array3D< T >::elementStrideBytes() const
{
    return m_stride.x;
}

template< typename T >
int Array3D< T >::rowStrideBytes() const
{
    return m_stride.y;
}

template< typename T >
int Array3D< T >::sliceStrideBytes() const
{
    return m_stride.z;
}

template< typename T >
Vector3i Array3D< T >::stride() const
{
    return m_stride;
}

template< typename T >
void Array3D< T >::fill( const T& fillValue )
{
    int ne = numElements();
    for( int k = 0; k < ne; ++k )
    {
        ( *this )[ k ] = fillValue;
    }
}

template< typename T >
void Array3D< T >::resize( const Vector3i& size )
{
    resize
    (
        size,
        {
            static_cast< int >( sizeof( T ) ),
            static_cast< int >( size.x * sizeof( T ) ),
            static_cast< int >( size.x * size.y * sizeof( T ) )
        }
    );
}

template< typename T >
void Array3D< T >::resize( const Vector3i& size, const Vector3i& stride )
{
    // If we request an invalid size then invalidate this.
    if( size.x == 0 || size.y == 0 || size.z == 0 ||
        stride.x < sizeof( T ) ||
        stride.y < size.x * sizeof( T ) ||
        stride.z < size.x * size.y * sizeof( T ) )
    {
        invalidate();
    }
    // Otherwise, it's a valid size.
    else
    {
        // Check if the total amount of memory is different.
        // If so, reallocate. Otherwise, can reuse it and just change the shape.
        if( size.z * stride.z != m_size.z * m_stride.z )
        {
            invalidate();
            m_data = reinterpret_cast< uint8_t* >
            (
                m_allocator->allocate( size.z * stride.z )
            );
        }

        m_size = size;
        m_stride = stride;
    }
}

template< typename T >
const T* Array3D< T >::pointer( ) const
{
    return reinterpret_cast< T* >( m_data );
}

template< typename T >
T* Array3D< T >::pointer( )
{
    return reinterpret_cast< T* >( m_data );
}

template< typename T >
const T* Array3D< T >::elementPointer( const Vector3i& xyz ) const
{
    return reinterpret_cast< const T* >( &( m_data[ Vector3i::dot( xyz, m_stride ) ] ) );
}

template< typename T >
T* Array3D< T >::elementPointer( const Vector3i& xyz )
{
    return reinterpret_cast< T* >( &( m_data[ Vector3i::dot( xyz, m_stride ) ] ) );
}

template< typename T >
const T* Array3D< T >::rowPointer( int y, int z ) const
{
    return elementPointer( { 0, y, z } );
}

template< typename T >
T* Array3D< T >::rowPointer( int y, int z )
{
    return elementPointer( { 0, y, z } );
}

template< typename T >
const T* Array3D< T >::slicePointer( int z ) const
{
    return elementPointer( { 0, 0, z } );
}

template< typename T >
T* Array3D< T >::slicePointer( int z )
{
    return elementPointer( { 0, 0, z } );
}

template< typename T >
Array3DView< const T > Array3D< T >::readView() const
{
    return Array3DView< const T >( m_data, m_size, m_stride );
}

template< typename T >
Array3DView< T > Array3D< T >::writeView() const
{
    return Array3DView< T >( m_data, m_size, m_stride );
}

template< typename T >
Array3D< T >::operator const Array3DView< const T >() const
{
    return readView();
}

template< typename T >
Array3D< T >::operator Array3DView< T >()
{
    return writeView();
}

template< typename T >
Array3D< T >::operator const T* () const
{
    return reinterpret_cast< T* >( m_data );
}

template< typename T >
Array3D< T >::operator T* ()
{
    return reinterpret_cast< T* >( m_data );
}

template< typename T >
const T& Array3D< T >::operator [] ( int k ) const
{
    int x;
    int y;
    int z;
    Indexing::indexToSubscript3D(
        static_cast< int >( k ), m_size.x, m_size.y, x, y, z );
    return ( *this )[ { x, y, z } ];
}

template< typename T >
T& Array3D< T >::operator [] ( int k )
{
    int x;
    int y;
    int z;
    Indexing::indexToSubscript3D(
        static_cast< int >( k ), m_size.x, m_size.y, x, y, z );
    return ( *this )[ { x, y, z } ];
}

template< typename T >
const T& Array3D< T >::operator [] ( const Vector3i& xyz ) const
{
    return *( elementPointer( xyz ) );
}

template< typename T >
T& Array3D< T >::operator [] ( const Vector3i& xyz )
{
    return *( elementPointer( xyz ) );
}
