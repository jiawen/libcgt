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
    return elementPointer( { 0, y } );
}

template< typename T >
const T& Array2DReadView< T >::operator [] ( int k ) const
{
    Vector2i xy = libcgt::core::indexToSubscript2D( k, m_size );
    return ( *this )[ xy ];
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
    return elementPointer( { 0, y } );
}

template< typename T >
T& Array2DWriteView< T >::operator [] ( int k ) const
{
    Vector2i xy = libcgt::core::indexToSubscript2D( k, this->size() );
    return ( *this )[ xy ];
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

template< typename T >
Array3DReadView< T >::Array3DReadView( const void* pointer,
    const Vector3i& size ) :
    m_size( size ),
    m_stride
    (
        static_cast< int >( sizeof( T ) ),
        static_cast< int >( size.x * sizeof( T ) ),
        static_cast< int >( size.x * size.y * sizeof( T ) )
    ),
    m_pointer( reinterpret_cast< const uint8_t* >( pointer ) )
{

}

template< typename T >
Array3DReadView< T >::Array3DReadView( const void* pointer,
    const Vector3i& size, const Vector3i& stride ) :
    m_size( size ),
    m_stride( stride ),
    m_pointer( reinterpret_cast< const uint8_t* >( pointer ) )
{

}

template< typename T >
bool Array3DReadView< T >::isNull() const
{
    return( m_pointer == nullptr );
}

template< typename T >
bool Array3DReadView< T >::notNull() const
{
    return( m_pointer != nullptr );
}

template< typename T >
void Array3DReadView< T >::setNull()
{
    m_size = { 0, 0 };
    m_stride = { 0, 0 };
    m_pointer = nullptr;
}

template< typename T >
Array3DReadView< T >::operator const T* ( ) const
{
    return reinterpret_cast< const T* >( m_pointer );
}

template< typename T >
const T* Array3DReadView< T >::pointer() const
{
    return reinterpret_cast< const T* >( m_pointer );
}

template< typename T >
const T* Array3DReadView< T >::elementPointer( const Vector3i& xyz ) const
{
    return reinterpret_cast< const T* >(
        &( m_pointer[ Vector3i::dot( xyz, m_stride ) ] ) );
}

template< typename T >
const T* Array3DReadView< T >::rowPointer( const Vector2i& yz ) const
{
    return elementPointer( { 0, yz } );
}

template< typename T >
const T* Array3DReadView< T >::slicePointer( int z ) const
{
    return elementPointer( { 0, 0, z } );
}

template< typename T >
const T& Array3DReadView< T >::operator [] ( int k ) const
{
    Vector3i xy = libcgt::core::indexToSubscript3D( k, m_size );
    return ( *this )[ xy ];
}

template< typename T >
const T& Array3DReadView< T >::operator [] ( const Vector3i& xy ) const
{
    return *( elementPointer( xy ) );
}

template< typename T >
int Array3DReadView< T >::width() const
{
    return m_size.x;
}

template< typename T >
int Array3DReadView< T >::height() const
{
    return m_size.y;
}

template< typename T >
int Array3DReadView< T >::depth() const
{
    return m_size.z;
}

template< typename T >
Vector3i Array3DReadView< T >::size() const
{
    return m_size;
}

template< typename T >
int Array3DReadView< T >::numElements() const
{
    return m_size.x * m_size.y * m_size.z;
}

template< typename T >
int Array3DReadView< T >::elementStrideBytes() const
{
    return m_stride.x;
}

template< typename T >
int Array3DReadView< T >::rowStrideBytes() const
{
    return m_stride.y;
}

template< typename T >
int Array3DReadView< T >::sliceStrideBytes() const
{
    return m_stride.z;
}

template< typename T >
Vector3i Array3DReadView< T >::stride() const
{
    return m_stride;
}

template< typename T >
bool Array3DReadView< T >::elementsArePacked() const
{
    return m_stride.x == sizeof( T );
}

template< typename T >
bool Array3DReadView< T >::rowsArePacked() const
{
    return m_stride.y == ( m_size.x * m_stride.x );
}

template< typename T >
bool Array3DReadView< T >::slicesArePacked() const
{
    return m_stride.z == ( m_size.y * m_stride.y );
}

template< typename T >
bool Array3DReadView< T >::packed() const
{
    return elementsArePacked() && rowsArePacked() && slicesArePacked();
}

template< typename T >
Array1DReadView< T > Array3DReadView< T >::xySlice( const Vector2i& xy )
{
    return Array1DReadView< T >(
        elementPointer( { xy.x, xy.y, 0 } ), m_size.z );
}

template< typename T >
Array1DReadView< T > Array3DReadView< T >::yzSlice( const Vector2i& yz )
{
    return Array1DReadView< T >(
        elementPointer( { 0, yz.x, yz.y } ), m_size.x );
}

template< typename T >
Array1DReadView< T > Array3DReadView< T >::xzSlice( const Vector2i& xz )
{
    return Array1DReadView< T >(
        elementPointer( { xz.x, 0, xz.y } ), m_size.y );
}

template< typename T >
Array2DReadView< T > Array3DReadView< T >::xSlice( int x )
{
    return Array2DReadView< T >(
        elementPointer( { x, 0, 0 } ), m_size.yz, m_stride.yz );
}

template< typename T >
Array2DReadView< T > Array3DReadView< T >::ySlice( int y )
{
    return Array2DReadView< T >(
        rowPointer( { y, 0 } ), m_size.xz(), m_stride.xz() );
}

template< typename T >
Array2DReadView< T > Array3DReadView< T >::zSlice( int z )
{
    return Array2DReadView< T >(
        slicePointer( z ), m_size.xy, m_stride.xy );
}

template< typename T >
Array3DWriteView< T >::Array3DWriteView( void* pointer,
    const Vector3i& size ) :
    Array3DReadView< T >( pointer, size ),
    m_write_pointer( reinterpret_cast< uint8_t* >( pointer ) )
{

}

template< typename T >
Array3DWriteView< T >::Array3DWriteView( void* pointer,
    const Vector3i& size, const Vector3i& stride ) :
    Array3DReadView< T >( pointer, size, stride ),
    m_write_pointer( reinterpret_cast< uint8_t* >( pointer ) )
{

}

template< typename T >
Array3DWriteView< T >::operator T* ( ) const
{
    return reinterpret_cast< T* >( m_write_pointer );
}

template< typename T >
T* Array3DWriteView< T >::pointer() const
{
    return reinterpret_cast< T* >( m_write_pointer );
}

template< typename T >
T* Array3DWriteView< T >::elementPointer( const Vector3i& xy ) const
{
    return reinterpret_cast< T* >(
        &( m_write_pointer[ Vector3i::dot( xy, this->stride() ) ] ) );
}

template< typename T >
T* Array3DWriteView< T >::rowPointer( const Vector2i& yz ) const
{
    return elementPointer( { 0, yz } );
}

template< typename T >
T* Array3DWriteView< T >::slicePointer( int z ) const
{
    return elementPointer( { 0, 0, z } );
}

template< typename T >
T& Array3DWriteView< T >::operator [] ( int k ) const
{
    Vector3i xy = libcgt::core::indexToSubscript3D( k, this->size() );
    return ( *this )[ xy ];
}

template< typename T >
T& Array3DWriteView< T >::operator [] ( const Vector3i& xy ) const
{
    return *( elementPointer( xy ) );
}

template< typename T >
Array1DWriteView< T > Array3DWriteView< T >::xySlice( const Vector2i& xy )
{
    return Array1DWriteView< T >(
        elementPointer( { xy.x, xy.y, 0 } ), m_size.z );
}

template< typename T >
Array1DWriteView< T > Array3DWriteView< T >::yzSlice( const Vector2i& yz )
{
    return Array1DWriteView< T >(
        elementPointer( { 0, yz.x, yz.y } ), m_size.x );
}

template< typename T >
Array1DWriteView< T > Array3DWriteView< T >::xzSlice( const Vector2i& xz )
{
    return Array1DWriteView< T >(
        elementPointer( { xz.x, 0, xz.y } ), m_size.y );
}

template< typename T >
Array2DWriteView< T > Array3DWriteView< T >::xSlice( int x )
{
    return Array2DWriteView< T >(
        elementPointer( { x, 0, 0 } ), m_size.yz, m_stride.yz );
}

template< typename T >
Array2DWriteView< T > Array3DWriteView< T >::ySlice( int y )
{
    return Array2DWriteView< T >(
        rowPointer( { y, 0 } ), m_size.xz(), m_stride.xz() );
}

template< typename T >
Array2DWriteView< T > Array3DWriteView< T >::zSlice( int z )
{
    return Array2DWriteView< T >(
        slicePointer( z ), m_size.xy, m_stride.xy );
}
