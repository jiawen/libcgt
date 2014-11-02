template< typename T >
Array3DView< T >::Array3DView( void* pPointer,
	int width, int height, int depth ) :

	m_width( width ),
	m_height( height ),
	m_depth( depth ),
	m_strideBytes( sizeof( T ) ),
	m_rowStrideBytes( width * sizeof( T ) ),
	m_sliceStrideBytes( width * height * sizeof( T ) ),
	m_pPointer( reinterpret_cast< uint8_t* >( pPointer ) )

{

}

template< typename T >
Array3DView< T >::Array3DView( void* pPointer, const Vector3i& size ) :

	m_width( size.x ),
	m_height( size.y ),
	m_depth( size.z ),
	m_strideBytes( sizeof( T ) ),
	m_rowStrideBytes( size.x * sizeof( T ) ),
	m_sliceStrideBytes( size.x * size.y * sizeof( T ) ),
	m_pPointer( reinterpret_cast< uint8_t* >( pPointer ) )

{

}

template< typename T >
Array3DView< T >::Array3DView( void* pPointer,
	int width, int height, int depth,
	int strideBytes, int rowStrideBytes, int sliceStrideBytes ) :

	m_width( width ),
	m_height( height ),
	m_depth( depth ),
	m_strideBytes( strideBytes ),
	m_rowStrideBytes( rowStrideBytes ),
	m_sliceStrideBytes( sliceStrideBytes ),
	m_pPointer( reinterpret_cast< uint8_t* >( pPointer ) )

{

}

template< typename T >
Array3DView< T >::Array3DView( void* pPointer,
	const Vector3i& size,
	int strideBytes, int rowStrideBytes, int sliceStrideBytes ) :

	m_width( size.x ),
	m_height( size.y ),
	m_depth( size.z ),
	m_strideBytes( strideBytes ),
	m_rowStrideBytes( rowStrideBytes ),
	m_sliceStrideBytes( sliceStrideBytes ),
	m_pPointer( reinterpret_cast< uint8_t* >( pPointer ) )

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
const T* Array3DView< T >::elementPointer( int x, int y, int z ) const
{
	return reinterpret_cast< T* >( m_pPointer + z * sliceStrideBytes() + y * rowStrideBytes() + x * elementStrideBytes() );
}

template< typename T >
T* Array3DView< T >::elementPointer( int x, int y, int z )
{
	return reinterpret_cast< T* >( m_pPointer + z * sliceStrideBytes() + y * rowStrideBytes() + x * elementStrideBytes() );
}

template< typename T >
const T* Array3DView< T >::rowPointer( int y, int z ) const
{
	return elementPointer( 0, y, z );
}

template< typename T >
T* Array3DView< T >::rowPointer( int y, int z )
{
	return elementPointer( 0, y, z );
}

template< typename T >
const T* Array3DView< T >::slicePointer( int z ) const
{
	return elementPointer( 0, 0, z );
}

template< typename T >
T* Array3DView< T >::slicePointer( int z )
{
	return elementPointer( 0, 0, z );
}

template< typename T >
const T& Array3DView< T >::operator [] ( int k ) const
{
	int x;
	int y;
	int z;
	Indexing::indexToSubscript3D( k, m_width, m_height, x, y, z );
	return ( *this )( x, y, z );
}

template< typename T >
T& Array3DView< T >::operator [] ( int k )
{
	int x;
	int y;
	int z;
	Indexing::indexToSubscript3D( k, m_width, m_height, x, y, z );
	return ( *this )( x, y, z );
}

template< typename T >
const T& Array3DView< T >::operator () ( int x, int y, int z ) const
{
	return *elementPointer( x, y, z );
}

template< typename T >
T& Array3DView< T >::operator () ( int x, int y, int z )
{
	return *elementPointer( x, y, z );
}

template< typename T >
const T& Array3DView< T >::operator [] ( const Vector3i& xyz ) const
{
	return ( *this )( xyz.x, xyz.y, xyz.z );
}

template< typename T >
T& Array3DView< T >::operator [] ( const Vector3i& xyz )
{
	return ( *this )( xyz.x, xyz.y, xyz.z );
}

template< typename T >
int Array3DView< T >::width() const
{
	return m_width;
}

template< typename T >
int Array3DView< T >::height() const
{
	return m_height;
}

template< typename T >
int Array3DView< T >::depth() const
{
	return m_depth;
}

template< typename T >
Vector3i Array3DView< T >::size() const
{
	return Vector3i( m_width, m_height, m_depth );
}

template< typename T >
int Array3DView< T >::numElements() const
{
	return width() * height() * depth();
}

template< typename T >
size_t Array3DView< T >::bytesReferenced() const
{
	return sizeof( T ) * numElements();
}

template< typename T >
size_t Array3DView< T >::bytesSpanned() const
{
	return std::abs( sliceStrideBytes() ) * depth();
}

template< typename T >
int Array3DView< T >::elementStrideBytes() const
{
	return m_elementStrideBytes;
}

template< typename T >
int Array3DView< T >::rowStrideBytes() const
{
	return m_rowStrideBytes;
}

template< typename T >
int Array3DView< T >::sliceStrideBytes() const
{
	return m_sliceStrideBytes;
}

template< typename T >
bool Array3DView< T >::elementsArePacked() const
{
	return m_strideBytes == sizeof( T );
}

template< typename T >
bool Array3DView< T >::rowsArePacked() const
{
	return m_rowStrideBytes == ( m_width * m_strideBytes );
}

template< typename T >
bool Array3DView< T >::slicesArePacked() const
{
	return m_sliceStrideBytes == ( m_height * m_rowStrideBytes );
}

template< typename T >
bool Array3DView< T >::packed() const
{
	return elementsArePacked() && rowsArePacked() && slicesArePacked();
}
