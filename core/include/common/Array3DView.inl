template< typename T >
Array3DView< T >::Array3DView( void* pPointer,
	int width, int height, int depth ) :

	m_width( width ),
	m_height( height ),
	m_depth( depth ),
	m_strideBytes( sizeof( T ) ),
	m_rowPitchBytes( width * sizeof( T ) ),
	m_slicePitchBytes( width * height * sizeof( T ) ),
	m_pPointer( reinterpret_cast< ubyte* >( pPointer ) )

{

}

template< typename T >
Array3DView< T >::Array3DView( void* pPointer, const Vector3i& size ) :

	m_width( size.x ),
	m_height( size.y ),
	m_depth( size.z ),
	m_strideBytes( sizeof( T ) ),
	m_rowPitchBytes( size.x * sizeof( T ) ),
	m_slicePitchBytes( size.x * size.y * sizeof( T ) ),
	m_pPointer( reinterpret_cast< ubyte* >( pPointer ) )

{

}

template< typename T >
Array3DView< T >::Array3DView( void* pPointer,
	int width, int height, int depth,
	int strideBytes, int rowPitchBytes, int slicePitchBytes ) :

	m_width( width ),
	m_height( height ),
	m_depth( depth ),
	m_strideBytes( strideBytes ),
	m_rowPitchBytes( rowPitchBytes ),
	m_slicePitchBytes( slicePitchBytes ),
	m_pPointer( reinterpret_cast< ubyte* >( pPointer ) )

{

}

template< typename T >
Array3DView< T >::Array3DView( void* pPointer,
	const Vector3i& size,
	int strideBytes, int rowPitchBytes, int slicePitchBytes ) :

	m_width( size.x ),
	m_height( size.y ),
	m_depth( size.z ),
	m_strideBytes( strideBytes ),
	m_rowPitchBytes( rowPitchBytes ),
	m_slicePitchBytes( slicePitchBytes ),
	m_pPointer( reinterpret_cast< ubyte* >( pPointer ) )

{

}

template< typename T >
const T* Array3DView< T >::rowPointer( int y, int z ) const
{
	return reinterpret_cast< T* >
	(
		&( m_pPointer[ z * m_slicePitchBytes + y * m_rowPitchBytes ] )
	);
}

template< typename T >
T* Array3DView< T >::rowPointer( int y, int z )
{
	return reinterpret_cast< T* >
	(
		&( m_pPointer[ z * m_slicePitchBytes + y * m_rowPitchBytes ] )
	);
}

template< typename T >
const T* Array3DView< T >::slicePointer( int z ) const
{
	return reinterpret_cast< T* >
	(
		&( m_pPointer[ z * m_slicePitchBytes ] )
	);
}

template< typename T >
T* Array3DView< T >::slicePointer( int z )
{
	return reinterpret_cast< T* >
	(
		&( m_pPointer[ z * m_slicePitchBytes ] )
	);
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
	const ubyte* pRowPointer = reinterpret_cast< const ubyte* >( rowPointer( y, z ) );
	const ubyte* pElementPointer = pRowPointer + x * m_strideBytes;
	const T* q = reinterpret_cast< const T* >( pElementPointer );
	return *q;
}

template< typename T >
T& Array3DView< T >::operator () ( int x, int y, int z )
{
	ubyte* pRowPointer = reinterpret_cast< ubyte* >( rowPointer( y, z ) );
	ubyte* pElementPointer = pRowPointer + x * m_strideBytes;
	T* q = reinterpret_cast< T* >( pElementPointer );
	return *q;
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
int Array3DView< T >::strideBytes() const
{
	return m_strideBytes;
}

template< typename T >
int Array3DView< T >::rowPitchBytes() const
{
	return m_rowPitchBytes;
}

template< typename T >
int Array3DView< T >::slicePitchBytes() const
{
	return m_slicePitchBytes;
}

template< typename T >
bool Array3DView< T >::elementsArePacked() const
{
	return m_strideBytes == sizeof( T );
}

template< typename T >
bool Array3DView< T >::rowsArePacked() const
{
	return m_rowPitchBytes == ( m_width * m_strideBytes );
}

template< typename T >
bool Array3DView< T >::slicesArePacked() const
{
	return m_slicePitchBytes == ( m_height * m_rowPitchBytes );
}

template< typename T >
bool Array3DView< T >::packed() const
{
	return elementsArePacked() && rowsArePacked() && slicesArePacked();
}
