template< typename T >
Array3DView< T >::Array3DView( int width, int height, int depth,
	void* pPointer ) :

	m_width( width ),
	m_height( height ),
	m_depth( depth ),
	m_rowPitchBytes( width * sizeof( T ) ),
	m_slicePitchBytes( width * height * sizeof( T ) ),
	m_pPointer( reinterpret_cast< ubyte* >( pPointer ) )

{

}

template< typename T >
Array3DView< T >::Array3DView( int width, int height, int depth,
	int rowPitchBytes, int slicePitchBytes, void* pPointer ) :

	m_width( width ),
	m_height( height ),
	m_depth( depth ),
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
	return rowPointer( y, z )[ x ];
}

template< typename T >
T& Array3DView< T >::operator () ( int x, int y, int z )
{
	return rowPointer( y, z )[ x ];
}

template< typename T >
const T& Array3DView< T >::operator [] ( const Vector3i& xyz ) const
{
	return rowPointer( xyz.y, xyz.z )[ xyz.x ];
}

template< typename T >
T& Array3DView< T >::operator [] ( const Vector3i& xyz )
{
	return rowPointer( xyz.y, xyz.z )[ xyz.x ];
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
int Array3DView< T >::rowPitchBytes() const
{
	return m_rowPitchBytes;
}

template< typename T >
int Array3DView< T >::slicePitchBytes() const
{
	return m_slicePitchBytes;
}
