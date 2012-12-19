template< typename T >
Array2DView< T >::Array2DView( int width, int height, void* pPointer ) :

	m_width( width ),
	m_height( height ),
	m_rowPitchBytes( width * sizeof( T ) ),
	m_pPointer( reinterpret_cast< ubyte* >( pPointer ) )

{

}

template< typename T >
Array2DView< T >::Array2DView( int width, int height, int rowPitchBytes, void* pPointer ) :

	m_width( width ),
	m_height( height ),
	m_rowPitchBytes( rowPitchBytes ),
	m_pPointer( reinterpret_cast< ubyte* >( pPointer ) )

{

}

template< typename T >
T* Array2DView< T >::rowPointer( int y )
{
	return reinterpret_cast< T* >( &( m_pPointer[ y * m_rowPitchBytes ] ) );
}

template< typename T >
const T* Array2DView< T >::rowPointer( int y ) const
{
	return reinterpret_cast< T* >( &( m_pPointer[ y * m_rowPitchBytes ] ) );
}

template< typename T >
const T& Array2DView< T >::operator [] ( int k ) const
{
	int x;
	int y;
	Indexing::indexToSubscript2D( k, m_width, x, y );
	return ( *this )( x, y );
}

template< typename T >
T& Array2DView< T >::operator [] ( int k )
{
	int x;
	int y;
	Indexing::indexToSubscript2D( k, m_width, x, y );
	return ( *this )( x, y );
}

template< typename T >
const T& Array2DView< T >::operator () ( int x, int y ) const
{
	return rowPointer( y )[ x ];
}

template< typename T >
T& Array2DView< T >::operator () ( int x, int y )
{
	return rowPointer( y )[ x ];
}

template< typename T >
const T& Array2DView< T >::operator () ( const Vector2i& xy ) const
{
	return rowPointer( xy.y )[ xy.x ];
}

template< typename T >
T& Array2DView< T >::operator () ( const Vector2i& xy )
{
	return rowPointer( xy.y )[ xy.x ];
}

template< typename T >
int Array2DView< T >::width() const
{
	return m_width;
}

template< typename T >
int Array2DView< T >::height() const
{
	return m_height;
}

template< typename T >
int Array2DView< T >::rowPitchBytes() const
{
	return m_rowPitchBytes;
}
