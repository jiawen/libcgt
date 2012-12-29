template< typename T >
Array2DView< T >::Array2DView( int width, int height, void* pPointer ) :

	m_width( width ),
	m_height( height ),
	m_strideBytes( sizeof( T ) ),
	m_rowPitchBytes( width * sizeof( T ) ),
	m_pPointer( reinterpret_cast< ubyte* >( pPointer ) )

{

}

template< typename T >
Array2DView< T >::Array2DView( const Vector2i& size, void* pPointer ) :

	m_width( size.x ),
	m_height( size.y ),
	m_strideBytes( sizeof( T ) ),
	m_rowPitchBytes( size.x * sizeof( T ) ),
	m_pPointer( reinterpret_cast< ubyte* >( pPointer ) )

{

}

template< typename T >
Array2DView< T >::Array2DView( int width, int height, int strideBytes, int rowPitchBytes, void* pPointer ) :

	m_width( width ),
	m_height( height ),
	m_strideBytes( strideBytes ),
	m_rowPitchBytes( rowPitchBytes ),
	m_pPointer( reinterpret_cast< ubyte* >( pPointer ) )

{

}

template< typename T >
Array2DView< T >::Array2DView( const Vector2i& size, int strideBytes, int rowPitchBytes, void* pPointer ) :

	m_width( size.x ),
	m_height( size.y ),
	m_strideBytes( strideBytes ),
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
	const ubyte* pRowPointer = reinterpret_cast< const ubyte* >( rowPointer( y ) );
	const ubyte* pElementPointer = pRowPointer + x * m_strideBytes;
	const T* q = reinterpret_cast< const T* >( pElementPointer );
	return *q;
}

template< typename T >
T& Array2DView< T >::operator () ( int x, int y )
{
	ubyte* pRowPointer = reinterpret_cast< ubyte* >( rowPointer( y ) );
	ubyte* pElementPointer = pRowPointer + x * m_strideBytes;
	T* q = reinterpret_cast< T* >( pElementPointer );
	return *q;
}

template< typename T >
const T& Array2DView< T >::operator [] ( const Vector2i& xy ) const
{
	return ( *this )( xy.x, xy.y );
}

template< typename T >
T& Array2DView< T >::operator [] ( const Vector2i& xy )
{
	return ( *this )( xy.x, xy.y );
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
Vector2i Array2DView< T >::size() const
{
	return Vector2i( m_width, m_height );
}

template< typename T >
int Array2DView< T >::strideBytes() const
{
	return m_strideBytes;
}

template< typename T >
int Array2DView< T >::rowPitchBytes() const
{
	return m_rowPitchBytes;
}

template< typename T >
bool Array2DView< T >::elementsArePacked() const
{
	return m_strideBytes == sizeof( T );
}

template< typename T >
bool Array2DView< T >::rowsArePacked() const
{
	return m_rowPitchBytes == ( m_width * m_strideBytes );
}

template< typename T >
bool Array2DView< T >::packed() const
{
	return elementsArePacked() && rowsArePacked();
}
