template< typename T >
Array2DView< T >::Array2DView( int _width, int _height, T* _pointer ) :

	width( _width ),
	height( _height ),
	rowPitchBytes( _width * sizeof( T ) ),
	pointer( _pointer )

{

}

template< typename T >
Array2DView< T >::Array2DView( int _width, int _height, int _rowPitchBytes, T* _pointer ) :

	width( _width ),
	height( _height ),
	rowPitchBytes( _rowPitchBytes ),
	pointer( _pointer )

{

}

template< typename T >
T* Array2DView< T >::rowPointer( int y )
{
	ubyte* pBuffer = reinterpret_cast< ubyte* >( pointer );
	return reinterpret_cast< T* >( &( pBuffer[ y * rowPitchBytes ] ) );
}

template< typename T >
const T* Array2DView< T >::rowPointer( int y ) const
{
	ubyte* pBuffer = reinterpret_cast< ubyte* >( pointer );
	return reinterpret_cast< T* >( &( pBuffer[ y * rowPitchBytes ] ) );
}

template< typename T >
const T& Array2DView< T >::operator [] ( int k ) const
{
	int x;
	int y;
	Indexing::indexToSubscript2D( k, width, x, y );
	return ( *this )( x, y );
}

template< typename T >
T& Array2DView< T >::operator [] ( int k )
{
	int x;
	int y;
	Indexing::indexToSubscript2D( k, width, x, y );
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
