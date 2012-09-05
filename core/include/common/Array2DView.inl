template< typename T >
Array2DView< T >::Array2DView( int _width, int _height, T* _pointer ) :

	width( _width ),
	height( _height ),
	pointer( _pointer )

{

}

template< typename T >
const T& Array2DView< T >::operator () ( int x, int y ) const
{
	int k = Indexing::subscriptToIndex( x, y, width );
	return pointer[ k ];
}

template< typename T >
T& Array2DView< T >::operator () ( int x, int y )
{
	int k = Indexing::subscriptToIndex( x, y, width );
	return pointer[ k ];
}

template< typename T >
const T& Array2DView< T >::operator () ( const Vector2i& xy ) const
{
	int k = Indexing::subscriptToIndex( xy, width );
	return pointer[ k ];
}

template< typename T >
T& Array2DView< T >::operator () ( const Vector2i& xy )
{
	int k = Indexing::subscriptToIndex( xy, width );
	return pointer[ k ];
}
