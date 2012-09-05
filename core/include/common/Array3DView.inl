template< typename T >
Array3DView< T >::Array3DView( int _width, int _height, int _depth, T* _pointer ) :

	width( _width ),
	height( _height ),
	depth( _depth ),
	pointer( _pointer )

{

}

template< typename T >
const T& Array3DView< T >::operator () ( int x, int y, int z ) const
{
	int k = Indexing::subscriptToIndex( x, y, z, width, height );
	return pointer[ k ];
}

template< typename T >
T& Array3DView< T >::operator () ( int x, int y, int z )
{
	int k = Indexing::subscriptToIndex( x, y, z, width, height );
	return pointer[ k ];
}

template< typename T >
const T& Array3DView< T >::operator () ( const Vector3i& xyz ) const
{
	int k = Indexing::subscriptToIndex( xyz, width, height );
	return pointer[ k ];
}

template< typename T >
T& Array3DView< T >::operator () ( const Vector3i& xyz )
{
	int k = Indexing::subscriptToIndex( xyz, width, height );
	return pointer[ k ];
}
