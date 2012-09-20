template< typename T >
Array3DView< T >::Array3DView( int _width, int _height, int _depth,
	T* _pointer ) :

	width( _width ),
	height( _height ),
	depth( _depth ),
	rowPitchBytes( _width * sizeof( T ) ),
	slicePitchBytes( _width * _height * sizeof( T ) ),
	pointer( _pointer )

{

}

template< typename T >
Array3DView< T >::Array3DView( int _width, int _height, int _depth,
	int _rowPitchBytes, int _slicePitchBytes, T* _pointer ) :

	width( _width ),
	height( _height ),
	depth( _depth ),
	rowPitchBytes( _rowPitchBytes ),
	slicePitchBytes( _slicePitchBytes ),
	pointer( _pointer )

{

}

template< typename T >
const T* Array3DView< T >::rowPointer( int y, int z ) const
{
	ubyte* pBuffer = reinterpret_cast< ubyte* >( pointer );
	return reinterpret_cast< T* >
	(
		&( pBuffer[ z * slicePitchBytes + y * rowPitchBytes ] )
	);
}

template< typename T >
T* Array3DView< T >::rowPointer( int y, int z )
{
	ubyte* pBuffer = reinterpret_cast< ubyte* >( pointer );
	return reinterpret_cast< T* >
	(
		&( pBuffer[ z * slicePitchBytes + y * rowPitchBytes ] )
	);
}

template< typename T >
const T* Array3DView< T >::slicePointer( int z ) const
{
	ubyte* pBuffer = reinterpret_cast< ubyte* >( pointer );
	return reinterpret_cast< T* >
	(
		&( pBuffer[ z * slicePitchBytes ] )
	);
}

template< typename T >
T* Array3DView< T >::slicePointer( int z )
{
	ubyte* pBuffer = reinterpret_cast< ubyte* >( pointer );
	return reinterpret_cast< T* >
	(
		&( pBuffer[ z * slicePitchBytes ] )
	);
}

template< typename T >
const T& Array3DView< T >::operator [] ( int k ) const
{
	int x;
	int y;
	int z;
	Indexing::indexToSubscript3D( k, width, height, x, y, z );
	return ( *this )( x, y, z );
}

template< typename T >
T& Array3DView< T >::operator [] ( int k )
{
	int x;
	int y;
	int z;
	Indexing::indexToSubscript3D( k, width, height, x, y, z );
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
const T& Array3DView< T >::operator () ( const Vector3i& xyz ) const
{
	return rowPointer( xyz.y, xyz.z )[ xyz.x ];
}

template< typename T >
T& Array3DView< T >::operator () ( const Vector3i& xyz )
{
	return rowPointer( xyz.y, xyz.z )[ xyz.x ];
}
