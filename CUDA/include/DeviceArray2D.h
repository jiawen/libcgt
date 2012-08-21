#pragma once

template< typename T >
struct DeviceArray2D
{
	T* pointer;
	int width;
	int height;
	size_t pitch;

	__host__
	DeviceArray2D( T* _pointer, int _width, int _height, size_t _pitch );

	__inline__ __device__
	T* rowPointer( int y );

	__inline__ __device__
	T& operator () ( int x, int y );

	__inline__ __device__
	T& operator () ( int2 xy );

	__inline__ __device__
	T& operator () ( uint2 xy );
};

template< typename T >
__host__
DeviceArray2D< T >::DeviceArray2D( T* _pointer, int _width, int _height, size_t _pitch ) :

	pointer( _pointer ),
	width( _width ),
	height( _height ),
	pitch( _pitch )

{

}

template< typename T >
__inline__ __device__
T* DeviceArray2D< T >::rowPointer( int y )
{
	// TODO: char --> ubyte
	char* p = reinterpret_cast< char* >( pointer );
	
	// TODO: switch pointer arithmetic to array indexing?
	return reinterpret_cast< T* >( p + y * pitch );
}

template< typename T >
__inline__ __device__
T& DeviceArray2D< T >::operator () ( int x, int y )
{
	return rowPointer( y )[ x ];
}

template< typename T >
__inline__ __device__
T& DeviceArray2D< T >::operator () ( int2 xy )
{
	return rowPointer( xy.y )[ xy.x ];
}

template< typename T >
__inline__ __device__
T& DeviceArray2D< T >::operator () ( uint2 xy )
{
	return rowPointer( xy.y )[ xy.x ];
}
