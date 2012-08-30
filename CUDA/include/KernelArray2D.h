#pragma once

#include <common/BasicTypes.h>

template< typename T >
struct KernelArray2D
{
	T* pointer;
	int width;
	int height;
	size_t pitch;

	__host__
	KernelArray2D( T* _pointer, int _width, int _height, size_t _pitch );

	__inline__ __device__
	T* rowPointer( int y );

	__inline__ __device__
	int2 size();

	__inline__ __device__
	T& operator () ( int x, int y );

	__inline__ __device__
	T& operator () ( int2 xy );

	__inline__ __device__
	int subscriptToIndex( int x, int y ) const;

	__inline__ __device__
	int2 indexToSubscript( int k ) const;
};

template< typename T >
__host__
KernelArray2D< T >::KernelArray2D( T* _pointer, int _width, int _height, size_t _pitch ) :

	pointer( _pointer ),
	width( _width ),
	height( _height ),
	pitch( _pitch )

{

}

template< typename T >
__inline__ __device__
T* KernelArray2D< T >::rowPointer( int y )
{
	ubyte* p = reinterpret_cast< ubyte* >( pointer );
	
	// TODO: switch pointer arithmetic to array indexing?
	return reinterpret_cast< T* >( p + y * pitch );
}

template< typename T >
__inline__ __device__
int2 KernelArray2D< T >::size()
{
	return make_int2( width, height );
}

template< typename T >
__inline__ __device__
T& KernelArray2D< T >::operator () ( int x, int y )
{
	return rowPointer( y )[ x ];
}

template< typename T >
__inline__ __device__
T& KernelArray2D< T >::operator () ( int2 xy )
{
	return rowPointer( xy.y )[ xy.x ];
}

template< typename T >
__inline__ __device__
int KernelArray2D< T >::subscriptToIndex( int x, int y ) const
{
	return y * width + x;
}

template< typename T >
__inline__ __device__
int2 KernelArray2D< T >::indexToSubscript( int k ) const
{
	int y = k / width;

	int x = k - y * width;
	return make_int2( x, y );
}