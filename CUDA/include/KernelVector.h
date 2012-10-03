#pragma once

template< typename T >
struct KernelVector
{
	T* pointer;
	int length;

	__inline__ __host__
	KernelVector();

	__inline__ __host__
	KernelVector( T* _pointer, int _length );

	__inline__ __device__
	const T& operator [] ( int i ) const;

	__inline__ __device__
	T& operator [] ( int i );
};

template< typename T >
KernelVector< T >::KernelVector() :

	pointer( nullptr ),
	length( -1 )

{

}

template< typename T >
KernelVector< T >::KernelVector( T* _pointer, int _length ) :

	pointer( _pointer ),
	length( _length )

{

}

template< typename T >
const T& KernelVector< T >::operator [] ( int i ) const
{
	return pointer[ i ];
}

template< typename T >
T& KernelVector< T >::operator [] ( int i )
{
	return pointer[ i ];
}
