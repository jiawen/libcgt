#pragma once

template< typename T >
struct KernelVector
{
	T* pointer;
	int length;

	__host__
	KernelVector( T* _pointer, int _length );

	__inline__ __device__
	const T& operator [] ( int i ) const;

	__inline__ __device__
	T& operator [] ( int i );
};

template< typename T >
__host__
KernelVector< T >::KernelVector( T* _pointer, int _length ) :

	pointer( _pointer ),
	length( _length )

{

}

template< typename T >
__inline__ __device__
const T& KernelVector< T >::operator [] ( int i ) const
{
	return pointer[ i ];
}

template< typename T >
__inline__ __device__
T& KernelVector< T >::operator [] ( int i )
{
	return pointer[ i ];
}
