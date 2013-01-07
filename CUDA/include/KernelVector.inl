template< typename T >
__inline__ __host__ __device__
KernelVector< T >::KernelVector() :

	pointer( nullptr ),
	length( -1 )

{

}

template< typename T >
__inline__ __host__ __device__
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
