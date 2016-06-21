template< typename T >
__inline__ __host__ __device__
KernelArray1D< T >::KernelArray1D() :
    pointer( nullptr ),
    length( 0 )
{

}

template< typename T >
__inline__ __host__ __device__
KernelArray1D< T >::KernelArray1D( T* _pointer, int _length ) :
    pointer( _pointer ),
    length( _length )
{

}

template< typename T >
__inline__ __device__
const T& KernelArray1D< T >::operator [] ( int i ) const
{
    return pointer[ i ];
}

template< typename T >
__inline__ __device__
T& KernelArray1D< T >::operator [] ( int i )
{
    return pointer[ i ];
}
