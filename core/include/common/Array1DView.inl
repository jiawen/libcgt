template< typename T >
Array1DView< T >::Array1DView< T >( int length, T* pPointer ) :

	m_length( length ),
	m_strideBytes( sizeof( T ) ),
	m_pPointer( pPointer )

{

}

template< typename T >
Array1DView< T >::Array1DView< T >( int length, int strideBytes, T* pPointer ) :

	m_length( length ),
	m_strideBytes( strideBytes ),
	m_pPointer( pPointer )

{

}

template< typename T >
const T* Array1DView< T >::pointer() const
{
	return m_pPointer;
}

template< typename T >
T* Array1DView< T >::pointer()
{
	return m_pPointer;
}

template< typename T >
const T& Array1DView< T >::operator [] ( int k ) const
{
	const ubyte* p = reinterpret_cast< const ubyte* >( m_pPointer );
	const T* q = reinterpret_cast< const T* >( p + k * m_strideBytes );
	return *q;
}

template< typename T >
T& Array1DView< T >::operator [] ( int k )
{
	ubyte* p = reinterpret_cast< ubyte* >( m_pPointer );
	T* q = reinterpret_cast< T* >( p + k * m_strideBytes );
	return *q;
}
