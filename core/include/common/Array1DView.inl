template< typename T >
Array1DView< T >::Array1DView( int length, void* pPointer ) :

	m_length( length ),
	m_strideBytes( sizeof( T ) ),
	m_pPointer( reinterpret_cast< ubyte* >( pPointer ) )

{

}

template< typename T >
Array1DView< T >::Array1DView( int length, int strideBytes, void* pPointer ) :

	m_length( length ),
	m_strideBytes( strideBytes ),
	m_pPointer( reinterpret_cast< ubyte* >( pPointer ) )

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
	const T* q = reinterpret_cast< const T* >( m_pPointer + k * m_strideBytes );
	return *q;
}

template< typename T >
T& Array1DView< T >::operator [] ( int k )
{
	T* q = reinterpret_cast< T* >( m_pPointer + k * m_strideBytes );
	return *q;
}

template< typename T >
int Array1DView< T >::length() const
{
	return m_length;
}

template< typename T >
int Array1DView< T >::strideBytes() const
{
	return m_strideBytes;
}

template< typename T >
bool Array1DView< T >::packed() const
{
	return( m_strideBytes == sizeof( T ) );
}
