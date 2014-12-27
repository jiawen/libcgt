template< typename T >
Array1DView< T >::Array1DView() :

	m_length( 0 ),
	m_elementStrideBytes( 0 ),
	m_pPointer( nullptr )

{

}

template< typename T >
Array1DView< T >::Array1DView( void* pPointer, int length ) :

	m_length( length ),
	m_strideBytes( sizeof( T ) ),
	m_pPointer( reinterpret_cast< uint8_t* >( pPointer ) )

{

}

template< typename T >
Array1DView< T >::Array1DView( void* pPointer, int length, int elementStrideBytes ) :

	m_length( length ),
    m_strideBytes( strideBytes ),
    m_pPointer( reinterpret_cast< uint8_t* >( pPointer ) )

{

}

template< typename T >
bool Array1DView< T >::isNull() const
{
	return( m_pPointer == nullptr );
}

template< typename T >
bool Array1DView< T >::notNull() const
{
	return( m_pPointer != nullptr );
}

template< typename T >
Array1DView< T >::operator const T* () const
{
	return m_pPointer;
}

template< typename T >
Array1DView< T >::operator T* ()
{
	return m_pPointer;
}

template< typename T >
const T* Array1DView< T >::pointer() const
{
	return reinterpret_cast< const T* >( m_pPointer );
}

template< typename T >
T* Array1DView< T >::pointer()
{
	return reinterpret_cast< T* >( m_pPointer );
}

template< typename T >
const T* Array1DView< T >::elementPointer( int x ) const
{
	return reinterpret_cast< T* >( m_pPointer + x * elementStrideBytes() );
}

template< typename T >
T* Array1DView< T >::elementPointer( int x )
{
	return reinterpret_cast< T* >( m_pPointer + x * elementStrideBytes() );
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
size_t Array1DView< T >::bytesReferenced() const
{
	return sizeof( T ) * length();
}

template< typename T >
size_t Array1DView< T >::bytesSpanned() const
{
	return std::abs( elementStrideBytes() ) * length();
}

template< typename T >
int Array1DView< T >::elementStrideBytes() const
{
	return m_elementStrideBytes;
}

template< typename T >
bool Array1DView< T >::packed() const
{
    return( m_strideBytes == sizeof( T ) );
}
