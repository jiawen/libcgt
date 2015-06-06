template< typename T >
Array1DView< T >::Array1DView() :
	m_size( 0 ),
	m_stride( 0 ),
	m_pPointer( nullptr )
{

}

template< typename T >
Array1DView< T >::Array1DView( void* pPointer, int size ) :
	m_size( size ),
	m_stride( sizeof( T ) ),
	m_pPointer( reinterpret_cast< uint8_t* >( pPointer ) )
{

}

template< typename T >
Array1DView< T >::Array1DView( void* pPointer, int size, int stride ) :
	m_size( size ),
    m_stride( stride ),
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
T* Array1DView< T >::elementPointer( int x )
{
	return reinterpret_cast< T* >( m_pPointer + x * elementStrideBytes() );
}

template< typename T >
T& Array1DView< T >::operator [] ( int k )
{
	T* q = reinterpret_cast< T* >( m_pPointer + k * m_stride );
	return *q;
}

template< typename T >
int Array1DView< T >::width() const
{
	return m_size;
}

template< typename T >
int Array1DView< T >::size() const
{
	return m_size;
}

template< typename T >
int Array1DView< T >::numElements() const
{
	return m_size;
}

template< typename T >
int Array1DView< T >::elementStrideBytes() const
{
	return m_stride;
}

template< typename T >
int Array1DView< T >::stride() const
{
	return m_stride;
}

template< typename T >
bool Array1DView< T >::elementsArePacked() const
{
    return( m_stride == sizeof( T ) );
}

template< typename T >
bool Array1DView< T >::packed() const
{
    return( m_stride == sizeof( T ) );
}

template< typename T >
Array1DView< T >::operator Array1DView< const T >() const
{
    return Array1DView< const T >( m_pPointer, m_size, m_stride );
}
