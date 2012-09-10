
template< typename T >
DeviceQueue< T >::DeviceQueue()
{

}

template< typename T >
DeviceQueue< T >::DeviceQueue( uint length )
{
	resize( length );
}

template< typename T >
bool DeviceQueue< T >::isNull() const
{
	return m_elements.isNull();
}

template< typename T >
bool DeviceQueue< T >::notNull() const
{
	return !isNull();
}

template< typename T >
bool DeviceQueue< T >::resize( uint length )
{
	bool lengthIsValid = libcgt::cuda::isPowerOfTwo( length );

	assert( lengthIsValid );
	if( lengthIsValid )
	{
		m_elements.resize( length );
		clear();
	}

	return lengthIsValid;
}

template< typename T >
void DeviceQueue< T >::clear()
{
	m_headTail.set( make_uint2( 0, 0 ) );
}

template< typename T >
int DeviceQueue< T >::count()
{
	uint2 ht = m_headTail.get();
	return( ht.y - ht.x );
}

template< typename T >
KernelQueue< T > DeviceQueue< T >::kernelQueue()
{
	return KernelQueue< T >( m_headTail.devicePointer(), m_elements.kernelVector() );
}

template< typename T >
bool DeviceQueue< T >::copyFromHost( const std::vector< T >& src )
{
	uint length = static_cast< uint >( src.size() );
	bool succeeded = resize( length );
	if( succeeded )
	{
		// resize clears the queue
		m_elements.copyFromHost( src );
	}

	return succeeded;
}

template< typename T >
void DeviceQueue< T >::copyToHost( std::vector< T >& dst ) const
{
	std::vector< T > h_elements;
	m_elements.copyToHost( h_elements );

	uint2 ht = m_headTail.get();
	int h = ht.x;
	int t = ht.y;

	int count = t - h;
	int len = m_elements.length();

	dst.clear();
	dst.reserve( count );
	for( int i = h; i < t; ++i )
	{
		dst.push_back( h_elements[ i % len ] );
	}
}
