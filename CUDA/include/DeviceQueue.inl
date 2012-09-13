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
void DeviceQueue< T >::resize( uint length )
{
	m_elements.resize( length );
	clear();
}

template< typename T >
void DeviceQueue< T >::clear()
{
	m_readIndexAndCount.set( make_uint2( 0, 0 ) );
}

template< typename T >
int DeviceQueue< T >::count()
{
	return m_readIndexAndCount.get().y;
}

template< typename T >
KernelQueue< T > DeviceQueue< T >::kernelQueue()
{
	return KernelQueue< T >( m_readIndexAndCount.devicePointer(), m_elements.kernelVector() );
}

template< typename T >
void DeviceQueue< T >::copyFromHost( const std::vector< T >& src )
{
	uint length = static_cast< uint >( src.size() );
	resize( length ); // resize clears the queue	
	m_elements.copyFromHost( src );
}

template< typename T >
void DeviceQueue< T >::copyToHost( std::vector< T >& dst ) const
{
	std::vector< T > h_elements;
	m_elements.copyToHost( h_elements );

	uint2 rc = m_readIndexAndCount.get();
	int h = rc.x;
	int count = rc.y;

	dst.clear();
	dst.reserve( count );
	for( int i = 0; i < count; ++i )
	{
		dst.push_back( h_elements[ h + i ] );
	}
}
