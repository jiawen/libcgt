template< typename T >
DeviceVector< T >::DeviceVector() :

	m_sizeInBytes( 0 ),
	m_length( -1 ),
	m_devicePointer( nullptr )

{

}


template< typename T >
DeviceVector< T >::DeviceVector( int length ) :

	m_sizeInBytes( 0 ),
	m_length( -1 ),
	m_devicePointer( nullptr )

{
	resize( length );
}

template< typename T >
// virtual
DeviceVector< T >::~DeviceVector()
{
	destroy();	
}

template< typename T >
bool DeviceVector< T >::isNull() const
{
	return( m_devicePointer == nullptr );
}

template< typename T >
bool DeviceVector< T >::notNull() const
{
	return( m_devicePointer != nullptr );
}

template< typename T >
int DeviceVector< T >::length() const
{
	return m_length;
}

template< typename T >
size_t DeviceVector< T >::sizeInBytes() const
{
	return m_sizeInBytes;
}

template< typename T >
void DeviceVector< T >::clear()
{
	CUDA_SAFE_CALL( cudaMemset( m_devicePointer, 0, m_sizeInBytes ) );
}

template< typename T >
void DeviceVector< T >::resize( int length )
{
	if( m_length == length )
	{
		return;
	}

	destroy();

	m_length = length;
	m_sizeInBytes = length * sizeof( T );

	CUDA_SAFE_CALL( cudaMalloc( reinterpret_cast< void** >( &m_devicePointer ), m_sizeInBytes ) );
}

template< typename T >
void DeviceVector< T >::copyFromHost( const std::vector< T >& src )
{
	resize( static_cast< int >( src.size() ) );
	CUDA_SAFE_CALL( cudaMemcpy( m_devicePointer, src.data(), m_sizeInBytes, cudaMemcpyHostToDevice ) );
}

template< typename T >
void DeviceVector< T >::copyToHost( std::vector< T >& dst ) const
{
	int len = length();
	dst.resize( len );
	T* dstPointer = dst.data();
	CUDA_SAFE_CALL( cudaMemcpy( dstPointer, m_devicePointer, m_sizeInBytes, cudaMemcpyDeviceToHost ) );
}

template< typename T >
DeviceVector< T >::operator T* () const
{
	return m_devicePointer;
}

template< typename T >
T* DeviceVector< T >::devicePointer() const
{
	return m_devicePointer;
}

template< typename T >
void DeviceVector< T >::destroy()
{
	if( notNull() )
	{
		CUDA_SAFE_CALL( cudaFree( m_devicePointer ) );
		m_devicePointer = nullptr;
	}

	m_sizeInBytes = 0;
	m_length = -1;
	m_devicePointer = nullptr;
}

template< typename T >
KernelVector< T > DeviceVector< T >::kernelVector() const
{
	return KernelVector< T >( m_devicePointer, m_length );
}
