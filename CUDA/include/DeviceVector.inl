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
DeviceVector< T >::DeviceVector( const std::vector< T >& src ) :

	m_sizeInBytes( 0 ),
	m_length( -1 ),
	m_devicePointer( nullptr )

{
	copyFromHost( src );
}

template< typename T >
DeviceVector< T >::DeviceVector( const DeviceVector< T >& copy ) :

	m_sizeInBytes( 0 ),
	m_length( -1 ),
	m_devicePointer( nullptr )

{
	copyFromDevice( copy );
}

template< typename T >
DeviceVector< T >::DeviceVector( DeviceVector< T >&& move )
{
	m_sizeInBytes = move.m_sizeInBytes;
	m_length = move.m_length;
	m_devicePointer = move.m_devicePointer;

	move.m_sizeInBytes = 0;
	move.m_length = -1;
	move.m_devicePointer = nullptr;
}

template< typename T >
DeviceVector< T >& DeviceVector< T >::operator = ( const DeviceVector< T >& copy )
{
	if( this != &copy )
	{
		copyFromDevice( copy );
	}
	return *this;
}

template< typename T >
DeviceVector< T >& DeviceVector< T >::operator = ( DeviceVector< T >&& move )
{
	if( this != &move )
	{
		destroy();

		m_sizeInBytes = move.m_sizeInBytes;
		m_length = move.m_length;
		m_devicePointer = move.m_devicePointer;

		move.m_sizeInBytes = 0;
		move.m_length = -1;
		move.m_devicePointer = nullptr;
	}
	return *this;
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
void DeviceVector< T >::clear()
{
	CUDA_SAFE_CALL( cudaMemset( m_devicePointer, 0, m_sizeInBytes ) );
}

template< typename T >
void DeviceVector< T >::fill( const T& value )
{
	std::vector< T > h_array( length(), value );
	copyFromHost( h_array );
}

template< typename T >
T DeviceVector< T >::get( int index ) const
{
	T output;
	CUDA_SAFE_CALL( cudaMemcpy( &output, m_devicePointer + index, sizeof( T ), cudaMemcpyDeviceToHost ) );
	return output;
}

template< typename T >
T DeviceVector< T >::operator [] ( int index ) const
{
	return get( index );
}

template< typename T >
void DeviceVector< T >::set( int index, const T& value )
{
	CUDA_SAFE_CALL( cudaMemcpy( m_devicePointer + index, &value, sizeof( T ), cudaMemcpyHostToDevice ) );
}

template< typename T >
void DeviceVector< T >::copyFromDevice( const DeviceVector< T >& src )
{
	resize( src.length() );
	CUDA_SAFE_CALL( cudaMemcpy( m_devicePointer, src.m_devicePointer, src.m_sizeInBytes, cudaMemcpyDeviceToDevice ) );
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
const T* DeviceVector< T >::devicePointer() const
{
	return m_devicePointer;
}

template< typename T >
T* DeviceVector< T >::devicePointer()
{
	return m_devicePointer;
}

template< typename T >
KernelVector< T > DeviceVector< T >::kernelVector() const
{
	return KernelVector< T >( m_devicePointer, m_length );
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
