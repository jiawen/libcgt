template< typename T >
DeviceArray2D< T >::DeviceArray2D() :

	m_width( -1 ),
	m_height( -1 ),

	m_pitch( 0 ),
	m_sizeInBytes( 0 ),
	m_devicePointer( nullptr )

{
}

template< typename T >
DeviceArray2D< T >::DeviceArray2D( int width, int height ) :

	m_width( -1 ),
	m_height( -1 ),

	m_pitch( 0 ),
	m_sizeInBytes( 0 ),
	m_devicePointer( nullptr )

{
	resize( width, height );
}

template< typename T >
DeviceArray2D< T >::DeviceArray2D( const Array2D< T >& src ) :

	m_width( -1 ),
	m_height( -1 ),

	m_pitch( 0 ),
	m_sizeInBytes( 0 ),
	m_devicePointer( nullptr )

{	
	resize( src.width(), src.height() );
	copyFromHost( src );
}

template< typename T >
DeviceArray2D< T >::DeviceArray2D( const DeviceArray2D< T >& copy ) :

	m_width( -1 ),
	m_height( -1 ),

	m_pitch( 0 ),
	m_sizeInBytes( 0 ),
	m_devicePointer( nullptr )

{
	copyFromDevice( copy );	
}

template< typename T >
DeviceArray2D< T >& DeviceArray2D< T >::operator = ( const DeviceArray2D< T >& copy )
{
	if( this != &copy )
	{
		copyFromDevice( copy );
	}
	return *this;
}

template< typename T >
// virtual
DeviceArray2D< T >::~DeviceArray2D()
{
	destroy();
}

template< typename T >
bool DeviceArray2D< T >::isNull() const
{
	return( m_devicePointer == nullptr );
}

template< typename T >
bool DeviceArray2D< T >::notNull() const
{
	return( m_devicePointer != nullptr );
}

template< typename T >
int DeviceArray2D< T >::width() const
{
	return m_width;
}

template< typename T >
int DeviceArray2D< T >::height() const
{
	return m_height;
}

template< typename T >
int DeviceArray2D< T >::numElements() const
{
	return m_width * m_height;
}

template< typename T >
int DeviceArray2D< T >::subscriptToIndex( int x, int y ) const
{
	return( y * m_width + x );
}

template< typename T >
int2 DeviceArray2D< T >::indexToSubscript( int k ) const
{
	int y = k / m_width;
	int x = k - y * m_width;
	return make_int2( x, y );
}

template< typename T >
size_t DeviceArray2D< T >::pitch() const
{
	return m_pitch;
}

template< typename T >
size_t DeviceArray2D< T >::sizeInBytes() const
{
	return m_sizeInBytes;
}

template< typename T >
void DeviceArray2D< T >::resize( int width, int height )
{
	if( width == m_width && height == m_height )
	{
		return;
	}

	destroy();

	m_width = width;
	m_height = height;

	CUDA_SAFE_CALL
	(
		cudaMallocPitch
		(
			reinterpret_cast< void** >( &m_devicePointer ),
			&m_pitch,
			m_width * sizeof( T ),
			m_height
		)
	);

	m_sizeInBytes = m_pitch * height;
}

template< typename T >
void DeviceArray2D< T >::clear()
{
	CUDA_SAFE_CALL( cudaMemset2D( devicePointer(), pitch(), 0, widthInBytes(), height() ) );
}

template< typename T >
void DeviceArray2D< T >::fill( const T& value )
{
	Array2D< T > h_array( width(), height(), value );
	copyFromHost( h_array );
}

template< typename T >
void DeviceArray2D< T >::copyFromDevice( const DeviceArray2D< T >& src )
{
	resize( src.m_width, src.m_height );

	CUDA_SAFE_CALL( cudaMemcpy2D( m_devicePointer, m_pitch, src.m_devicePointer, src.m_pitch,
		src.m_width, src.m_height, cudaMemcpyDeviceToDevice ) );
}

template< typename T >
void DeviceArray2D< T >::copyFromHost( const Array2D< T >& src )
{
	resize( src.width(), src.height() );
	CUDA_SAFE_CALL
	(
		cudaMemcpy2D
		(
			devicePointer(), pitch(),
			src, src.width() * sizeof( T ),
			src.width() * sizeof( T ), src.height(),
			cudaMemcpyHostToDevice
		)
	);
}

template< typename T >
void DeviceArray2D< T >::copyToHost( Array2D< T >& dst ) const
{
	dst.resize( width(), height() );
	CUDA_SAFE_CALL
	(
		cudaMemcpy2D
		(
			dst, dst.width() * sizeof( T ),
			devicePointer(), pitch(),
			widthInBytes(), height(),
			cudaMemcpyDeviceToHost
		)
	);
}

template< typename T >
void DeviceArray2D< T >::copyFromArray( cudaArray* src )
{
	CUDA_SAFE_CALL
	(
		cudaMemcpy2DFromArray
		(
			devicePointer(), pitch(),
			src,
			0, 0,
			widthInBytes(), height(),
			cudaMemcpyDeviceToDevice
		)
	);
}

template< typename T >
void DeviceArray2D< T >::copyToArray( cudaArray* dst ) const
{
	CUDA_SAFE_CALL
	(
		cudaMemcpy2DToArray
		(
			dst,
			0, 0,
			devicePointer(), pitch(),
			widthInBytes(), height(),
			cudaMemcpyDeviceToDevice
		)
	);
}

template< typename T >
DeviceArray2D< T >::operator T* () const
{
	return m_devicePointer;
}

template< typename T >
T* DeviceArray2D< T >::devicePointer() const
{
	return m_devicePointer;
}

template< typename T >
KernelArray2D< T > DeviceArray2D< T >::kernelArray() const
{
	return KernelArray2D< T >( m_devicePointer, m_width, m_height, m_pitch );
}

template< typename T >
void DeviceArray2D< T >::load( const char* filename )
{
	Array2D< T > h_arr( filename );
	if( !( h_arr.isNull() ) )
	{
		resize( h_arr.width(), h_arr.height() );
		copyFromHost( h_arr );
	}
}

template< typename T >
void DeviceArray2D< T >::save( const char* filename ) const
{
	Array2D< T > h_arr( width(), height() );
	copyToHost( h_arr );
	h_arr.save( filename );
}

template< typename T >
void DeviceArray2D< T >::destroy()
{
	if( notNull() )
	{
		CUDA_SAFE_CALL( cudaFree( m_devicePointer ) );
		m_devicePointer = nullptr;
	}

	m_width = -1;
	m_height = -1;
	m_pitch = 0;
	m_sizeInBytes = 0;
}

template< typename T >
size_t DeviceArray2D< T >::widthInBytes() const
{
	return m_width * sizeof( T );
}
