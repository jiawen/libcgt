template< typename T >
DeviceVariable< T >::DeviceVariable( const T& initialValue ) :

	md_pDevicePointer( nullptr )

{
	CUDA_SAFE_CALL( cudaMalloc< T >( &md_pDevicePointer, sizeof( T ) ) );
	set( initialValue );
}

template< typename T >
DeviceVariable< T >::DeviceVariable( const DeviceVariable< T >& copy ) :

	md_pDevicePointer( nullptr )

{
	CUDA_SAFE_CALL( cudaMalloc< T >( &md_pDevicePointer, sizeof( T ) ) );
	set( copy );
}

template< typename T >
DeviceVariable< T >& DeviceVariable< T >::operator = ( const DeviceVariable< T >& copy )
{
	if( this != &copy )
	{
		set( copy );
	}
	return *this;
}

template< typename T >
DeviceVariable< T >::~DeviceVariable()
{
	if( md_pDevicePointer != nullptr )
	{
		CUDA_SAFE_CALL( cudaFree( md_pDevicePointer ) );
	}
}

template< typename T >
T DeviceVariable< T >::get() const
{
	T output;
	CUDA_SAFE_CALL( cudaMemcpy( &output, md_pDevicePointer, sizeof( T ), cudaMemcpyDeviceToHost ) );
	return output;
}

template< typename T >
void DeviceVariable< T >::set( const T& value )
{
	CUDA_SAFE_CALL( cudaMemcpy( md_pDevicePointer, &value, sizeof( T ), cudaMemcpyHostToDevice ) );
}

template< typename T >
void DeviceVariable< T >::set( const DeviceVariable< T >& value )
{
	CUDA_SAFE_CALL( cudaMemcpy( md_pDevicePointer, value.md_pDevicePointer, sizeof( T ), cudaMemcpyDeviceToDevice ) );
}

template< typename T >
const T* DeviceVariable< T >::devicePointer() const
{
	return md_pDevicePointer;
}

template< typename T >
T* DeviceVariable< T >::devicePointer()
{
	return md_pDevicePointer;
}
