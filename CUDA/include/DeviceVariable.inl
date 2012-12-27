template< typename T >
DeviceVariable< T >::DeviceVariable( const T& initialValue ) :

	md_pDevicePointer( nullptr )

{
	checkCudaErrors( cudaMalloc< T >( &md_pDevicePointer, sizeof( T ) ) );
	set( initialValue );
}

template< typename T >
DeviceVariable< T >::DeviceVariable( const DeviceVariable< T >& copy ) :

	md_pDevicePointer( nullptr )

{
	checkCudaErrors( cudaMalloc< T >( &md_pDevicePointer, sizeof( T ) ) );
	set( copy );
}

template< typename T >
DeviceVariable< T >::DeviceVariable( DeviceVariable< T >&& move )
{
	md_pDevicePointer = move.md_pDevicePointer;
	move.md_pDevicePointer = nullptr;
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
DeviceVariable< T >& DeviceVariable< T >::operator = ( DeviceVariable< T >&& move )
{
	if( this != &move )
	{
		destroy();

		md_pDevicePointer = move.md_pDevicePointer;
		move.md_pDevicePointer = nullptr;
	}
	return *this;
}

template< typename T >
DeviceVariable< T >& DeviceVariable< T >::operator = ( const T& copy )
{
	set( copy );
	return *this;
}

template< typename T >
DeviceVariable< T >::~DeviceVariable()
{
	destroy();	
}

template< typename T >
T DeviceVariable< T >::get() const
{
	T output;
	checkCudaErrors( cudaMemcpy( &output, md_pDevicePointer, sizeof( T ), cudaMemcpyDeviceToHost ) );
	return output;
}

template< typename T >
void DeviceVariable< T >::set( const T& value )
{
	checkCudaErrors( cudaMemcpy( md_pDevicePointer, &value, sizeof( T ), cudaMemcpyHostToDevice ) );
}

template< typename T >
void DeviceVariable< T >::set( const DeviceVariable< T >& value )
{
	checkCudaErrors( cudaMemcpy( md_pDevicePointer, value.md_pDevicePointer, sizeof( T ), cudaMemcpyDeviceToDevice ) );
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

template< typename T >
void DeviceVariable< T >::destroy()
{
	if( md_pDevicePointer != nullptr )
	{
		checkCudaErrors( cudaFree( md_pDevicePointer ) );
	}
}
