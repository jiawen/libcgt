#pragma once

// A simple wrapper for a single device variable
// T should be a plain old data type or a struct where sizeof() is well defined
template< typename T >
class DeviceVariable
{
public:

	DeviceVariable( const T& initialValue = T() );
	~DeviceVariable();

	// TODO: copy
	// TODO: move

	T get();
	void set( const T& value );

	const T* devicePointer() const;
	T* devicePointer();

private:

	T* md_pDevicePointer;

};

template< typename T >
DeviceVariable< T >::DeviceVariable( const T& initialValue ) :

	md_pDevicePointer( nullptr )

{
	cudaMalloc< T >( &md_pDevicePointer, sizeof( T ) );
	set( initialValue );
}

template< typename T >
DeviceVariable< T >::~DeviceVariable()
{
	if( md_pDevicePointer != nullptr )
	{
		cudaFree( md_pDevicePointer );
	}
}

template< typename T >
T DeviceVariable< T >::get()
{
	T output;
	cudaMemcpy( &output, md_pDevicePointer, sizeof( T ), cudaMemcpyDeviceToHost );
	return output;
}

template< typename T >
void DeviceVariable< T >::set( const T& value )
{
	cudaMemcpy( md_pDevicePointer, &value, sizeof( T ), cudaMemcpyHostToDevice );
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
