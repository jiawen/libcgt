#pragma once

#include <cutil.h>

// A simple wrapper for a single device variable
// T should be a plain old data type or a struct where sizeof() is well defined
template< typename T >
class DeviceVariable
{
public:

	DeviceVariable( const T& initialValue = T() );
	DeviceVariable( const DeviceVariable< T >& copy );
	DeviceVariable( DeviceVariable< T >&& move );
	DeviceVariable< T >& operator = ( const DeviceVariable< T >& copy );
	DeviceVariable< T >& operator = ( DeviceVariable< T >&& move );
	DeviceVariable< T >& operator = ( const T& copy ); // copy from host, same as set()
	~DeviceVariable();

	// copy device --> host
	T get() const;
	
	// copy host --> device
	void set( const T& value );

	// copy device --> device
	void set( const DeviceVariable< T >& value );

	const T* devicePointer() const;
	T* devicePointer();

private:

	void destroy();

	T* md_pDevicePointer;

};

#include "DeviceVariable.inl"
