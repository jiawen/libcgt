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

#include "DeviceVariable.inl"