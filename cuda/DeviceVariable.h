#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>

// A simple wrapper for a single device variable.
// T must be a plain old data type or a struct where sizeof() is well defined.
template< typename T >
class DeviceVariable
{
public:

    DeviceVariable( const T& initialValue = T() );
    DeviceVariable( const DeviceVariable< T >& copy );
    DeviceVariable( DeviceVariable< T >&& move );
    DeviceVariable< T >& operator = ( const DeviceVariable< T >& copy );
    DeviceVariable< T >& operator = ( DeviceVariable< T >&& move );

    // Copy from host, same as set().
    DeviceVariable< T >& operator = ( const T& copy );
    ~DeviceVariable();

    // Copy device --> host.
    T get() const;

    // Copy host --> device.
    void set( const T& value );

    // Copy device --> device.
    void set( const DeviceVariable< T >& value );

    // TODO: add implicit conversions to const T& and T&?

    const T* devicePointer() const;
    T* devicePointer();

private:

    void destroy();

    T* md_pDevicePointer;

};

#include "libcgt/cuda/DeviceVariable.inl"
