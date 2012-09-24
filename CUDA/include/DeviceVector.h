#pragma once

#include <vector>

#include <cuda_runtime.h>
#include <cutil.h>

#include "KernelVector.h"

// Basic vector interface around CUDA global memory
// Wraps around cudaMalloc() (linear allocation)
template< typename T >
class DeviceVector
{
public:

	DeviceVector();
	DeviceVector( int length );
	DeviceVector( const DeviceVector< T >& copy );
	DeviceVector( DeviceVector< T >&& move );
	DeviceVector< T >& operator = ( const DeviceVector< T >& copy );
	DeviceVector< T >& operator = ( DeviceVector< T >&& move );
	virtual ~DeviceVector();

	bool isNull() const;
	bool notNull() const;

	int length() const;
	size_t sizeInBytes() const;	

	// resizes the vector
	// original data is not preserved
	void resize( int length );

	// sets the vector to 0 (all bytes to 0)
	void clear();

	// fills this array with value
	void fill( const T& value );

	// get an element of the vector from the device
	// WARNING: probably slow as it incurs a cudaMemcpy
	T get( int index ) const;

	// sets an element of the vector from the host
	// WARNING: probably slow as it incurs a cudaMemcpy
	void set( int index, const T& value );

	// copy length() elements from input --> device vector
	// this is automatically resized
	void copyFromHost( const std::vector< T >& src );

	// copy length() elements from device vector --> host
	// dst is automatically resized
	void copyToHost( std::vector< T >& dst ) const;

	// implicit cast to device pointer
	operator T* () const;

	T* devicePointer() const;

	KernelVector< T > kernelVector() const;

private:

	int m_sizeInBytes;
	int m_length;
	T* m_devicePointer;

	void destroy();
};

#include "DeviceVector.inl"
