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
	virtual ~DeviceVector();

	bool isNull() const;
	bool notNull() const;

	int length() const;
	size_t sizeInBytes() const;

	// sets the vector to 0 (all bytes to 0)
	void clear();

	// resizes the vector
	// original data is not preserved
	void resize( int length );

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