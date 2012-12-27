#pragma once

// STL
#include <vector>

// CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>

// libcgt
#include <common/Array1DView.h>

// local
#include "KernelVector.h"

// Basic vector interface around CUDA global memory
// Wraps around cudaMalloc() (linear allocation)
template< typename T >
class DeviceVector
{
public:

	DeviceVector();
	DeviceVector( int length );
	DeviceVector( const std::vector< T >& src );
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

	// get an element of the vector from the device
	// WARNING: probably slow as it incurs a cudaMemcpy
	T operator [] ( int index ) const;

	// sets an element of the vector from the host
	// WARNING: probably slow as it incurs a cudaMemcpy
	void set( int index, const T& value );	

	// copy from another DeviceVector to this
	// this is automatically resized
	void copyFromDevice( const DeviceVector< T >& src );

	// copy length() elements from input --> device vector
	// this is automatically resized
	void copyFromHost( const std::vector< T >& src );

	// copy length() elements from device vector --> host
	// dst is automatically resized
	void copyToHost( std::vector< T >& dst ) const;

	// copy dst.length() elements from device vector --> host
	// starting from srcOffset
	// srcOffset must be >= 0
	// length() - srcOffset must be >= dst.length()
	// dst must be packed()
	// return false on failure
	bool copyToHost( Array1DView< T >& dst, int srcOffset = 0 ) const;

	const T* devicePointer() const;
	T* devicePointer();

	KernelVector< T > kernelVector() const;

private:

	size_t m_sizeInBytes;
	int m_length;
	T* m_devicePointer;

	void destroy();
};

#include "DeviceVector.inl"
