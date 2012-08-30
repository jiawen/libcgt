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

	T* devicePtr() const;

	KernelVector< T > kernelVector() const;

private:

	int m_sizeInBytes;
	int m_length;
	T* m_devicePtr;

	void destroy();
};

template< typename T >
DeviceVector< T >::DeviceVector() :

	m_sizeInBytes( 0 ),
	m_length( -1 ),
	m_devicePtr( NULL )

{

}


template< typename T >
DeviceVector< T >::DeviceVector( int length ) :

	m_sizeInBytes( 0 ),
	m_length( -1 ),
	m_devicePtr( NULL )

{
	resize( length );
}

template< typename T >
// virtual
DeviceVector< T >::~DeviceVector()
{
	destroy();	
}

template< typename T >
bool DeviceVector< T >::isNull() const
{
	return( m_devicePtr == NULL );
}

template< typename T >
bool DeviceVector< T >::notNull() const
{
	return( m_devicePtr != NULL );
}

template< typename T >
int DeviceVector< T >::length() const
{
	return m_length;
}

template< typename T >
size_t DeviceVector< T >::sizeInBytes() const
{
	return m_sizeInBytes;
}

template< typename T >
void DeviceVector< T >::clear()
{
	cudaMemset( m_devicePtr, 0, m_sizeInBytes );
}

template< typename T >
void DeviceVector< T >::resize( int length )
{
	if( m_length == length )
	{
		return;
	}

	destroy();

	m_length = length;
	m_sizeInBytes = length * sizeof( T );

	CUDA_SAFE_CALL( cudaMalloc( reinterpret_cast< void** >( &m_devicePtr ), m_sizeInBytes ) );
}

template< typename T >
void DeviceVector< T >::copyFromHost( const std::vector< T >& src )
{
	resize( static_cast< int >( src.size() ) );
	CUDA_SAFE_CALL( cudaMemcpy( m_devicePtr, src.data(), m_sizeInBytes, cudaMemcpyHostToDevice ) );
}

template< typename T >
void DeviceVector< T >::copyToHost( std::vector< T >& dst ) const
{
	dst.resize( length() );
	T* dstPointer = dst.data();
	CUDA_SAFE_CALL( cudaMemcpy( dstPointer, m_devicePtr, m_sizeInBytes, cudaMemcpyDeviceToHost ) );
}

template< typename T >
DeviceVector< T >::operator T* () const
{
	return m_devicePtr;
}

template< typename T >
T* DeviceVector< T >::devicePtr() const
{
	return m_devicePtr;
}

template< typename T >
void DeviceVector< T >::destroy()
{
	if( notNull() )
	{
		CUDA_SAFE_CALL( cudaFree( m_devicePtr ) );
		m_devicePtr = NULL;
	}

	m_sizeInBytes = 0;
	m_length = -1;	
}

template< typename T >
KernelVector< T > DeviceVector< T >::kernelVector() const
{
	return KernelVector< T >( m_devicePtr, m_length );
}
