#pragma once

#include <cuda_runtime.h>
#include <cutil.h>

#include <common/Array2D.h>

#include "KernelArray2D.h"

// Basic 2D array interface around CUDA global memory
// Wraps around cudaMallocPitch() (linear allocation with pitch)
template< typename T >
class DeviceArray2D
{
public:

	DeviceArray2D();
	DeviceArray2D( int width, int height );
	DeviceArray2D( const Array2D< T >& src );
	virtual ~DeviceArray2D();
	
	bool isNull() const;
	bool notNull() const;

	int width() const;
	int height() const;
	int numElements() const;

	// The number of bytes between rows
	size_t pitch() const;	

	// Total size of the data in bytes (counting alignment)
	size_t sizeInBytes() const;

	// resizes the vector
	// original data is not preserved
	void resize( int width, int height );

	// sets the vector to 0 (all bytes to 0)
	void clear();

	// copy from cudaArray src to this
	void copyFromArray( cudaArray* src );

	// copy from this to cudaArray dst
	void copyToArray( cudaArray* dst ) const;

	// copy from host array src to this
	void copyFromHost( const Array2D< T >& src );

	// copy from this to host array dst
	void copyToHost( Array2D< T >& dst ) const;
	
	// copy length() elements from device vector --> host
	// void copyToHost( void* output );

	// implicit cast to device pointer
	operator T* () const;

	T* devicePtr() const;

	KernelArray2D< T > kernelArray() const;

	void load( const char* filename );
	void save( const char* filename ) const;

private:

	int m_width;
	int m_height;
	size_t m_pitch;
	size_t m_sizeInBytes;
	T* m_devicePtr;

	// frees the memory if this is not null
	void destroy();

	// Size of one row in bytes (not counting alignment)
	// Used for cudaMemset, which requires both a pitch and the original width
	size_t widthInBytes() const;
};

template< typename T >
DeviceArray2D< T >::DeviceArray2D() :

	m_width( -1 ),
	m_height( -1 ),

	m_pitch( 0 ),
	m_sizeInBytes( 0 ),
	m_devicePtr( NULL )

{
}

template< typename T >
DeviceArray2D< T >::DeviceArray2D( int width, int height ) :

	m_width( -1 ),
	m_height( -1 ),

	m_pitch( 0 ),
	m_sizeInBytes( 0 ),
	m_devicePtr( NULL )

{
	resize( width, height );
}

template< typename T >
DeviceArray2D< T >::DeviceArray2D( const Array2D< T >& src ) :

	m_width( -1 ),
	m_height( -1 ),

	m_pitch( 0 ),
	m_sizeInBytes( 0 ),
	m_devicePtr( NULL )

{
	resize( src.width(), src.height() );
	copyFromHost( src );
}

template< typename T >
// virtual
DeviceArray2D< T >::~DeviceArray2D()
{
	destroy();
}

template< typename T >
bool DeviceArray2D< T >::isNull() const
{
	return( m_devicePtr == NULL );
}

template< typename T >
bool DeviceArray2D< T >::notNull() const
{
	return( m_devicePtr != NULL );
}

template< typename T >
int DeviceArray2D< T >::width() const
{
	return m_width;
}

template< typename T >
int DeviceArray2D< T >::height() const
{
	return m_height;
}

template< typename T >
int DeviceArray2D< T >::numElements() const
{
	return m_width * m_height;
}

template< typename T >
size_t DeviceArray2D< T >::pitch() const
{
	return m_pitch;
}

template< typename T >
size_t DeviceArray2D< T >::sizeInBytes() const
{
	return m_sizeInBytes;
}

template< typename T >
void DeviceArray2D< T >::resize( int width, int height )
{
	if( width == m_width && height == m_height )
	{
		return;
	}

	destroy();

	m_width = width;
	m_height = height;

	CUDA_SAFE_CALL
	(
		cudaMallocPitch
		(
			reinterpret_cast< void** >( &m_devicePtr ),
			&m_pitch,
			m_width * sizeof( T ),
			m_height
		)
	);

	m_sizeInBytes = m_pitch * height;
}

template< typename T >
void DeviceArray2D< T >::clear()
{
	CUDA_SAFE_CALL( cudaMemset2D( devicePtr(), pitch(), 0, widthInBytes(), height() ) );
}

template< typename T >
void DeviceArray2D< T >::copyFromArray( cudaArray* src )
{
	CUDA_SAFE_CALL
	(
		cudaMemcpy2DFromArray
		(
			devicePtr(), pitch(),			
			src,
			0, 0,
			widthInBytes(), height(),
			cudaMemcpyDeviceToDevice
		)
	);
}

template< typename T >
void DeviceArray2D< T >::copyToArray( cudaArray* dst ) const
{
	CUDA_SAFE_CALL
	(
		cudaMemcpy2DToArray
		(
			dst,
			0, 0,
			devicePtr(), pitch(),
			widthInBytes(), height(),
			cudaMemcpyDeviceToDevice
		)
	);
}

template< typename T >
void DeviceArray2D< T >::copyFromHost( const Array2D< T >& src )
{
	CUDA_SAFE_CALL
	(
		cudaMemcpy2D
		(
			devicePtr(), pitch(),
			src, src.width() * sizeof( T ),
			src.width() * sizeof( T ), src.height(),
			cudaMemcpyHostToDevice
		)
	);
}

template< typename T >
void DeviceArray2D< T >::copyToHost( Array2D< T >& dst ) const
{
	CUDA_SAFE_CALL
	(
		cudaMemcpy2D
		(
			dst, dst.width() * sizeof( T ),
			devicePtr(), pitch(),
			widthInBytes(), height(),
			cudaMemcpyDeviceToHost
		)
	);
}

template< typename T >
DeviceArray2D< T >::operator T* () const
{
	return m_devicePtr;
}

template< typename T >
T* DeviceArray2D< T >::devicePtr() const
{
	return m_devicePtr;
}

template< typename T >
KernelArray2D< T > DeviceArray2D< T >::kernelArray() const
{
	return KernelArray2D< T >( m_devicePtr, m_width, m_height, m_pitch );
}

template< typename T >
void DeviceArray2D< T >::load( const char* filename )
{
	Array2D< T > h_arr( filename );
	if( !( h_arr.isNull() ) )
	{
		resize( h_arr.width(), h_arr.height() );
		copyFromHost( h_arr );
	}
}

template< typename T >
void DeviceArray2D< T >::save( const char* filename ) const
{
	Array2D< T > h_arr( width(), height() );
	copyToHost( h_arr );
	h_arr.save( filename );
}

template< typename T >
void DeviceArray2D< T >::destroy()
{
	if( notNull() )
	{
		CUDA_SAFE_CALL( cudaFree( m_devicePtr ) );
		m_devicePtr = NULL;
	}

	m_width = -1;
	m_height = -1;
	m_pitch = 0;
	m_sizeInBytes = 0;
}

template< typename T >
size_t DeviceArray2D< T >::widthInBytes() const
{
	return m_width * sizeof( T );
}
