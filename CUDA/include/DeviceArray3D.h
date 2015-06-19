#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <helper_cuda.h>

#include <common/Array3D.h>

#include "KernelArray3D.h"

// Basic 3D array interface around CUDA global memory
// Wraps around cudaMalloc3D() (linear allocation with pitch)
template< typename T >
class DeviceArray3D
{
public:

	DeviceArray3D();
	DeviceArray3D( int width, int height, int depth );
	DeviceArray3D( const int& size );
	DeviceArray3D( const Array3D< T >& src );
	DeviceArray3D( const DeviceArray3D< T >& copy );
	DeviceArray3D( DeviceArray3D< T >&& move );
	DeviceArray3D< T >& operator = ( const Array3D< T >& src );
	DeviceArray3D< T >& operator = ( const DeviceArray3D< T >& copy );
	DeviceArray3D< T >& operator = ( DeviceArray3D< T >&& move );
	virtual ~DeviceArray3D();
	
	bool isNull() const;
	bool notNull() const;

	int width() const;
	int height() const;
	int depth() const;
	int3 size() const;
	int numElements() const;

	// The number of bytes between rows within any slice
	size_t rowPitch() const;

	// The number of bytes between slices
	size_t slicePitch() const;	

	// Total size of the data in bytes (counting alignment)
	size_t sizeInBytes() const;

	// resizes the array, original data is not preserved
	void resize( int width, int height, int depth );
	void resize( const int3& size );
	
	// TODO: implement get/set/operator() with int3, vector3
	// TODO: implement constructors for strided, pitched, slicePitched

	// fills this array with the 0 byte pattern
	void clear();

	// fills this array with value
	void fill( const T& value );

	// get an element of the array from the device
	// WARNING: probably slow as it incurs a cudaMemcpy
	T get( int x, int y, int z ) const;
	T get( const int3& subscript ) const;

	// get an element of the array from the device
	// WARNING: probably slow as it incurs a cudaMemcpy
	T operator () ( int x, int y, int z ) const;
	T operator [] ( const int3& subscript ) const;

	// sets an element of the array from the host
	// WARNING: probably slow as it incurs a cudaMemcpy
	void set( int x, int y, int z, const T& value );

	// copy from another DeviceArray3D to this
	// this is automatically resized
	void copyFromDevice( const DeviceArray3D< T >& src );

	// copy from host array src to this
	void copyFromHost( const Array3D< T >& src );

	// copy from this to host array dst
	void copyToHost( Array3D< T >& dst ) const;

	const cudaPitchedPtr pitchedPointer() const;
	cudaPitchedPtr pitchedPointer();

	KernelArray3D< T > kernelArray() const;

	void load( const char* filename );
	void save( const char* filename ) const;

private:

	int m_width;
	int m_height;
	int m_depth;

	size_t m_sizeInBytes;
	cudaPitchedPtr m_pitchedPointer;
	cudaExtent m_extent;

	// frees the memory if this is not null
	void destroy();	
};

#include "DeviceArray3D.inl"
