#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <cutil.h>

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
	DeviceArray3D( const Array3D< T >& src );
	virtual ~DeviceArray3D();
	
	bool isNull() const;
	bool notNull() const;

	int width() const;
	int height() const;
	int depth() const;
	int3 size() const;
	int numElements() const;

	// indexing fun
	int subscriptToIndex( int x, int y, int z ) const;
	int3 indexToSubscript( int index ) const;

	// The number of bytes between rows within any slice
	size_t rowPitch() const;

	// The number of bytes between slices
	size_t slicePitch() const;	

	// Total size of the data in bytes (counting alignment)
	size_t sizeInBytes() const;

	// resizes the vector
	// original data is not preserved
	void resize( int width, int height, int depth );

	// fills this array with the 0 byte pattern
	void clear();

	// fills this array with value
	void fill( const T& value );

	// copy from host array src to this
	void copyFromHost( const Array3D< T >& src );

	// copy from this to host array dst
	void copyToHost( Array3D< T >& dst ) const;

	// implicit cast to pitched pointer
	operator cudaPitchedPtr() const;

	cudaPitchedPtr pitchedPointer() const;

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
