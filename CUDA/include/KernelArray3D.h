#pragma once

#include <common/BasicTypes.h>

template< typename T >
struct KernelArray3D
{
	// A cudaPitchedPtr contains
	// width: logical width of the array in elements
	// xsize = elementSize * width, in bytes
	// ysize = height of the array in elements
	// pitch = roundUpToAlignment( xsize ), in bytes

	cudaPitchedPtr md_pitchedPointer;
	int m_depth;

	__inline__ __device__ __host__
	KernelArray3D();

	__inline__ __device__ __host__
	KernelArray3D( cudaPitchedPtr d_pitchedPointer, int width, int height, int depth );

	// wraps a KernelArray3D (with pitchedPointer) around linear device memory
	// (assumes that the memory pointed to by d_pLinearPointer is tightly packed,
	// if it's not, then the caller should construct a cudaPitchedPtr directly)
	__inline__ __device__ __host__
	KernelArray3D( T* d_pLinearPointer, int width, int height, int depth );

	__inline__ __device__ __host__
	KernelArray3D( T* d_pLinearPointer, const int3& size );

	__inline__ __device__
	const T* rowPointer( int y, int z ) const;

	__inline__ __device__
	T* rowPointer( int y, int z );

	__inline__ __device__
	const T* slicePointer( int z ) const;

	__inline__ __device__
	T* slicePointer( int z );
	
	__inline__ __device__
	int width() const;

	__inline__ __device__
	int height() const;

	__inline__ __device__
	int depth() const;

	__inline__ __device__
	int3 size() const;

	__inline__ __device__
	size_t rowPitch() const;

	__inline__ __device__
	size_t slicePitch() const;

	__inline__ __device__
	const T& operator () ( int x, int y, int z ) const;

	__inline__ __device__
	T& operator () ( int x, int y, int z );

	__inline__ __device__
	const T& operator [] ( const int3& xyz ) const;

	__inline__ __device__
	T& operator [] ( const int3& xyz );

	template< typename S >
	__inline__ __device__ __host__
	KernelArray3D< S > reinterpretAs( int outputWidth, int outputHeight, int outputDepth );

	template< typename S >
	__inline__ __device__ __host__
	KernelArray3D< S > reinterpretAs( const int3& outputSize );

};

#include "KernelArray3D.inl"
