#pragma once

#include <common/BasicTypes.h>

template< typename T >
struct KernelArray2D
{
	T* md_pPitchedPointer;
	int m_width;
	int m_height;
	size_t m_pitch;

	__inline__ __device__ __host__
	KernelArray2D();

	__inline__ __device__ __host__
	KernelArray2D( T* d_pPitchedPointer, int width, int height, size_t pitch );

	// wraps a KernelArray2D around linear device memory
	// (assumes that the memory pointed to by d_pLinearPointer is tightly packed,
	// if it's not, then the caller should construct using the constructor with pitch)
	__inline__ __device__ __host__
	KernelArray2D( T* d_pLinearPointer, int width, int height );		

	__inline__ __device__
	const T* rowPointer( int y ) const;

	__inline__ __device__
	T* rowPointer( int y );

	__inline__ __device__
	int width() const;

	__inline__ __device__
	int height() const;

	__inline__ __device__
	size_t pitch() const;

	__inline__ __device__
	int2 size() const;

	__inline__ __device__
	const T& operator () ( int x, int y ) const;

	__inline__ __device__
	const T& operator () ( const int2& xy ) const;

	__inline__ __device__
	T& operator () ( int x, int y );

	__inline__ __device__
	T& operator () ( const int2& xy );
};

#include "KernelArray2D.inl"