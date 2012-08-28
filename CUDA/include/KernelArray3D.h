#pragma once

#include <common/BasicTypes.h>

template< typename T >
struct KernelArray3D
{
	cudaPitchedPtr pitchedPointer;
	int width;
	int height;
	int depth;
	size_t slicePitch;

	__host__
	KernelArray3D( cudaPitchedPtr _pitchedPointer, int _width, int _height, int _depth );

	__inline__ __device__
	T* rowPointer( int y, int z );
	
	__inline__ __device__
	const T* rowPointer( int y, int z ) const;

	__inline__ __device__
	T* slicePointer( int z );

	__inline__ __device__
	const T* slicePointer( int z ) const;

	__inline__ __device__
	const T& operator () ( int x, int y, int z ) const;

	__inline__ __device__
	T& operator () ( int x, int y, int z );

	__inline__ __device__
	T& operator () ( const int3& xyz );

	__inline__ __device__
	const T& operator () ( const int3& xyz ) const;

	__inline__ __device__
	int subscriptToIndex( int x, int y, int z ) const;

	__inline__ __device__
	int3 indexToSubscript( int k ) const;
};

template< typename T >
__host__
KernelArray3D< T >::KernelArray3D( cudaPitchedPtr _pitchedPointer, int _width, int _height, int _depth ) :

	pitchedPointer( _pitchedPointer ),
	width( _width ),
	height( _height ),
	depth( _depth ),
	slicePitch( _pitchedPointer.pitch * _pitchedPointer.ysize )

{

}

template< typename T >
__inline__ __device__
T* KernelArray3D< T >::rowPointer( int y, int z )
{
	ubyte* p = reinterpret_cast< ubyte* >( pitchedPointer.ptr );

	size_t rowPitch = pitchedPointer.pitch;

	// TODO: switch pointer arithmetic to array indexing?
	ubyte* pSlice = p + z * slicePitch;
	return reinterpret_cast< T* >( pSlice + y * rowPitch );
}

template< typename T >
__inline__ __device__
const T* KernelArray3D< T >::rowPointer( int y, int z ) const
{
	ubyte* p = reinterpret_cast< ubyte* >( pitchedPointer.ptr );

	size_t rowPitch = pitchedPointer.pitch;

	// TODO: switch pointer arithmetic to array indexing?
	ubyte* pSlice = p + z * slicePitch;
	return reinterpret_cast< T* >( pSlice + y * rowPitch );
}

template< typename T >
__inline__ __device__
T* KernelArray3D< T >::slicePointer( int z )
{
	ubyte* p = reinterpret_cast< ubyte* >( pitchedPointer.ptr );

	// TODO: switch pointer arithmetic to array indexing?
	return reinterpret_cast< T* >( p + z * slicePitch );
}

template< typename T >
__inline__ __device__
const T* KernelArray3D< T >::slicePointer( int z ) const
{
	char* p = reinterpret_cast< char* >( pitchedPointer.ptr );

	// TODO: switch pointer arithmetic to array indexing?
	return reinterpret_cast< T* >( p + z * slicePitch );
}

template< typename T >
__inline__ __device__
const T& KernelArray3D< T >::operator () ( int x, int y, int z ) const
{
	return rowPointer( y, z )[ x ];
}

template< typename T >
__inline__ __device__
T& KernelArray3D< T >::operator () ( int x, int y, int z )
{
	return rowPointer( y, z )[ x ];
}

template< typename T >
__inline__ __device__
T& KernelArray3D< T >::operator () ( const int3& xyz )
{
	return rowPointer( xyz.y, xyz.z )[ xyz.x ];
}

template< typename T >
__inline__ __device__
const T& KernelArray3D< T >::operator () ( const int3& xyz ) const
{
	return rowPointer( xyz.y, xyz.z )[ xyz.x ];
}

template< typename T >
int KernelArray3D< T >::subscriptToIndex( int x, int y, int z ) const
{
	return z * width * height + y * width + x;
}

template< typename T >
int3 KernelArray3D< T >::indexToSubscript( int k ) const
{
	int wh = width * height;
	int z = k / wh;

	int ky = k - z * wh;
	int y = ky / width;

	int x = ky - y * width;
	return make_int3( x, y, z );
}