#pragma once

template< typename T >
struct DeviceArray3D
{
	cudaPitchedPtr pitchedPointer;
	int width;
	int height;
	int depth;
	size_t slicePitch;

	__host__
	DeviceArray3D( cudaPitchedPtr _pitchedPointer, int _width, int _height, int _depth );

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
};

template< typename T >
__host__
DeviceArray3D< T >::DeviceArray3D( cudaPitchedPtr _pitchedPointer, int _width, int _height, int _depth ) :

	pitchedPointer( _pitchedPointer ),
	width( _width ),
	height( _height ),
	depth( _depth ),
	slicePitch( _pitchedPointer.pitch * _pitchedPointer.ysize )

{

}

template< typename T >
__inline__ __device__
T* DeviceArray3D< T >::rowPointer( int y, int z )
{
	// TODO: char --> ubyte
	char* p = reinterpret_cast< char* >( pitchedPointer.ptr );

	size_t rowPitch = pitchedPointer.pitch;

	// TODO: switch pointer arithmetic to array indexing?
	char* pSlice = p + z * slicePitch;
	return reinterpret_cast< T* >( pSlice + y * rowPitch );
}

template< typename T >
__inline__ __device__
const T* DeviceArray3D< T >::rowPointer( int y, int z ) const
{
	// TODO: char --> ubyte
	char* p = reinterpret_cast< char* >( pitchedPointer.ptr );

	size_t rowPitch = pitchedPointer.pitch;

	// TODO: switch pointer arithmetic to array indexing?
	char* pSlice = p + z * slicePitch;
	return reinterpret_cast< T* >( pSlice + y * rowPitch );
}

template< typename T >
__inline__ __device__
T* DeviceArray3D< T >::slicePointer( int z )
{
	// TODO: char --> ubyte
	char* p = reinterpret_cast< char* >( pitchedPointer.ptr );

	// TODO: switch pointer arithmetic to array indexing?
	return reinterpret_cast< T* >( p + z * slicePitch );
}

template< typename T >
__inline__ __device__
const T* DeviceArray3D< T >::slicePointer( int z ) const
{
	// TODO: char --> ubyte
	char* p = reinterpret_cast< char* >( pitchedPointer.ptr );

	// TODO: switch pointer arithmetic to array indexing?
	return reinterpret_cast< T* >( p + z * slicePitch );
}

template< typename T >
__inline__ __device__
const T& DeviceArray3D< T >::operator () ( int x, int y, int z ) const
{
	return rowPointer( y, z )[ x ];
}

template< typename T >
__inline__ __device__
T& DeviceArray3D< T >::operator () ( int x, int y, int z )
{
	return rowPointer( y, z )[ x ];
}

template< typename T >
__inline__ __device__
T& DeviceArray3D< T >::operator () ( const int3& xyz )
{
	return rowPointer( xyz.y, xyz.z )[ xyz.x ];
}

template< typename T >
__inline__ __device__
const T& DeviceArray3D< T >::operator () ( const int3& xyz ) const
{
	return rowPointer( xyz.y, xyz.z )[ xyz.x ];
}
