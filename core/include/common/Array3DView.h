#pragma once

#include "math/Indexing.h"
#include "vecmath/Vector3i.h"
#include "BasicTypes.h"

// a 3D array view that wraps around a raw pointer
// and does not take ownership

template< typename T >
class Array3DView
{
public:

	// create an Array3DView with the default row pitch of _width * sizeof( T )
	// and slice pitch of rowPitch * depth
	Array3DView( int _width, int _height, int _depth, T* _pointer );

	Array3DView( int _width, int _height, int _depth,
		int _rowPitchBytes, int _slicePitchBytes, T* _pointer );

	const T* rowPointer( int y, int z ) const;
	T* rowPointer( int y, int z );

	const T* slicePointer( int z ) const;
	T* slicePointer( int z );

	const T& operator [] ( int k ) const; // read
	T& operator [] ( int k ); // write

	const T& operator () ( int x, int y, int z ) const; // read
	T& operator () ( int x, int y, int z ); // write

	const T& operator () ( const Vector3i& xyz ) const; // read
	T& operator () ( const Vector3i& xyz ); // write

	int width;
	int height;
	int depth;
	int rowPitchBytes;
	int slicePitchBytes;
	T* pointer;
};

#include "Array3DView.inl"