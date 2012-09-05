#pragma once

#include "math/Indexing.h"
#include "vecmath/Vector3i.h"

// a 3D array view that wraps around a raw pointer
// and does not take ownership

// TODO: rowPitch, slicePitch to get sub window of volume

template< typename T >
class Array3DView
{
public:

	Array3DView( int _width, int _height, int _depth, T* _pointer );

	const T& operator () ( int x, int y, int z ) const; // read
	T& operator () ( int x, int y, int z ); // write

	const T& operator () ( const Vector3i& xyz ) const; // read
	T& operator () ( const Vector3i& xyz ); // write

	int width;
	int height;
	int depth;
	T* pointer;
};

#include "Array3DView.inl"