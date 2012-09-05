#pragma once

#include "math/Indexing.h"
#include "vecmath/Vector2i.h"

// a 2D array view that wraps around a raw pointer
// and does not take ownership

// TODO: rowPitch to get sub window of volume

template< typename T >
class Array2DView
{
public:

	Array2DView( int _width, int _height, T* _pointer );

	const T& operator () ( int x, int y ) const; // read
	T& operator () ( int x, int y ); // write

	const T& operator () ( const Vector2i& xy ) const; // read
	T& operator () ( const Vector2i& xy ); // write

	int width;
	int height;
	T* pointer;
};

#include "Array2DView.inl"