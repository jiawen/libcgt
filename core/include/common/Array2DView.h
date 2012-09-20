#pragma once

#include "common/BasicTypes.h"
#include "math/Indexing.h"
#include "vecmath/Vector2i.h"

// a 2D array view that wraps around a raw pointer
// and does not take ownership

template< typename T >
class Array2DView
{
public:

	// create an Array2DView with the default row pitch = _width * sizeof( T )
	Array2DView( int _width, int _height, T* _pointer );

	Array2DView( int _width, int _height, int _rowPitchBytes, T* _pointer );

	const T* rowPointer( int y ) const;
	T* rowPointer( int y );

	const T& operator [] ( int k ) const; // read
	T& operator [] ( int k ); // write

	const T& operator () ( int x, int y ) const; // read
	T& operator () ( int x, int y ); // write

	const T& operator () ( const Vector2i& xy ) const; // read
	T& operator () ( const Vector2i& xy ); // write

	int width;
	int height;
	int rowPitchBytes;
	T* pointer;
};

#include "Array2DView.inl"