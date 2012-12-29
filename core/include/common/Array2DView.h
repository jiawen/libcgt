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

	// TODO: elementsArePacked(), rowsArePacked(), isLinear() for completely packed

	// create an Array2DView with the default row pitch = width * sizeof( T )
	Array2DView( int width, int height, void* pPointer );

	Array2DView( int width, int height, int rowPitchBytes, void* pPointer );

	const T* rowPointer( int y ) const;
	T* rowPointer( int y );

	const T& operator [] ( int k ) const; // read
	T& operator [] ( int k ); // write

	const T& operator () ( int x, int y ) const; // read
	T& operator () ( int x, int y ); // write

	const T& operator [] ( const Vector2i& xy ) const; // read
	T& operator [] ( const Vector2i& xy ); // write

	int width() const;
	int height() const;
	int rowPitchBytes() const;

private:

	int m_width;
	int m_height;
	int m_rowPitchBytes;
	ubyte* m_pPointer;
};

#include "Array2DView.inl"
