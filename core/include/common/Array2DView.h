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

	// create an Array2DView with
	// the default stride of sizeof( T )
	// and the default row pitch of width * sizeof( T )
	Array2DView( int width, int height, void* pPointer );
	Array2DView( const Vector2i& size, void* pPointer );

	// create an Array2DView with specified size, stride, and pitch
	Array2DView( int width, int height, int strideBytes, int rowPitchBytes, void* pPointer );
	Array2DView( const Vector2i& size, int strideBytes, int rowPitchBytes, void* pPointer );

	const T* rowPointer( int y ) const;
	T* rowPointer( int y );

	const T& operator [] ( int k ) const; // read
	T& operator [] ( int k ); // write

	const T& operator () ( int x, int y ) const; // read
	T& operator () ( int x, int y ); // write

	const T& operator [] ( const Vector2i& xy ) const; // read
	T& operator [] ( const Vector2i& xy ); // write

	// the logical size of the array view
	// (i.e., how many elements of type T there are)
	int width() const;
	int height() const;
	Vector2i size() const;

	// the space between the start of elements in bytes
	int strideBytes() const;

	// the space between the start of rows in bytes
	int rowPitchBytes() const;

	// returns true if there is no space between adjacent elements *within* a row
	bool elementsArePacked() const;

	// returns true if there is no space between adjacent rows
	// (if rowPitchBytes() == width() * strideBytes())
	bool rowsArePacked() const;

	// returns true if elementsArePacked() && rowsArePacked()
	// also known as "linear"
	bool packed() const;

private:

	int m_width;
	int m_height;
	int m_strideBytes;
	int m_rowPitchBytes;
	ubyte* m_pPointer;
};

#include "Array2DView.inl"
