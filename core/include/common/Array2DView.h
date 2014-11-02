#pragma once

#include <cstdint>

#include "math/Indexing.h"
#include "vecmath/Vector2i.h"

// a 2D array view that wraps around a raw pointer
// and does not take ownership

template< typename T >
class Array2DView
{
public:

	// pointer = nullptr, width = height = 0
	Array2DView();

	// create an Array2DView with
	// the default stride of sizeof( T ),
	// and the default row pitch of width * sizeof( T )	
	Array2DView( void* pPointer, int width, int height );
	Array2DView( void* pPointer, const Vector2i& size );

	// create an Array2DView with specified sizes and strides
	Array2DView( void* pPointer, int width, int height, int elementStrideBytes, int rowStrideBytes );
	Array2DView( void* pPointer, const Vector2i& size, int elementStrideBytes, int rowStrideBytes );	

	bool isNull() const;
	bool notNull() const;

	operator const T* () const;
	operator T* ();

	const T* pointer() const;
	T* pointer();

	const T* elementPointer( int x, int y ) const;
	T* elementPointer( int x, int y );

	const T* rowPointer( int y ) const;
	T* rowPointer( int y );

	const T& operator [] ( int k ) const; // read
	T& operator [] ( int k ); // write

	const T& operator () ( int x, int y ) const; // read
	T& operator () ( int x, int y ); // write

	const T& operator [] ( const Vector2i& xy ) const; // read
	T& operator [] ( const Vector2i& xy ); // write

	// The logical size of this view,
	// (i.e., how many elements of type T there are).
	int width() const;
	int height() const;	
	Vector2i size() const;
	int numElements() const;

	// How many bytes this view would occupy if it were packed.
	// Equal to numElements() * sizeof( T ).
	size_t bytesReferenced() const;

	// how many bytes does this view span:
	// the total number of bytes in a rectangular region
	// that view overlaps, including the empty spaces.
	// Equal to abs(rowStrideBytes()) * height()
	size_t bytesSpanned() const;

	// the space between the start of elements in bytes
	int elementStrideBytes() const;

	// the space between the start of rows in bytes
	int rowStrideBytes() const;

	// Returns true if there is no space between adjacent elements *within* a row,
	// i.e., if elementStrideBytes() == sizeof( T ).
	bool elementsArePacked() const;

	// Returns true if there is no space between adjacent rows,
	// i.e., if rowStrideBytes() == width() * elementStrideBytes().
	bool rowsArePacked() const;

	// returns true if elementsArePacked() && rowsArePacked()
	// also known as "linear"
	bool packed() const;

private:

	int m_width;
	int m_height;
	int m_elementStrideBytes;
	int m_rowStrideBytes;
	uint8_t* m_pPointer;
};

#include "Array2DView.inl"
