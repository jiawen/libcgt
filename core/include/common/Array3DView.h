#pragma once

#include "math/Indexing.h"
#include "vecmath/Vector3i.h"

// a 3D array view that wraps around a raw pointer
// and does not take ownership

template< typename T >
class Array3DView
{
public:

	// pointer = nullptr, width = height = depth = 0
	Array3DView();

	// create an Array3DView with
	// the default element stride of sizeof( T ),
	// the default row stride of width * sizeof( T ),
	// and the default slice pitch of width * height * sizeof( T )
	Array3DView( void* pPointer, int width, int height, int depth );
	Array3DView( void* pPointer, const Vector3i& size );

	// create an Array3DView with specified size, stride, and pitches
	Array3DView( void* pPointer,
		int width, int height, int depth,
		int elementStrideBytes, int rowStrideBytes, int sliceStrideBytes );
	Array3DView( void* pPointer,
		const Vector3i& size,
		int elementStrideBytes, int rowStrideBytes, int sliceStrideBytes );

	bool isNull() const;
	bool notNull() const;

	operator const T* () const;
	operator T* ();

	const T* pointer() const;
	T* pointer();

	const T* elementPointer( int x, int y, int z ) const;
	T* elementPointer( int x, int y, int z );

	const T* rowPointer( int y, int z ) const;
	T* rowPointer( int y, int z );

	const T* slicePointer( int z ) const;
	T* slicePointer( int z );

	const T& operator [] ( int k ) const; // read
	T& operator [] ( int k ); // write

	const T& operator () ( int x, int y, int z ) const; // read
	T& operator () ( int x, int y, int z ); // write

	const T& operator [] ( const Vector3i& xyz ) const; // read
	T& operator [] ( const Vector3i& xyz ); // write

	// the logical size of the array view
	// (i.e., how many elements of type T there are)
	int width() const;
	int height() const;
	int depth() const;
	Vector3i size() const;
	int numElements() const;

	// How many bytes this view would occupy if it were packed.
	// Equal to numElements() * sizeof( T ).
	size_t bytesReferenced() const;

	// how many bytes does this view span:
	// the total number of bytes in a rectangular region
	// that view overlaps, including the empty spaces.
	// Equal to abs(sliceStrideBytes()) * depth()
	size_t bytesSpanned() const;

	// the space between the start of elements in bytes
	int elementStrideBytes() const;

	// the space between the start of rows in bytes
	int rowStrideBytes() const;

	// the space between the start of slices in bytes
	int sliceStrideBytes() const;

	// returns true if there is no space between adjacent elements *within* a row
	bool elementsArePacked() const;

	// returns true if there is no space between adjacent rows
	// (if rowStrideBytes() == width() * elementStrideBytes())
	bool rowsArePacked() const;

	// returns true if there is no space between adjacent slices
	// (if sliceStrideBytes() == height() * rowStrideBytes())
	bool slicesArePacked() const;

	// returns true if elementsArePacked() && rowsArePacked() && slicesArePacked()
	// also known as "linear"
	bool packed() const;

private:

	int m_width;
	int m_height;
	int m_depth;
	int m_elementStrideBytes;
	int m_rowStrideBytes;
	int m_sliceStrideBytes;
	uint8_t* m_pPointer;
};

#include "Array3DView.inl"