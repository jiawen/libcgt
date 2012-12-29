#pragma once

#include "common/BasicTypes.h"
#include "math/Indexing.h"
#include "vecmath/Vector3i.h"

// a 3D array view that wraps around a raw pointer
// and does not take ownership

template< typename T >
class Array3DView
{
public:

	// create an Array3DView with
	// the default stride of sizeof( T )
	// the default row pitch of width * sizeof( T )
	// and the default slice pitch of width * height * sizeof( T )
	Array3DView( int width, int height, int depth, void* pPointer );
	Array3DView( const Vector3i& size, void* pPointer );

	// create an Array3DView with specified size, stride, and pitch
	Array3DView( int width, int height, int depth,
		int strideBytes, int rowPitchBytes, int slicePitchBytes, void* pPointer );
	Array3DView( const Vector3i& size,
		int strideBytes, int rowPitchBytes, int slicePitchBytes, void* pPointer );

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

	// the space between the start of elements in bytes
	int strideBytes() const;

	// the space between the start of rows in bytes
	int rowPitchBytes() const;

	// the space between the start of slices in bytes
	int slicePitchBytes() const;

	// returns true if there is no space between adjacent elements *within* a row
	bool elementsArePacked() const;

	// returns true if there is no space between adjacent rows
	// (if rowPitchBytes() == width() * strideBytes())
	bool rowsArePacked() const;

	// returns true if there is no space between adjacent slices
	// (if slicePitchBytes() == height() * rowPitchBytes())
	bool slicesArePacked() const;

	// returns true if elementsArePacked() && rowsArePacked() && slicesArePacked()
	// also known as "linear"
	bool packed() const;

private:

	int m_width;
	int m_height;
	int m_depth;
	int m_strideBytes;
	int m_rowPitchBytes;
	int m_slicePitchBytes;
	ubyte* m_pPointer;
};

#include "Array3DView.inl"