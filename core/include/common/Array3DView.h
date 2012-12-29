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

	// TODO: elementsArePacked(), rowsArePacked(), slicesArePacked(), isLinear() for completely packed

	// create an Array3DView with the default row pitch of width * sizeof( T )
	// and slice pitch of rowPitch * depth
	Array3DView( int width, int height, int depth, void* pPointer );

	Array3DView( int width, int height, int depth,
		int rowPitchBytes, int slicePitchBytes, void* pPointer );

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

	int width() const;
	int height() const;
	int depth() const;
	int rowPitchBytes() const;
	int slicePitchBytes() const;

private:

	int m_width;
	int m_height;
	int m_depth;
	int m_rowPitchBytes;
	int m_slicePitchBytes;
	ubyte* m_pPointer;
};

#include "Array3DView.inl"