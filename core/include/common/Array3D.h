#pragma once

#include <cstdio>

#include "common/Array3DView.h"
#include "common/BasicTypes.h"
#include "vecmath/Vector3i.h"
#include "math/Indexing.h"

// TODO: implement strided/pitch/slicepitch constructor
// TODO: sizeInBytes, pitch, etc should be size_t

// A simple 3D array class (with row-major storage)
template< typename T >
class Array3D
{
public:	

	// Default null array with dimensions -1 and no data allocated
	Array3D();
	Array3D( const char* filename );
	Array3D( int width, int height, int depth, const T& fillValue = T() );
	Array3D( const Vector3i& size, const T& fillValue = T() );
	Array3D( const Array3D< T >& copy );
	Array3D( Array3D< T >&& move );
	Array3D& operator = ( const Array3D< T >& copy );
	Array3D& operator = ( Array3D< T >&& move );
	virtual ~Array3D();
	
	bool isNull() const;
	bool notNull() const;
	void invalidate(); // makes this array null by setting its dimensions to -1 and frees the data

	int width() const;
	int height() const;
	int depth() const;
	Vector3i size() const;
	int numElements() const;
	int strideBytes() const; // number of bytes between successive elements
	int rowPitchBytes() const; // number of bytes between successive rows on any slice
	int slicePitchBytes() const; // number of bytes between successive slices

	void fill( const T& fillValue );

	// resizes the array, original data is not preserved
	// if width, height, or depth <= 0, the array is invalidated
	void resize( int width, int height, int depth );
	void resize( const Vector3i& size );	

	operator const Array3DView< T >() const;
	operator Array3DView< T >();

	// Returns a pointer to the beginning of the y-th row of the z-th slice
	const T* rowPointer( int y, int z ) const;
	T* rowPointer( int y, int z );

	// Returns a pointer to the beginning of the z-th slice
	const T* slicePointer( int z ) const;
	T* slicePointer( int z );

	operator const T* () const;
	operator T* ();

	const T& operator [] ( int k ) const; // read
	T& operator [] ( int k ); // write

	const T& operator () ( int x, int y, int z ) const; // read
	T& operator () ( int x, int y, int z ); // write

	const T& operator [] ( const Vector3i& xyz ) const; // read
	T& operator [] ( const Vector3i& xyz ); // write

	// reinterprets this array as an array of another format,
	// destroying this array
	//
	// by default (outputWidth, outputHeight and outputDepth = -1)
	// the output width is width() * sizeof( T ) / sizeof( S )
	// (i.e., a 3 x 4 x 5 float4 gets cast to a 12 x 4 x x 5 float1)
	//
	// If the source is null or the desired output size is invalid
	// returns the null array.
	template< typename S >
	Array3D< S > reinterpretAs( int outputWidth = -1, int outputHeight = -1, int outputDepth = -1,
		int outputRowPitchBytes = -1, int outputSlicePitchBytes = -1 );

	// only works if T doesn't have pointers, with sizeof() well defined
	bool load( const char* filename );
	bool load( FILE* fp );

	// only works if T doesn't have pointers, with sizeof() well defined
	bool save( const char* filename );
	bool save( FILE* fp );

private:
	
	int m_width;
	int m_height;
	int m_depth;
	int m_strideBytes;
	int m_rowPitchBytes;
	int m_slicePitchBytes;
	T* m_array;

	// to allow reinterpretAs< S >
	template< typename S >
	friend class Array3D;
};

#include "Array3D.inl"
