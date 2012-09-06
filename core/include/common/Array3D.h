#pragma once

#include <cstdio>

#include "common/BasicTypes.h"
#include <vecmath/Vector3i.h>
#include <math/Indexing.h>

// TODO: switch to using std::vector as underlying representation

// A simple 3D array class (with row-major storage)
template< typename T >
class Array3D
{
public:	

	// Default null array with dimensions -1 and no data allocated
	Array3D();
	Array3D( const char* filename );
	Array3D( int width, int height, int depth, const T& fill = T() );
	Array3D( const Array3D< T >& copy );
	Array3D( Array3D< T >&& move );
	Array3D& operator = ( const Array3D< T >& copy );
	Array3D& operator = ( Array3D< T >&& move );
	virtual ~Array3D();
	
	bool isNull() const;
	bool notNull() const;
	void invalidate();

	int width() const;
	int height() const;
	int depth() const;
	Vector3i size() const;
	int numElements() const;

	void fill( const T& val );

	// resizing with width, height, or depth <= 0 will invalidate this array
	void resize( int width, int height, int depth );
	void resize( const Vector3i& size );

	// Returns a pointer to the beginning of the y-th row of the z-th slice
	T* rowPointer( int y, int z );
	const T* rowPointer( int y, int z ) const;

	// Returns a pointer to the beginning of the z-th slice
	T* slicePointer( int z );
	const T* slicePointer( int z ) const;

	operator T* () const;

	const T& operator () ( int k ) const; // read
	T& operator () ( int k ); // write

	const T& operator () ( int x, int y, int z ) const; // read
	T& operator () ( int x, int y, int z ); // write

	// reinterprets this array as an array of another format,
	// destroying this array
	//
	// by default (outputWidth and outputHeight = -1)
	// the output width is width() * sizeof( T ) / sizeof( S )
	// (i.e., a 3 x 4 x 5 float4 gets cast to a 12 x 4 x x 5 float1)
	//
	// If the source is null or the desired output size is invalid
	// returns the null array.
	template< typename S >
	Array3D< S > reinterpretAs( int outputWidth = -1, int outputHeight = -1, int outputDepth = -1 );

	// only works if T doesn't have pointers, with sizeof() well defined
	bool load( const char* filename );

	// only works if T doesn't have pointers, with sizeof() well defined
	bool save( const char* filename );

private:
	
	int m_width;
	int m_height;
	int m_depth;
	T* m_array;

	// to allow reinterpretAs< S >
	template< typename S >
	friend class Array3D;
};

#include "Array3D.inl"
