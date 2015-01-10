#pragma once

#include <cstdio>
#include <cstring>

#include "common/Array3DView.h"
#include "common/BasicTypes.h"
#include "vecmath/Vector3i.h"
#include "math/Indexing.h"

// TODO: strides should be size_t, so always positive

// A simple 3D array class (with row-major storage)
template< typename T >
class Array3D
{
public:	

	// Default null array with dimensions 0 and no data allocated.
	Array3D();

    // Takes ownership of pointer and views it as a 2D array with strides.
    // All sizes and strides must be positive.
    Array3D( void* pointer, const Vector3i& size );
    Array3D( void* pointer, const Vector3i& size, const Vector3i& strides );

	Array3D( const char* filename );

    // All sizes and strides must be positive.
	Array3D( const Vector3i& size, const T& fillValue = T() );
    Array3D( const Vector3i& size, const Vector3i& strides, const T& fillValue = T() );
	
    Array3D( const Array3D< T >& copy );
	Array3D( Array3D< T >&& move );
	Array3D& operator = ( const Array3D< T >& copy );
	Array3D& operator = ( Array3D< T >&& move );
	virtual ~Array3D();
	
	bool isNull() const;
	bool notNull() const;

    // Makes this array null *without* freeing the underlying memory: it is returned instead.
    // Dimensions are set to 0.
    Array3DView< T > relinquish();

    // Makes this array null and frees the underlying memory.
    // Dimensions are set to 0.
    void invalidate();

	int width() const;
	int height() const;
	int depth() const;
	Vector3i size() const;
	int numElements() const;

    // The space between the start of elements in bytes.
    int elementStrideBytes() const;

    // The space between the start of rows in bytes.
    int rowStrideBytes() const;

    // The space between the start of slices in bytes.
    int sliceStrideBytes() const;

    // { elementStride, rowStride, sliceStride } in bytes.
    Vector3i strides() const;

	void fill( const T& fillValue );

	// resizes the array, original data is not preserved
	// if width, height, or depth <= 0, the array is invalidated
    void resize( const Vector3i& size );
    void resize( const Vector3i& size, const Vector3i& strides );

    // Get a pointer to the first element.
    const T* pointer() const;
    T* pointer();

    const T* elementPointer( const Vector3i& xy ) const;
    T* elementPointer( const Vector3i& xy );

	// Returns a pointer to the beginning of the y-th row of the z-th slice
	const T* rowPointer( int y, int z ) const;
	T* rowPointer( int y, int z );

	// Returns a pointer to the beginning of the z-th slice
	const T* slicePointer( int z ) const;
	T* slicePointer( int z );

    operator const Array3DView< const T >() const;
    operator Array3DView< T >();

	operator const T* () const;
	operator T* ();

	const T& operator [] ( int k ) const; // read
	T& operator [] ( int k ); // write

	const T& operator [] ( const Vector3i& xyz ) const; // read
	T& operator [] ( const Vector3i& xyz ); // write

	// only works if T doesn't have pointers, with sizeof() well defined
	bool load( const char* filename );
	bool load( FILE* fp );

	// only works if T doesn't have pointers, with sizeof() well defined
	bool save( const char* filename );
	bool save( FILE* fp );

private:
	
    Vector3i m_size;
    Vector3i m_strides;
    uint8_t* m_array;
};

#include "Array3D.inl"
