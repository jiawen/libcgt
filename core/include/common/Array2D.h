#pragma once

#include <cstdio>
#include <cstdint>
#include <cstring>

#include "common/Array2DView.h"
#include "math/Indexing.h"
#include "vecmath/Vector2i.h"

// TODO: strides should be size_t, so always positive

// A simple 2D array class (with row-major storage)
template< typename T >
class Array2D
{
public:

    // Default null array with dimensions 0 and no data allocated.
    Array2D();

    // Takes ownership of pointer and views it as a 2D array with strides.
    // All sizes and strides must be positive.
    Array2D( void* pointer, const Vector2i& size );
    Array2D( void* pointer, const Vector2i& size, const Vector2i& strides );

    // Read file from disk.
    Array2D( const char* filename );

    // All sizes and strides must be positive.
    Array2D( const Vector2i& size, const T& fillValue = T() );
    Array2D( const Vector2i& size, const Vector2i& strides, const T& fillValue = T() );

    Array2D( const Array2D< T >& copy );
    Array2D( Array2D< T >&& move );
    Array2D& operator = ( const Array2D< T >& copy );
    Array2D& operator = ( Array2D< T >&& move );
    virtual ~Array2D();

    bool isNull() const;
    bool notNull() const;

    // Makes this array null *without* freeing the underlying memory: it is returned instead.
    // Dimensions are set to 0.
    Array2DView< T > relinquish();

    // Makes this array null and frees the underlying memory.
    // Dimensions are set to 0.
    void invalidate();

    int width() const;
    int height() const;
    Vector2i size() const;
    int numElements() const;

    // The space between the start of elements in bytes.
    int elementStrideBytes() const;

    // The space between the start of rows in bytes.
    int rowStrideBytes() const;

    // { elementStride, rowStride } in bytes.
    Vector2i strides() const;

    void fill( const T& fillValue );

    // Resizes the array, freeing the original data.
    // If width or height <= 0, the array is invalidated
    void resize( const Vector2i& size );
    void resize( const Vector2i& size, const Vector2i& strides );

    // Get a pointer to the first element.
    const T* pointer() const;
    T* pointer();

    const T* elementPointer( const Vector2i& xy ) const;
    T* elementPointer( const Vector2i& xy );

    const T* rowPointer( int y ) const;
    T* rowPointer( int y );

    operator Array2DView< const T >() const;
    operator Array2DView< T >();

    operator const T* () const;
    operator T* ();

    const T& operator [] ( int k ) const; // read
    T& operator [] ( int k ); // write

    const T& operator [] ( const Vector2i& xy ) const; // read
    T& operator [] ( const Vector2i& xy ); // write

    // only works if T doesn't have pointers, with sizeof() well defined
    bool load( const char* filename );
    bool load( FILE* fp );

    // only works if T doesn't have pointers, with sizeof() well defined
    bool save( const char* filename ) const;
    bool save( FILE* fp ) const;

private:

    Vector2i m_size;
    Vector2i m_strides;
    uint8_t* m_array;
};

#include "Array2D.inl"
