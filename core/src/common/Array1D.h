#pragma once

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <memory>

#include "common/Array1DView.h"

// A simple 1D array class.
template< typename T >
class Array1D
{
public:

    Array1D() = default;
    Array1D( std::initializer_list< T > values );    

    // Read file from disk.
    Array1D( const char* filename );

    // All sizes and strides must be positive.
    Array1D( size_t size, size_t stride = sizeof( T ), const T& fillValue = T() );

    // Take ownership of an array allocated externally.
    // It will be deleted using delete[] on a uint8_t*.
    Array1D( std::unique_ptr< void > pointer, size_t size, size_t stride = sizeof( T ) );

    Array1D( const Array1D< T >& copy );
    Array1D( Array1D< T >&& move ); // TODO(VS2015): = default
    Array1D& operator = ( const Array1D< T >& copy );
    Array1D& operator = ( Array1D< T >&& move ); // TODO(VS2015): = default

    bool isNull() const;
    bool notNull() const;

    // The logical number of elements in this array.
    size_t width() const;
    size_t size() const;
    size_t numElements() const;

    // The space between the start of elements in bytes.
    size_t elementStrideBytes() const;
    size_t stride() const;

    void fill( const T& fillValue );

    // Resizes the array, freeing the original data.
    // If width or height <= 0, the array is invalidated
    void resize( size_t size );
    void resize( size_t size, size_t elementStrideBytes );

    // Get a pointer to the first element.
    const T* pointer() const;
    T* pointer();

    const T* elementPointer( int x ) const;
    T* elementPointer( int x );

    operator Array1DView< const T >() const;
    operator Array1DView< T >();

    operator const T* () const;
    operator T* ();

    const T& operator [] ( int k ) const; // read
    T& operator [] ( int k ); // write

    // only works if T doesn't have pointers, with sizeof() well defined
    bool load( const char* filename );
    bool load( FILE* fp );

    // only works if T doesn't have pointers, with sizeof() well defined
    bool save( const char* filename ) const;
    bool save( FILE* fp ) const;

private:

    size_t m_size = 0;
    size_t m_stride = 0;
    std::unique_ptr< uint8_t[] > m_array;
};

#include "Array1D.inl"
