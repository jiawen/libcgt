#pragma once

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <memory>

#include "common/Array1DView.h"
#include "common/NewDeleteAllocator.h"

// A simple 1D array class.
template< typename T >
class Array1D
{
public:

    // Construct a null Array1D.
    Array1D() = default;

    // Construct an Array1D from initialized values.
    Array1D( std::initializer_list< T > values,
        Allocator* allocator = NewDeleteAllocator::instance() );

    // Allocate a new array with a given size, element stride, and fill value.
    // Preconditions:
    //   stride >= sizeof( T )
    Array1D( size_t size, size_t stride = sizeof( T ), const T& fillValue = T(),
        Allocator* allocator = NewDeleteAllocator::instance() );

    // Take ownership of an externally allocated array with the given size,
    // stride, and optional allocator.
    Array1D( void* pointer, size_t size, size_t stride = sizeof( T ),
        Allocator* allocator = NewDeleteAllocator::instance() );

    Array1D( const Array1D< T >& copy );
    Array1D( Array1D< T >&& move );
    Array1D& operator = ( const Array1D< T >& copy );
    Array1D& operator = ( Array1D< T >&& move );

    virtual ~Array1D();

    bool isNull() const;
    bool notNull() const;

    // Relinquish the underlying pointer by invalidating this Array2D and
    // returning the pointer and allocator. This object continues to point
    // to the existing allocator.
    std::pair< Array1DView< T >, Allocator* > relinquish();

    // Makes this array null and frees the underlying memory.
    // Dimensions are set to 0.
    void invalidate();

    // The logical number of elements in this array.
    size_t width() const;
    size_t size() const;
    size_t numElements() const;

    // The space between the start of elements in bytes.
    size_t elementStrideBytes() const;
    size_t stride() const;

    void fill( const T& fillValue );

    // Resizes the array while keeping the old stride.
    // This destroys the original data if the new size * stride does not equal
    // the original.
    // If any size or stride is 0, the array is set to null.
    void resize( size_t size );

    // Resizes the array, destroying the original data if the new size * stride
    // does not equal the original.
    // If any size or stride is 0, the array is set to null.
    void resize( size_t size, size_t elementStrideBytes );

    // Get a pointer to the first element.
    const T* pointer() const;
    T* pointer();

    const T* elementPointer( size_t x ) const;
    T* elementPointer( size_t x );

    Array1DView< const T > readView() const;
    Array1DView< T > writeView() const;

    operator Array1DView< const T >() const;
    operator Array1DView< T >();

    operator const T* () const;
    operator T* ();

    const T& operator [] ( int k ) const; // read
    T& operator [] ( int k ); // write

    const T& operator [] ( size_t k ) const; // read
    T& operator [] ( size_t k ); // write

private:

    size_t m_size = 0;
    size_t m_stride = 0;
    uint8_t* m_data = nullptr;
    Allocator* m_allocator = NewDeleteAllocator::instance();
};

#include "Array1D.inl"
