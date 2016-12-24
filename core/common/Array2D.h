#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>

#include "ArrayView.h"
#include "NewDeleteAllocator.h"
#include "libcgt/core/math/Indexing.h"
#include "libcgt/core/vecmath/Vector2i.h"

// TODO: use Vector2< size_t > when the new Vector class is ready.
// Then, we can get rid of all the static_casts.

// A simple 2D array class. The first index is the most densely packed.
template< typename T >
class Array2D
{
public:

    // Default null array with dimensions 0 and no data allocated.
    Array2D() = default;

    // Takes ownership of pointer and views it as a 2D array with stride.
    // All sizes and strides must be positive.
    Array2D( void* pointer, const Vector2i& size,
        Allocator* allocator = NewDeleteAllocator::instance() );

    // Takes ownership of pointer and views it as a 2D array with stride.
    // All sizes and strides must be positive.
    Array2D( void* pointer, const Vector2i& size, const Vector2i& stride,
        Allocator* allocator = NewDeleteAllocator::instance() );

    // All sizes and strides must be positive.
    Array2D( const Vector2i& size, const T& fillValue = T(),
        Allocator* allocator = NewDeleteAllocator::instance() );
    Array2D( const Vector2i& size, const Vector2i& stride,
        const T& fillValue = T(),
        Allocator* allocator = NewDeleteAllocator::instance() );

    Array2D( const Array2D< T >& copy );
    Array2D( Array2D< T >&& move );
    Array2D& operator = ( const Array2D< T >& copy );
    Array2D& operator = ( Array2D< T >&& move );
    virtual ~Array2D();

    bool isNull() const;
    bool notNull() const;

    // Relinquish the underlying pointer by invalidating this Array2D and
    // returning the pointer and allocator. This object continues to point
    // to the existing allocator.
    std::pair< Array2DWriteView< T >, Allocator* > relinquish();

    // Makes this array null and frees the underlying memory.
    // Dimensions are set to 0.
    void invalidate();

    size_t width() const;
    size_t height() const;
    Vector2i size() const;
    size_t numElements() const;

    // The space between the start of elements in bytes.
    size_t elementStrideBytes() const;

    // The space between the start of rows in bytes.
    size_t rowStrideBytes() const;

    // { elementStride, rowStride } in bytes.
    Vector2i stride() const;

    void fill( const T& fillValue );

    // Resizes the array while keeping the old stride.
    // This destroys the original data if the new size * stride does not equal
    // the original.
    // If any size or stride is 0, the array is set to null.
    void resize( const Vector2i& size );

    // Resizes the array, destroying the original data if the new size * stride
    // does not equal the original.
    // If any size or stride is 0, the array is set to null.
    void resize( const Vector2i& size, const Vector2i& stride );

    // Get a pointer to the first element.
    const T* pointer() const;
    T* pointer();

    const T* elementPointer( const Vector2i& xy ) const;
    T* elementPointer( const Vector2i& xy );

    const T* rowPointer( size_t y ) const;
    T* rowPointer( size_t y );

    Array2DReadView< T > readView() const;
    Array2DWriteView< T > writeView();

    operator Array2DReadView< T >() const;
    operator Array2DWriteView< T >();

    operator const T* () const;
    operator T* ();

    const T& operator [] ( size_t k ) const; // read
    T& operator [] ( size_t k ); // write

    const T& operator [] ( const Vector2i& xy ) const; // read
    T& operator [] ( const Vector2i& xy ); // write

private:

    Vector2i m_size = { 0, 0 };
    Vector2i m_stride = { 0, 0 };
    uint8_t* m_data = nullptr;
    Allocator* m_allocator = NewDeleteAllocator::instance();
};

#include "Array2D.inl"
