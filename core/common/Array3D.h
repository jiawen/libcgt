#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>

#include "ArrayView.h"
#include "NewDeleteAllocator.h"
#include "libcgt/core/math/Indexing.h"
#include "libcgt/core/vecmath/Vector3i.h"

// TODO: use Vector3< size_t > when the new Vector class is ready.
// Then, we can get rid of all the static_casts.

// A simple 3D array class. The first index is the most densely packed.
template< typename T >
class Array3D
{
public:

    // Default null array with dimensions 0 and no data allocated.
    Array3D() = default;

    // Takes ownership of pointer and views it as a 3D array with stride.
    // All sizes and strides must be positive.
    // stride.x >= sizeof( T )
    // stride.y >= size.x * sizeof( T )
    // stride.z >= size.x * size.y * sizeof( T )
    Array3D( void* pointer, const Vector3i& size,
        Allocator* allocator = NewDeleteAllocator::instance() );
    Array3D( void* pointer, const Vector3i& size, const Vector3i& stride,
        Allocator* allocator = NewDeleteAllocator::instance() );

    // All sizes and strides must be positive.
    // stride.x >= sizeof( T )
    // stride.y >= size.x * sizeof( T )
    // stride.z >= size.x * size.y * sizeof( T )
    Array3D( const Vector3i& size, const T& fillValue = T(),
        Allocator* allocator = NewDeleteAllocator::instance() );
    Array3D( const Vector3i& size, const Vector3i& stride,
        const T& fillValue = T(),
        Allocator* allocator = NewDeleteAllocator::instance() );

    Array3D( const Array3D< T >& copy );
    Array3D( Array3D< T >&& move );
    Array3D& operator = ( const Array3D< T >& copy );
    Array3D& operator = ( Array3D< T >&& move );
    virtual ~Array3D();

    bool isNull() const;
    bool notNull() const;

    // Relinquish the underlying pointer by invalidating this Array2D and
    // returning the pointer and allocator. This object continues to point
    // to the existing allocator.
    std::pair< Array3DWriteView< T >, Allocator* > relinquish();

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
    Vector3i stride() const;

    void fill( const T& fillValue );

    // Resizes the array while keeping the old stride.
    // This destroys the original data if the new size * stride does not equal
    // the original.
    // If any size or stride is 0, the array is set to null.
    void resize( const Vector3i& size );

    // Resizes the array, destroying the original data if the new size * stride
    // does not equal the original.
    // If any size or stride is 0, the array is set to null.
    void resize( const Vector3i& size, const Vector3i& stride );

    // Get a pointer to the first element.
    const T* pointer() const;
    T* pointer();

    const T* elementPointer( const Vector3i& xyz ) const;
    T* elementPointer( const Vector3i& xyz );

    // Returns a pointer to the beginning of the y-th row of the z-th slice
    const T* rowPointer( int y, int z ) const;
    T* rowPointer( int y, int z );

    // Returns a pointer to the beginning of the z-th slice
    const T* slicePointer( int z ) const;
    T* slicePointer( int z );

    Array3DReadView< T > readView() const;
    Array3DWriteView< T > writeView() const;

    operator Array3DReadView< T >() const;
    operator Array3DWriteView< T >();

    operator const T* () const;
    operator T* ();

    const T& operator [] ( int k ) const; // read
    T& operator [] ( int k ); // write

    const T& operator [] ( const Vector3i& xyz ) const; // read
    T& operator [] ( const Vector3i& xyz ); // write

private:

    Vector3i m_size = Vector3i{ 0, 0, 0 };
    Vector3i m_stride = Vector3i{ 0, 0, 0 };
    uint8_t* m_data = nullptr;
    Allocator* m_allocator = NewDeleteAllocator::instance();
};

#include "Array3D.inl"
