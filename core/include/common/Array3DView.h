#pragma once

#include <cstdint>

#include "common/WrapConstPointerT.h"
#include "math/Indexing.h"
#include "vecmath/Vector2i.h"
#include "vecmath/Vector3i.h"

// a 3D array view that wraps around a raw pointer
// and does not take ownership

template< typename T >
class Array3DView
{
public:

    // The null Array2DView:
    // pointer = nullptr, width = height = depth = 0.
    Array3DView();

    // Create an Array3DView with:
    // the default element stride of sizeof( T ),
    // the default row stride of width * sizeof( T ),
    // and the default slice pitch of width * height * sizeof( T ).
    Array3DView( void* pPointer, const Vector3i& size );

    // Create an Array3DView with specified
    // size { x, y, z } in elements
    // and strides { elementStride, rowStride, sliceStride } in bytes.
    Array3DView( void* pPointer, const Vector3i& size, const Vector3i& strides );

    bool isNull() const;
    bool notNull() const;

    operator const T* () const;
    operator T* ();

    const T* pointer() const;
    T* pointer();

    T* elementPointer( const Vector3i& xyz );
    T* rowPointer( const Vector2i& yz );
    T* slicePointer( int z );

    T& operator [] ( int k );
    T& operator [] ( const Vector3i& xyz );

    // The logical size of the array view
    // (i.e., how many elements of type T there are).
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

    // Returns true if there is no space between adjacent elements *within* a row
    bool elementsArePacked() const;

    // Returns true if there is no space between adjacent rows,
    // i.e., if rowStrideBytes() == width() * elementStrideBytes().
    bool rowsArePacked() const;

    // Returns true if there is no space between adjacent slices,
    // i.e., if sliceStrideBytes() == height() * rowStrideBytes().
    bool slicesArePacked() const;

    // Returns true if elementsArePacked() && rowsArePacked() && slicesArePacked(),
    // also known as "linear".
    bool packed() const;

private:

    Vector3i m_size;
    Vector3i m_strides;
    typename WrapConstPointerT< T, uint8_t >::pointer m_pPointer;
};

#include "Array3DView.inl"