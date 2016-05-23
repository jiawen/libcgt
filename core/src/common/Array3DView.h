#pragma once

#include <cstdint>

#include "Array1DView.h"
#include "Array2DView.h"
#include "not_const.h"
#include "WrapConstPointerT.h"
#include "math/Indexing.h"
#include "vecmath/Vector2i.h"
#include "vecmath/Vector3i.h"

// A 3D array view that wraps around a raw pointer and does not take ownership.
template< typename T >
class Array3DView
{
public:

    // T --> void*. const T --> const void*.
    typedef typename WrapConstPointerT< T, void >::pointer VoidPointer;
    // T --> uint8_t*. const T --> const uint8_t*.
    typedef typename WrapConstPointerT< T, uint8_t >::pointer UInt8Pointer;

    // The null Array2DView:
    // pointer = nullptr, width = height = depth = 0.
    Array3DView() = default;

    // Create an Array3DView with:
    // the default element stride of sizeof( T ),
    // the default row stride of width * sizeof( T ),
    // and the default slice stride of width * height * sizeof( T ).
    Array3DView( VoidPointer pointer, const Vector3i& size );

    // Create an Array3DView with specified
    // size { x, y, z } in elements
    // and stride { elementStride, rowStride, sliceStride } in bytes.
    Array3DView( VoidPointer pointer, const Vector3i& size, const Vector3i& stride );

    bool isNull() const;
    bool notNull() const;
    void setNull();

    template< typename U = T,
        typename = typename std::enable_if
            < libcgt::core::common::not_const< U >::value >::type >
    operator const T* () const;

    operator T* ();

    template< typename U = T,
        typename = typename std::enable_if
            < libcgt::core::common::not_const< U >::value >::type >
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
    Vector3i stride() const;

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

    // Implicit conversion operator from Array2DView< T > to Array2DView< const T >.
    // Enabled only if T is not const.
    template< typename U = T,
        typename = typename std::enable_if
            < libcgt::core::common::not_const< U >::value >::type >
    operator Array3DView< const T >() const;

    // Extract 1D slice at a given x and y coordinate.
    Array1DView< T > xySlice( int x, int y );

    // Extract 1D slice at a given y and z coordinate.
    Array1DView< T > yzSlice( int y, int z );

    // Extract 1D slice at a given x and z coordinate.
    Array1DView< T > xzSlice( int x, int z );

    // Extract 2D slice at a given x coordinate.
    Array2DView< T > xSlice( int x );

    // Extract 2D slice at a given y coordinate.
    Array2DView< T > ySlice( int y );

    // Extract 2D slice at a given z coordinate.
    Array2DView< T > zSlice( int z );

private:

    Vector3i m_size;
    Vector3i m_stride;
    UInt8Pointer m_pointer = nullptr;
};

#include "Array3DView.inl"
