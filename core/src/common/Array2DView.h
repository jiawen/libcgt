#pragma once

#include <cstdint>

#include "Array1DView.h"
#include "not_const.h"
#include "WrapConstPointerT.h"
#include "math/Indexing.h"
#include "vecmath/Vector2i.h"

// A 2D array view that wraps around a raw pointer and does not take ownership.
template< typename T >
class Array2DView
{
public:

    // The null Array2DView:
    // pointer = nullptr, width = height = 0.
    Array2DView() = default;

    // Create an Array2DView with:
    // the default element stride of sizeof( T )
    // and the default row stride of width * sizeof( T ).
    Array2DView( void* pointer, const Vector2i& size );

    // Create an Array2DView with specified
    // size { x, y } in elements
    // and strides { elementStride, rowStride } in bytes.
    Array2DView( void* pointer, const Vector2i& size, const Vector2i& strides );

    bool isNull() const;
    bool notNull() const;

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

    T* elementPointer( const Vector2i& xy );
    T* rowPointer( int y );

    T& operator [] ( int k );
    T& operator [] ( const Vector2i& xy );

    // The logical size of the array view
    // (i.e., how many elements of type T there are).
    int width() const;
    int height() const;
    Vector2i size() const;
    int numElements() const;

    // The space between the start of elements in bytes.
    int elementStrideBytes() const;

    // The space between the start of rows in bytes.
    int rowStrideBytes() const;

    // { elementStride, rowStride } in bytes.
    Vector2i stride() const;

    // Returns true if there is no space between adjacent elements *within* a row.
    bool elementsArePacked() const;

    // Returns true if there is no space between adjacent rows,
    // i.e., if rowStrideBytes() == width() * elementStrideBytes().
    bool rowsArePacked() const;

    // Returns true if elementsArePacked() && rowsArePacked(),
    // also known as "linear".
    bool packed() const;

    // Implicit conversion operator from Array2DView< T > to Array2DView< const T >.
    // Enabled only if T is not const.
    // TODO: can remove fully qualified namespace after this class is moved into
    // the libcgt namespace.
    template< typename U = T,
        typename = typename std::enable_if
            < libcgt::core::common::not_const< U >::value >::type >
    operator Array2DView< const T >() const;

    // Extract row y from this Array2DView as an Array1DView.
    Array1DView< T > row( int y );

    // Extract column x from this Array2DView as an Array1DView.
    Array1DView< T > column( int x );

private:

    Vector2i m_size;
    Vector2i m_stride;
    typename WrapConstPointerT< T, uint8_t >::pointer m_pointer = nullptr;
};

#include "Array2DView.inl"
