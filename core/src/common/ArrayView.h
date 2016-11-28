#pragma once

#include <cstdint>

#include "math/Indexing.h"
#include "vecmath/Vector2i.h"

// A 1D array view that wraps around a raw pointer and does not take ownership.
template< typename T >
class Array1DReadView
{
public:

    // The null Array1DReadView:
    // pointer = nullptr, width = height = 0.
    Array1DReadView() = default;

    // Create an Array1DReadView with:
    // the default element stride of sizeof( T )
    // and the default stride of sizeof( T ).
    Array1DReadView( const void* pointer, size_t size );

    // Create an Array1DReadView with specified size in elements and stride in
    // bytes.
    Array1DReadView( const void* pointer, size_t size,
        std::ptrdiff_t stride );

    bool isNull() const;
    bool notNull() const;
    void setNull();

    operator const T* () const;
    const T* pointer() const;

    const T* elementPointer( size_t x ) const;

    const T& operator [] ( size_t x ) const;

    // The logical size of the array view
    // (i.e., how many elements of type T there are).
    size_t width() const;
    size_t size() const;
    size_t numElements() const;

    // The space between the start of elements, in bytes.
    // For a 1D view, stride and elementStrideBytes are equivalent.
    std::ptrdiff_t elementStrideBytes() const;
    std::ptrdiff_t stride() const;

    // Returns true if there is no space between adjacent elements.
    // Note that null views are not packed.
    bool elementsArePacked() const;
    bool packed() const;

private:

    size_t m_size;
    std::ptrdiff_t m_stride;
    const uint8_t* m_pointer = nullptr;
};

template< typename T >
class Array1DWriteView : public Array1DReadView< T >
{
public:

    // The null Array1DReadView:
    // pointer = nullptr, width = height = 0.
    Array1DWriteView() = default;

    // Create an Array1DReadView with:
    // the default element stride of sizeof( T )
    // and the default row stride of width * sizeof( T ).
    Array1DWriteView( void* pointer, size_t size );

    // Create an Array1DReadView with specified
    // size { x, y } in elements
    // and strides { elementStride, rowStride } in bytes.
    Array1DWriteView( void* pointer, size_t size,
        std::ptrdiff_t stride );

    operator T* () const;
    T* pointer() const;

    T* elementPointer( size_t x ) const;

    T& operator [] ( size_t x ) const;

private:

    uint8_t* m_write_pointer = nullptr;
};

// A 2D array view that wraps around a raw pointer and does not take ownership.
template< typename T >
class Array2DReadView
{
public:

    // The null Array2DReadView:
    // pointer = nullptr, width = height = 0.
    Array2DReadView() = default;

    // Create an Array2DReadView with:
    // the default element stride of sizeof( T )
    // and the default row stride of width * sizeof( T ).
    Array2DReadView( const void* pointer, const Vector2i& size );

    // Create an Array2DReadView with specified
    // size { x, y } in elements
    // and strides { elementStride, rowStride } in bytes.
    Array2DReadView( const void* pointer, const Vector2i& size,
        const Vector2i& stride );

    bool isNull() const;
    bool notNull() const;
    void setNull();

    operator const T* () const;
    const T* pointer() const;

    const T* elementPointer( const Vector2i& xy ) const;
    const T* rowPointer( int y ) const;

    const T& operator [] ( int k ) const;
    const T& operator [] ( const Vector2i& xy ) const;

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

    // Returns true if there is no space between adjacent elements *within* a
    // row.
    //
    // Note that null views are not packed.
    bool elementsArePacked() const;

    // Returns true if there is no space between adjacent rows,
    // i.e., if rowStrideBytes() == width() * elementStrideBytes().
    //
    // Note that null views are not packed.
    bool rowsArePacked() const;

    // Returns true if elementsArePacked() && rowsArePacked(),
    // also known as "linear".
    //
    // Note that null views are not packed.
    bool packed() const;

    // Extract row y from this Array2DReadView as an Array1DView.
    Array1DReadView< T > row( int y );

    // Extract column x from this Array2DReadView as an Array1DView.
    Array1DReadView< T > column( int x );

private:

    Vector2i m_size;
    Vector2i m_stride;
    const uint8_t* m_pointer = nullptr;
};

template< typename T >
class Array2DWriteView : public Array2DReadView< T >
{
public:

    // The null Array2DReadView:
    // pointer = nullptr, width = height = 0.
    Array2DWriteView() = default;

    // Create an Array2DReadView with:
    // the default element stride of sizeof( T )
    // and the default row stride of width * sizeof( T ).
    Array2DWriteView( void* pointer, const Vector2i& size );

    // Create an Array2DReadView with specified
    // size { x, y } in elements
    // and strides { elementStride, rowStride } in bytes.
    Array2DWriteView( void* pointer, const Vector2i& size,
        const Vector2i& stride );

    operator T* () const;
    T* pointer() const;

    T* elementPointer( const Vector2i& xy ) const;
    T* rowPointer( int y ) const;

    T& operator [] ( int k ) const;
    T& operator [] ( const Vector2i& xy ) const;

    // Extract row y from this Array2DReadView as an Array1DView.
    Array1DWriteView< T > row( int y );

    // Extract column x from this Array2DReadView as an Array1DView.
    Array1DWriteView< T > column( int x );

private:

    uint8_t* m_write_pointer = nullptr;
};

#include "ArrayView.inl"
