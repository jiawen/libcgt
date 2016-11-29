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

    // Extract row y as a 1D view.
    Array1DReadView< T > row( int y );

    // Extract column x as a 1D view.
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

    // Extract row y as a 1D view.
    Array1DWriteView< T > row( int y );

    // Extract column x as a 1D view.
    Array1DWriteView< T > column( int x );

private:

    uint8_t* m_write_pointer = nullptr;
};

// A 3D array view that wraps around a raw pointer and does not take ownership.
template< typename T >
class Array3DReadView
{
public:

    // The null Array3DReadView:
    // pointer = nullptr, width = height = 0.
    Array3DReadView() = default;

    // Create an Array3DReadView with:
    // the default element stride of sizeof( T )
    // and the default row stride of width * sizeof( T ).
    Array3DReadView( const void* pointer, const Vector3i& size );

    // Create an Array3DReadView with specified
    // size { x, y } in elements
    // and strides { elementStride, rowStride } in bytes.
    Array3DReadView( const void* pointer, const Vector3i& size,
        const Vector3i& stride );

    bool isNull() const;
    bool notNull() const;
    void setNull();

    operator const T* () const;
    const T* pointer() const;

    const T* elementPointer( const Vector3i& xyz ) const;
    const T* rowPointer( const Vector2i& yz ) const;
    const T* slicePointer( int z ) const;

    const T& operator [] ( int k ) const;
    const T& operator [] ( const Vector3i& xy ) const;

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

    // Returns true if there is no space between adjacent slices,
    // i.e., if sliceStrideBytes() == height() * rowStrideBytes().
    //
    // Note that null views are not packed.
    bool slicesArePacked() const;

    // Returns true if elementsArePacked() && rowsArePacked() &&
    // slicesArePacked(). Also known as "linear".
    //
    // Note that null views are not packed.
    bool packed() const;

    // Extract 1D slice at a given x and y coordinate.
    Array1DReadView< T > xySlice( const Vector2i& xy );

    // Extract 1D slice at a given y and z coordinate.
    Array1DReadView< T > yzSlice( const Vector2i& yz );

    // Extract 1D slice at a given x and z coordinate.
    Array1DReadView< T > xzSlice( const Vector2i& xz );

    // Extract 2D slice at a given x coordinate.
    Array2DReadView< T > xSlice( int x );

    // Extract 2D slice at a given y coordinate.
    Array2DReadView< T > ySlice( int y );

    // Extract 2D slice at a given z coordinate.
    Array2DReadView< T > zSlice( int z );

private:

    Vector3i m_size;
    Vector3i m_stride;
    const uint8_t* m_pointer = nullptr;
};

template< typename T >
class Array3DWriteView : public Array3DReadView< T >
{
public:

    // The null Array3DReadView:
    // pointer = nullptr, width = height = 0.
    Array3DWriteView() = default;

    // Create an Array3DReadView with:
    // the default element stride of sizeof( T )
    // and the default row stride of width * sizeof( T ).
    Array3DWriteView( void* pointer, const Vector3i& size );

    // Create an Array3DReadView with specified
    // size { x, y } in elements
    // and strides { elementStride, rowStride } in bytes.
    Array3DWriteView( void* pointer, const Vector3i& size,
        const Vector3i& stride );

    operator T* () const;
    T* pointer() const;

    T* elementPointer( const Vector3i& xyz ) const;
    T* rowPointer( const Vector2i& yz ) const;
    T* slicePointer( int z ) const;

    T& operator [] ( int k ) const;
    T& operator [] ( const Vector3i& xy ) const;

    // Extract 1D slice at a given x and y coordinate.
    Array1DWriteView< T > xySlice( const Vector2i& xy );

    // Extract 1D slice at a given y and z coordinate.
    Array1DWriteView< T > yzSlice( const Vector2i& yz );

    // Extract 1D slice at a given x and z coordinate.
    Array1DWriteView< T > xzSlice( const Vector2i& xz );

    // Extract 2D slice at a given x coordinate.
    Array2DWriteView< T > xSlice( int x );

    // Extract 2D slice at a given y coordinate.
    Array2DWriteView< T > ySlice( int y );

    // Extract 2D slice at a given z coordinate.
    Array2DWriteView< T > zSlice( int z );

private:

    uint8_t* m_write_pointer = nullptr;
};

#include "ArrayView.inl"
