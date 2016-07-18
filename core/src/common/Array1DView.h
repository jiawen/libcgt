#pragma once

#include <cstdint>

#include "common/not_const.h"
#include "common/WrapConstPointerT.h"

// A 1D array view that wraps around a raw pointer and does not take ownership.
template< typename T >
class Array1DView
{
public:

    // T --> void*. const T --> const void*.
    using VoidPointer = typename WrapConstPointerT< T, void >::pointer;
    // T --> uint8_t*. const T --> const uint8_t*.
    using UInt8Pointer = typename WrapConstPointerT< T, uint8_t >::pointer;

    // The null Array2DView:
    // pointer = nullptr, width = 0.
    Array1DView() = default;

    // Create an Array1DView with the default element stride of sizeof( T ).
    Array1DView( VoidPointer pointer, size_t size );

    // Create an Array1DView with the specified size and element stride.
    Array1DView( VoidPointer pointer, size_t size, ptrdiff_t stride );

    bool isNull() const;
    bool notNull() const;
    void setNull();

    template< typename U = T,
        typename = typename std::enable_if
            < libcgt::core::not_const< U >::value >::type >
    operator const T* () const;

    operator T* ();

    template< typename U = T,
        typename = typename std::enable_if
            < libcgt::core::not_const< U >::value >::type >
    const T* pointer() const;

    T* pointer();

    T* elementPointer( size_t x );

    T& operator [] ( size_t k );

    // The logical size of the array view
    // (i.e., how many elements of type T there are).
    // For a 1D view, width, size, and numElements are all equivalent.
    size_t width() const;
    size_t size() const;
    size_t numElements() const;

    // The space between the start of elements, in bytes.
    // For a 1D view, stride and elementStrideBytes are equivalent.
    ptrdiff_t elementStrideBytes() const;
    ptrdiff_t stride() const;

    // Returns true if the array is tightly packed,
    // i.e. elementStrideBytes() == sizeof( T ).
    bool elementsArePacked() const;
    bool packed() const;

    // Implicit conversion operator from Array1DView< T > to
    // Array1DView< const T >.
    //
    // Enabled only if T is not const.
    template< typename U = T,
        typename = typename std::enable_if
            < libcgt::core::not_const< U >::value >::type >
    operator Array1DView< const T >() const;

private:

    size_t m_size = 0;
    ptrdiff_t m_stride = 0;
    UInt8Pointer m_pointer = nullptr;
};

#include "Array1DView.inl"
