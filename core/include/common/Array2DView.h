#pragma once

#include "common/BasicTypes.h"
#include "common/WrapConstPointerT.h"
#include "math/Indexing.h"
#include "vecmath/Vector2i.h"

// A 2D array view that wraps around a raw pointer and does not take ownership.
template< typename T >
class Array2DView
{
public:

    // The null Array2DView:
    // pointer = nullptr, width = height = 0
	Array2DView();

	// Create an Array2DView with:
    // the default element stride of sizeof( T )
	// and the default row stride of width * sizeof( T )	
	Array2DView( void* pPointer, const Vector2i& size );

	// Create an Array2DView with specified
    // size { x, y } in elements
    // and strides { elementStride, rowStride } in bytes.
    Array2DView( void* pPointer, const Vector2i& size, const Vector2i& strides );

    bool isNull( ) const;
    bool notNull( ) const;

    operator const T* () const;
    operator T* ();

    const T* pointer() const;
    T* pointer();

    T* elementPointer( const Vector2i& xy );
	T* rowPointer( int y );

	T& operator [] ( int k );
	T& operator [] ( const Vector2i& xy );

	// The logical size of this view,
	// (i.e., how many elements of type T there are).
	int width() const;
	int height() const;	
	Vector2i size() const;
	int numElements() const;

	// How many bytes this view would occupy if it were packed.
	// Equal to numElements() * sizeof( T ).
	size_t bytesReferenced() const;

	// how many bytes does this view span:
	// the total number of bytes in a rectangular region
	// that view overlaps, including the empty spaces.
	// Equal to abs(rowStrideBytes()) * height()
	size_t bytesSpanned() const;

	// the space between the start of elements in bytes
	int elementStrideBytes() const;

	// the space between the start of rows in bytes
	int rowStrideBytes() const;

	// Returns true if there is no space between adjacent elements *within* a row,
	// i.e., if elementStrideBytes() == sizeof( T ).
	bool elementsArePacked() const;

	// Returns true if there is no space between adjacent rows,
	// i.e., if rowStrideBytes() == width() * elementStrideBytes().
	bool rowsArePacked() const;

	// returns true if elementsArePacked() && rowsArePacked()
	// also known as "linear"
	bool packed() const;

    // Conversion operator to Array2DView< const T >
    operator Array2DView< const T >() const;

private:

    Vector2i m_size;
    Vector2i m_strides;
    typename WrapConstPointerT< T, uint8_t >::pointer m_pPointer;
};

#include "Array2DView.inl"
