#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "Array1D.h"
#include "Array1DView.h"
#include "Array2D.h"
#include "Array2DView.h"
#include "Array3D.h"
#include "Array3DView.h"
#include "BasicTypes.h" // TODO: uint8x4 --> Vector4ub

#include <vecmath/Box3i.h>
#include <vecmath/Range1i.h>
#include <vecmath/Rect2i.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector2i.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector3i.h>
#include <vecmath/Vector4f.h>

namespace libcgt { namespace core { namespace arrayutils {

// Cast from Array1DView< TIn > --> Array1DView< TOut >.
// sizeof( TIn ) must be equal sizeof( TOut ).
template< typename TOut, typename TIn >
Array1DView< TOut > cast( Array1DView< TIn > src );

// Cast from Array2DView< TIn > --> Array2DView< TOut >.
// sizeof( TIn ) must be equal sizeof( TOut ).
template< typename TOut, typename TIn >
Array2DView< TOut > cast( Array2DView< TIn > src );

// Cast from Array3DView< TIn > --> Array3DView< TOut >.
// sizeof( TIn ) must be equal sizeof( TOut ).
template< typename TOut, typename TIn >
Array3DView< TOut > cast( Array3DView< TIn > src );

// Copy between two Array1DViews, with potentially varying stride.
// If both are packed(), uses memcpy to do a single fast copy.
// Otherwise, copies element by element.
// Returns false if the dimensions don't match, or if either is null.
template< typename T >
bool copy( Array1DView< const T > src, Array1DView< T > dst );

// Copy between two Array2DViews, with potentially varying strides.
// If both are packed(), uses memcpy to do a single fast copy.
// If both row strides are equal, then calls copy() on each row.
// Otherwise, iterates the copy one element at a time.
// Returns false if the dimensions don't match, or if either is null.
template< typename T >
bool copy( Array2DView< const T > src, Array2DView< T > dst );

// Copy between two Array3DViews, with potentially varying strides.
// If both are packed(), uses memcpy to do a single fast copy.
// If both slice strides are the same, then calls copy() on each slice.
// Otherwise, iterates the copy one element at a time.
// Returns false if the dimensions don't match, or if either is null.
template< typename T >
bool copy( Array3DView< const T > src, Array3DView< T > dst );

// TODO: rename this to sliceChannel()?
// Given an existing Array1DView< TIn >, returns a Array1DView< TOut > with the
// same stride, but with elements of type TOut where TOut is a component of
// TIn and is at offset "componentOffsetBytes" within TIn.
template< typename TOut, typename TIn >
Array1DView< TOut > componentView( Array1DView< TIn > src,
    int componentOffsetBytes );

// Given an existing Array2DView< TIn >, returns a Array2DView< TOut > with the
// same stride, but with elements of type TOut where TOut is a component of
// TIn and is at offset "componentOffsetBytes" within TIn.
template< typename TOut, typename TIn >
Array2DView< TOut > componentView( Array2DView< TIn > src,
    int componentOffsetBytes );

// Given an existing Array3DView< TIn >, returns a Array3DView< TOut > with the
// same stride, but with elements of type TOut where TOut is a component of
// TIn and is at offset "componentOffsetBytes" within TIn.
template< typename TOut, typename TIn >
Array3DView< TOut > componentView( Array3DView< TIn > src,
    int componentOffsetBytes );

// Get a linear subset of an Array1DView, starting at x.
template< typename T >
Array1DView< T > crop( Array1DView< T > view, int x );

// Get a linear subset of an Array1DView.
template< typename T >
Array1DView< T > crop( Array1DView< T > view, const Range1i& range );

// Get a rectangular subset of an Array2DView, starting at xy.
template< typename T >
Array2DView< T > crop( Array2DView< T > view, const Vector2i& xy );

// Get a rectangular subset of an Array2DView.
template< typename T >
Array2DView< T > crop( Array2DView< T > view, const Rect2i& rect );

// Get a box subset of a Array3DView, starting at xyz.
template< typename T >
Array3DView< T > crop( Array3DView< T > view, const Vector3i& xyz );

// Get a box subset of an Array3DView.
template< typename T >
Array3DView< T > crop( Array3DView< T > view, const Box3i& box );

// TODO: implement clear() using fill(), specialize fill() on uint8_t
// use specialized view if isPacked() and T == 0?
template< typename T >
bool clear( Array2DView< T > view );

template< typename T >
bool fill( Array1DView< T > view, const T& value );

template< typename T >
bool fill( Array2DView< T > view, const T& value );

// Create a view that is flipped left <--> right from src.
// Flipping twice is the identity.
template< typename T >
Array1DView< T > flipX( Array1DView< T > src );

// Create a view that is flipped left <--> right from src.
// Flipping twice is the identity.
template< typename T >
Array2DView< T > flipX( Array2DView< T > src );

// Create a view that is flipped up <--> down from src.
// Flipping twice is the identity.
template< typename T >
Array2DView< T > flipY( Array2DView< T > src );

// Flip the contents of a view in-place.
// Flipping twice is the identity.
template< typename T >
void flipYInPlace( Array2DView< T > v );

// Create a view that swaps x and y.
// Transposing twice is the identity.
template< typename T >
Array2DView< T > transpose( Array2DView< T > src );

// Create a view that is flipped left <--> right from src.
// Flipping twice is the identity.
template< typename T >
Array3DView< T > flipX( Array3DView< T > src );

// Create a view that is flipped up <--> down from src.
// Flipping twice is the identity.
template< typename T >
Array3DView< T > flipY( Array3DView< T > src );

// Create a view that is flipped front <--> back from src.
// Flipping twice is the identity.
template< typename T >
Array3DView< T > flipZ( Array3DView< T > src );

// Get a view of the first n elements of src.
// Returns null if src has less than n elements.
template< typename T >
Array1DView< T > head( Array1DView< T > src, size_t n );

// Get a view of the last n elements of src.
// Returns null if src has less than n elements.
template< typename T >
Array1DView< T > tail( Array1DView< T > src, size_t n );

// Classical "map" function: dst[ i ] = f( src[ i ] ).
// f should be a function object that mapping TSrc -> TDst.
template< typename TSrc, typename TDst, typename Func >
bool map( Array1DView< const TSrc > src, Array1DView< TDst > dst, Func f );

// Classical "map" function: dst[ xy ] = f( src[ xy ] ).
// f should be a function object that mapping TSrc -> TDst.
template< typename TSrc, typename TDst, typename Func >
bool map( Array2DView< const TSrc > src, Array2DView< TDst > dst, Func f );

// Classical "map" function: dst[ xyz ] = f( src[ xyz ] ).
// f should be a function object that mapping TSrc -> TDst.
template< typename TSrc, typename TDst, typename Func >
bool map( Array3DView< const TSrc > src, Array3DView< TDst > dst, Func f );

// Variant of map that also passes the index to f:
// dst[ x ] = f( x, src[ x ] )
template< typename TSrc, typename TDst, typename Func >
bool mapIndexed( Array1DView< const TSrc > src, Array1DView< TDst > dst,
    Func f );

// Variant of map that also passes the index to f:
// dst[ xy ] = f( xy, src[ xy ] )
template< typename TSrc, typename TDst, typename Func >
bool mapIndexed( Array2DView< const TSrc > src, Array2DView< TDst > dst,
    Func f );

template< typename T >
Array1DView< T > reshape( Array2DView< T > src );

template< typename T >
Array1DView< T > reshape( Array3DView< T > src );

template< typename T >
Array1DView< const T > readViewOf( const std::vector< T >& v );

template< typename T >
Array1DView< T > writeViewOf( std::vector< T >& v );

} } } // arrayutils, core, libcgt

class ArrayUtils
{
public:

    // TODO: implement load / save of ImageStack TMP files.
    static bool saveTXT( Array1DView< const int16_t > view, const char* filename );
    static bool saveTXT( Array1DView< const int32_t > view, const char* filename );

    static bool saveTXT( Array1DView< const float >& view, const char* filename );
    static bool saveTXT( Array1DView< const Vector2f >& view, const char* filename );
    static bool saveTXT( Array1DView< const Vector3f >& view, const char* filename );
    static bool saveTXT( Array1DView< const Vector4f >& view, const char* filename );

    static bool saveTXT( Array2DView< const uint8_t > view, const char* filename );
    static bool saveTXT( Array2DView< const uint8x4 > view, const char* filename );

    static bool saveTXT( Array2DView< const int16_t > view, const char* filename );

    static bool saveTXT( Array2DView< const float > view, const char* filename );
    static bool saveTXT( Array2DView< const Vector4f > view, const char* filename );

    static bool saveTXT( Array3DView< const float > view, const char* filename );
    static bool saveTXT( Array3DView< const Vector2f > view, const char* filename );
};

#include "common/ArrayUtils.inl"
