#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "libcgt/core/common/Array1D.h"
#include "libcgt/core/common/Array2D.h"
#include "libcgt/core/common/Array3D.h"
#include "libcgt/core/common/ArrayView.h"
// TODO: uint8x4 --> Vector<4, uint8_t>
#include "libcgt/core/common/BasicTypes.h"
#include "libcgt/core/vecmath/Box3i.h"
#include "libcgt/core/vecmath/Range1i.h"
#include "libcgt/core/vecmath/Rect2i.h"
#include "libcgt/core/vecmath/Vector2f.h"
#include "libcgt/core/vecmath/Vector2i.h"
#include "libcgt/core/vecmath/Vector3f.h"
#include "libcgt/core/vecmath/Vector3i.h"
#include "libcgt/core/vecmath/Vector4f.h"
#include "libcgt/core/vecmath/Vector4i.h"

namespace libcgt { namespace core { namespace arrayutils {

// Cast from Array1DWriteView< TIn > --> Array1DWriteView< TOut >.
// sizeof( TIn ) must be equal sizeof( TOut ).
template< typename TOut, typename TIn >
Array1DReadView< TOut > cast( Array1DReadView< TIn > src );

// Cast from Array1DWriteView< TIn > --> Array1DWriteView< TOut >.
// sizeof( TIn ) must be equal sizeof( TOut ).
template< typename TOut, typename TIn >
Array1DWriteView< TOut > cast( Array1DWriteView< TIn > src );

// Cast from Array2DReadView< TIn > --> Array2DReadView< TOut >.
// sizeof( TIn ) must be equal sizeof( TOut ).
template< typename TOut, typename TIn >
Array2DReadView< TOut > cast( Array2DReadView< TIn > src );

// Cast from Array2DWriteView< TIn > --> Array2DWriteView< TOut >.
// sizeof( TIn ) must be equal sizeof( TOut ).
template< typename TOut, typename TIn >
Array2DWriteView< TOut > cast( Array2DWriteView< TIn > src );

// Cast from Array3DReadView< TIn > --> Array3DReadView< TOut >.
// sizeof( TIn ) must be equal sizeof( TOut ).
template< typename TOut, typename TIn >
Array3DReadView< TOut > cast( Array3DReadView< TIn > src );

// Cast from Array3DWriteView< TIn > --> Array3DWriteView< TOut >.
// sizeof( TIn ) must be equal sizeof( TOut ).
template< typename TOut, typename TIn >
Array3DWriteView< TOut > cast( Array3DWriteView< TIn > src );

// Copy between two 1D views, with potentially varying stride.
// If both are packed(), uses memcpy to do a single fast copy.
// Otherwise, copies element by element.
// Returns false if the dimensions don't match, or if either is null.
template< typename T >
bool copy( Array1DReadView< T > src, Array1DWriteView< T > dst );

// Copy between two 2D views, with potentially varying strides.
// If both are packed(), uses memcpy to do a single fast copy.
// If both row strides are equal, then calls copy() on each row.
// Otherwise, iterates the copy one element at a time.
// Returns false if the dimensions don't match, or if either is null.
template< typename T >
bool copy( Array2DReadView< T > src, Array2DWriteView< T > dst );

// Copy between two 3D views, with potentially varying strides.
// If both are packed(), uses memcpy to do a single fast copy.
// If both slice strides are the same, then calls copy() on each slice.
// Otherwise, iterates the copy one element at a time.
// Returns false if the dimensions don't match, or if either is null.
template< typename T >
bool copy( Array3DReadView< T > src, Array3DWriteView< T > dst );

// TODO: rename this to sliceChannel()?
// Given an existing Array1DReadView< TIn >, returns a
// Array1DReadView< TOut > with the same stride, but with elements of type
// TOut where TOut is a component of TIn and is at offset
// "componentOffsetBytes" within TIn.
template< typename TOut, typename TIn >
Array1DReadView< TOut > componentView( Array1DReadView< TIn > src,
    int componentOffsetBytes );

// Given an existing Array1DWriteView< TIn >, returns a
// Array1DWriteView< TOut > with the same stride, but with elements of type
// TOut where TOut is a component of TIn and is at offset
// "componentOffsetBytes" within TIn.
template< typename TOut, typename TIn >
Array1DWriteView< TOut > componentView( Array1DWriteView< TIn > src,
    int componentOffsetBytes );

// Given an existing Array2DReadView< TIn >, returns a
// Array2DReadView< TOut > with the same stride, but with elements of type
// TOut where TOut is a component of TIn and is at offset
// "componentOffsetBytes" within TIn.
template< typename TOut, typename TIn >
Array2DReadView< TOut > componentView( Array2DReadView< TIn > src,
    int componentOffsetBytes );

// Given an existing Array2DWriteView< TIn >, returns a
// Array2DWriteView< TOut > with the same stride, but with elements of type
// TOut where TOut is a component of TIn and is at offset
// "componentOffsetBytes" within TIn.
template< typename TOut, typename TIn >
Array2DWriteView< TOut > componentView( Array2DWriteView< TIn > src,
    int componentOffsetBytes );

// Given an existing Array3DWriteView< TIn >, returns a
// Array3DWriteView< TOut > with the same stride, but with elements of type
// TOut where TOut is a component of TIn and is at offset
// "componentOffsetBytes" within TIn.
template< typename TOut, typename TIn >
Array3DWriteView< TOut > componentView( Array3DWriteView< TIn > src,
    int componentOffsetBytes );

// Get a linear subset of a 1D view, starting at x.
template< typename T >
Array1DWriteView< T > crop( Array1DWriteView< T > view, int x );

// Get a linear subset of a 1D view.
template< typename T >
Array1DWriteView< T > crop( Array1DWriteView< T > view, const Range1i& range );

// Get a rectangular subset of a 2D view, starting at xy.
template< typename T >
Array2DReadView< T > crop( Array2DReadView< T > view, const Vector2i& xy );

// Get a rectangular subset of a 2D view, starting at xy.
template< typename T >
Array2DWriteView< T > crop( Array2DWriteView< T > view, const Vector2i& xy );

// Get a rectangular subset of a 2D view.
template< typename T >
Array2DReadView< T > crop( Array2DReadView< T > view, const Rect2i& rect );

// Get a rectangular subset of a 2D view.
template< typename T >
Array2DWriteView< T > crop( Array2DWriteView< T > view, const Rect2i& rect );

// Get a box subset of a 3D view, starting at xyz.
template< typename T >
Array3DWriteView< T > crop( Array3DWriteView< T > view, const Vector3i& xyz );

// Get a box subset of a 3D view.
template< typename T >
Array3DWriteView< T > crop( Array3DWriteView< T > view, const Box3i& box );

// TODO: implement clear() using fill(), specialize fill() on uint8_t
// use specialized view if isPacked() and T == 0?
template< typename T >
bool clear( Array2DWriteView< T > view );

template< typename T >
bool fill( Array1DWriteView< T > view, const T& value );

template< typename T >
bool fill( Array2DWriteView< T > view, const T& value );

// Create a view that is flipped left <--> right from src.
// Flipping twice is the identity.
template< typename T >
Array1DReadView< T > flipX( Array1DReadView< T > src );

// Create a view that is flipped left <--> right from src.
// Flipping twice is the identity.
template< typename T >
Array1DWriteView< T > flipX( Array1DWriteView< T > src );

// Create a view that is flipped left <--> right from src.
// Flipping twice is the identity.
template< typename T >
Array2DReadView< T > flipX( Array2DReadView< T > src );

// Create a view that is flipped left <--> right from src.
// Flipping twice is the identity.
template< typename T >
Array2DWriteView< T > flipX( Array2DWriteView< T > src );

// Create a view that is flipped up <--> down from src.
// Flipping twice is the identity.
template< typename T >
Array2DReadView< T > flipY( Array2DReadView< T > src );

// Create a view that is flipped up <--> down from src.
// Flipping twice is the identity.
template< typename T >
Array2DWriteView< T > flipY( Array2DWriteView< T > src );

// Flip the contents of a view in-place.
// Flipping twice is the identity.
template< typename T >
void flipYInPlace( Array2DWriteView< T > v );

// Create a view that swaps x and y.
// Transposing twice is the identity.
template< typename T >
Array2DReadView< T > transpose( Array2DReadView< T > src );

// Create a view that swaps x and y.
// Transposing twice is the identity.
template< typename T >
Array2DWriteView< T > transpose( Array2DWriteView< T > src );

// Create a view that is flipped left <--> right from src.
// Flipping twice is the identity.
template< typename T >
Array3DWriteView< T > flipX( Array3DWriteView< T > src );

// Create a view that is flipped up <--> down from src.
// Flipping twice is the identity.
template< typename T >
Array3DWriteView< T > flipY( Array3DWriteView< T > src );

// Create a view that is flipped front <--> back from src.
// Flipping twice is the identity.
template< typename T >
Array3DWriteView< T > flipZ( Array3DWriteView< T > src );

// Get a view of the first n elements of src.
// Returns null if src has less than n elements.
template< typename T >
Array1DWriteView< T > head( Array1DWriteView< T > src, size_t n );

// Get a view of the last n elements of src.
// Returns null if src has less than n elements.
template< typename T >
Array1DWriteView< T > tail( Array1DWriteView< T > src, size_t n );

// Classical "map" function: dst[ x ] = f( src[ x ] ).
// f should be a function object that mapping TSrc -> TDst.
template< typename TSrc, typename TDst, typename Func >
bool map( Array1DReadView< TSrc > src, Array1DWriteView< TDst > dst, Func f );

// Classical "map" function: dst[ xy ] = f( src[ xy ] ).
// f should be a function object that mapping TSrc -> TDst.
template< typename TSrc, typename TDst, typename Func >
bool map( Array2DReadView< TSrc > src, Array2DWriteView< TDst > dst, Func f );

// Classical "map" function: dst[ xyz ] = f( src[ xyz ] ).
// f should be a function object that mapping TSrc -> TDst.
template< typename TSrc, typename TDst, typename Func >
bool map( Array3DReadView< TSrc > src, Array3DWriteView< TDst > dst, Func f );

// Variant of map that also passes the index to f:
// dst[ x ] = f( x, src[ x ] )
template< typename TSrc, typename TDst, typename Func >
bool mapIndexed( Array1DReadView< TSrc > src, Array1DWriteView< TDst > dst,
    Func f );

// Variant of map that also passes the index to f:
// dst[ xy ] = f( xy, src[ xy ] )
template< typename TSrc, typename TDst, typename Func >
bool mapIndexed( Array2DReadView< TSrc > src, Array2DWriteView< TDst > dst,
    Func f );

// Variant of map that also passes the index to f:
// dst[ xyz ] = f( xyz, src[ xyz ] )
template< typename TSrc, typename TDst, typename Func >
bool mapIndexed( Array3DReadView< TSrc > src, Array3DWriteView< TDst > dst,
    Func f );

// Interpret "src" as a 1D view. src's rows must be packed.
template< typename T >
Array1DReadView< T > flatten( Array2DReadView< T > src );

// Interpret "src" as a 1D view. src's rows must be packed.
template< typename T >
Array1DWriteView< T > flatten( Array2DWriteView< T > src );

// Interpret "src" as a 1D view. src's rows and slices must be packed.
template< typename T >
Array1DReadView< T > flatten( Array3DReadView< T > src );

// Interpret "src" as a 1D view. src's rows and slices must be packed.
template< typename T >
Array1DWriteView< T > flatten( Array3DWriteView< T > src );

// Get a readView of the contents of the vector v, starting at element
// (not byte) offset "offset".
template< typename T >
Array1DReadView< T > readViewOf( const std::vector< T >& v,
    size_t offset = 0 );

// Get a writeView of the contents of the vector v, starting at element
// (not byte) offset "offset".
template< typename T >
Array1DWriteView< T > writeViewOf( std::vector< T >& v, size_t offset = 0 );

Array1DReadView< uint8_t > readViewOf( const std::string& s );


} } } // arrayutils, core, libcgt

// TODO: implement load / save of ImageStack TMP files.

// TODO: move this into its own file.
bool saveTXT( Array1DReadView< int16_t > view, const std::string& filename );
bool saveTXT( Array1DReadView< int32_t > view, const std::string& filename );

bool saveTXT( Array1DReadView< Vector3i >& view, const std::string& filename );

bool saveTXT( Array1DReadView< float >& view, const std::string& filename );
bool saveTXT( Array1DReadView< Vector2f >& view, const std::string& filename );
bool saveTXT( Array1DReadView< Vector3f >& view, const std::string& filename );
bool saveTXT( Array1DReadView< Vector4f >& view, const std::string& filename );

bool saveTXT( Array2DReadView< uint8_t > view, const std::string& filename );
bool saveTXT( Array2DReadView< uint8x4 > view, const std::string& filename );

bool saveTXT( Array2DReadView< int16_t > view, const std::string& filename );

bool saveTXT( Array2DReadView< float > view, const std::string& filename );
bool saveTXT( Array2DReadView< Vector2f > view, const std::string& filename );
bool saveTXT( Array2DReadView< Vector3f > view, const std::string& filename );
bool saveTXT( Array2DReadView< Vector4f > view, const std::string& filename );

bool saveTXT( Array3DReadView< uint16x2 > view, const std::string& filename );

bool saveTXT( Array3DReadView< Vector2i > view, const std::string& filename );
bool saveTXT( Array3DReadView< Vector3i > view, const std::string& filename );
bool saveTXT( Array3DReadView< Vector4i > view, const std::string& filename );

bool saveTXT( Array3DReadView< float > view, const std::string& filename );
bool saveTXT( Array3DReadView< Vector2f > view, const std::string& filename );

#include "libcgt/core/common/ArrayUtils.inl"
