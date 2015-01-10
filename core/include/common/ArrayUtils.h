#pragma once

#include <vector>
#include "Array1DView.h"
#include "Array2D.h"
#include "Array3D.h"

#include <vecmath/Box3i.h>
#include <vecmath/Rect2i.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector2i.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector3i.h>
#include <vecmath/Vector4f.h>

class ArrayUtils
{
public:

    // TODO: vector --> Array1DView<T>
	template< typename T >
	static bool loadBinary( FILE* fp, std::vector< T >& output );

	template< typename T >
	static bool saveBinary( const std::vector< T >& input, const char* filename );

	template< typename T >
	static bool saveBinary( const std::vector< T >& input, FILE* fp );

	static bool saveTXT( const std::vector< float >& array, const char* filename );
	static bool saveTXT( const std::vector< int >& array, const char* filename );

	static bool saveTXT( const std::vector< Vector2f >& array, const char* filename );
	static bool saveTXT( const std::vector< Vector3f >& array, const char* filename );
	static bool saveTXT( const std::vector< Vector4f >& array, const char* filename );

	static bool saveTXT( Array2DView< const uint8x4 >& view, const char* filename );

	static bool saveTXT( const Array2D< float >& array, const char* filename );
	static bool saveTXT( const Array3D< float >& array, const char* filename );

	static bool saveTXT( const Array3D< Vector2f >& array, const char* filename );

	static bool saveTXT( const Array2D< Vector4f >& array, const char* filename );

    template< typename T >
    static bool clear( Array2DView< T > view );

    template< typename T >
    static bool fill( Array2DView< T > view, const T& value );

	// returns a view that's flipped left <--> right
	// using it twice flips it back to normal
	template< typename T >
    static Array1DView< T > flippedLRView( Array1DView< T > view );

	// returns a view that's flipped left <--> right
	// using it twice flips it back to normal
	template< typename T >
	static Array2DView< T > flippedLRView( Array2DView< T > view );

	// returns a view that's flipped up <--> down
	// using it twice flips it back to normal
    template< typename T >
    static Array2DView< T > flippedUpDownView( Array2DView< T > view );

    // Get a of a rectangular subset of a Array2DView, starting at xy.
	template< typename T >
    static Array2DView< T > croppedView( Array2DView< T > view, const Vector2i& xy );

    // Get a of a rectangular subset of a Array2DView.
    template< typename T >
    static Array2DView< T > croppedView( Array2DView< T > view, const Rect2i& rect );

    // Get a of a box subset of a Array3DView, starting at xyz.
    template< typename T >
    static Array3DView< T > croppedView( Array3DView< T > view, const Vector3i& xyz );

    // Get a of a box subset of a Array3DView.
	template< typename T >
    static Array3DView< T > croppedView( Array3DView< T > view, const Box3i& box );

    // Copy between two Array1DViews, with potentially varying stride.
	// If both are packed(), uses memcpy to do a single fast copy.
	// Otherwise, iterates the copy one element at a time.
	// Returns false if the dimensions don't match, or if either is null.
    template< typename T >
	static bool copy( Array1DView< const T > src, Array1DView< T > dst );

	// Copy between two Array2DViews, with potentially varying strides.
	// If both are packed(), uses memcpy to do a single fast copy.
	// If both row strides are the same, then uses memcpy row by row.
	// Otherwise, iterates the copy one element at a time.
	// Returns false if the dimensions don't match, or if either is null.
	template< typename T >
	static bool copy( Array2DView< const T > src, Array2DView< T > dst );
};

#include "common/ArrayUtils.inl"