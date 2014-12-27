#pragma once

#include <vector>
#include "Array1DView.h"
#include "Array2D.h"
#include "Array3D.h"

#include <vecmath/Vector2f.h>
#include <vecmath/Vector2i.h>
#include <vecmath/Vector3f.h>
#include <vecmath/Vector3i.h>
#include <vecmath/Vector4f.h>

class ArrayUtils
{
public:

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

	template< typename T >
    static Array2DView< T > croppedView( Array2DView< T > view, const Vector2i& xy );

    // TODO: use Rect2i once it has an initializer list
	// a view of a rectangular subset of a Array3DView, starting at x, y
	template< typename T >
    static Array2DView< T > croppedView( Array2DView< T > view, const Vector2i& xy, const Vector2i& size );

	// a view of a rectangular subset of a Array2DView, starting at x, y
	template< typename T >
    static Array3DView< T > croppedView( Array3DView< T > view, const Vector3i& xyz );

    // TODO: use Box3i once it has an initializer list
	// a view of a box subset of a Array3DView, starting at x, y, z
	template< typename T >
    static Array3DView< T > croppedView( Array3DView< T > view, const Vector3i& xyz, const Vector3i& size );

	// copy between two Array2DViews, with potentially varying stride and pitch
	// if both are packed(), uses memcpy to copy quickly
	// if both strides are the same, then uses memcpy row by row
	// otherwise, iterates
	// returns false if the dimensions don't match, or if either is null
	template< typename T >
	static bool copy( Array2DView< const T > src, Array2DView< T > dst );
};

#include "common/ArrayUtils.inl"