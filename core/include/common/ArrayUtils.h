#pragma once

#include <vector>
#include "Array1DView.h"
#include "Array2D.h"
#include "Array3D.h"

#include <vecmath/Vector2f.h>
#include <vecmath/Vector3f.h>
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

	static bool saveTXT( const Array2DView< uint8x4 >& view, const char* filename );

	static bool saveTXT( const Array2D< float >& array, const char* filename );
	static bool saveTXT( const Array3D< float >& array, const char* filename );

	static bool saveTXT( const Array3D< Vector2f >& array, const char* filename );

	static bool saveTXT( const Array2D< Vector4f >& array, const char* filename );

	// returns a view that's flipped left <--> right
	// using it twice flips it back to normal
	template< typename T >
	static Array1DView< T > flipLR( Array1DView< T > view );

	// returns a view that's flipped left <--> right
	// using it twice flips it back to normal
	template< typename T >
	static Array2DView< T > flipLR( Array2DView< T > view );

	// returns a view that's flipped up <--> down
	// using it twice flips it back to normal
	template< typename T >
	static Array2DView< T > flipUD( Array2DView< T > view );		

	// a view of a linear subset of a Array1DView, starting at x
	template< typename T >
	static Array1DView< T > crop( Array1DView< T > view, int x, int width );

	// a view of a rectangular subset of a Array2DView, starting at x, y
	template< typename T >
	static Array2DView< T > crop( Array2DView< T > view, int x, int y, int width, int height );	

	// a view of a box subset of a Array3DView, starting at x, y, z
	template< typename T >
	static Array3DView< T > crop( Array3DView< T > view, int x, int y, int z, int width, int height, int depth );

	template< typename T >
	static bool fill( Array2DView< T > view, const T& value );

	// Specialization for uint8_t.
	template<>
	static bool fill( Array2DView< uint8_t > view, const uint8_t& value );

	// copy between two Array1DViews, with potentially varying stride and pitch
	// if both are packed(), uses memcpy to copy quickly
	// otherwise, iterates
	// returns false if the dimensions don't match, or if either is null
	template< typename T >
	static bool copy( Array1DView< T > src, Array1DView< T > dst );

	// copy between two Array2DViews, with potentially varying stride and pitch
	// if both are packed(), uses memcpy to copy quickly
	// if both strides are the same, then uses memcpy row by row
	// otherwise, iterates
	// returns false if the dimensions don't match, or if either is null
	template< typename T >
	static bool copy( Array2DView< const T > src, Array2DView< T > dst );
};

#include "common/ArrayUtils.inl"