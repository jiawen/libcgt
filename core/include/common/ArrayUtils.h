#pragma once

#include <vector>
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

	static bool saveTXT( const Array2D< float >& array, const char* filename );
	static bool saveTXT( const Array3D< float >& array, const char* filename );

	static bool saveTXT( const Array3D< Vector2f >& array, const char* filename );

	static bool saveTXT( const Array2D< Vector4f >& array, const char* filename );

	// copy between two Array2DViews, with potentially varying stride and pitch
	// if both are packed(), uses memcpy to copy quickly
	// if both strides are the same, then uses memcpy row by row
	// otherwise, iterates
	// returns false if the dimensions don't match, or if either is null
	template< typename T >
	static bool copy( Array2DView< T > src, Array2DView< T > dst );
};

#include "common/ArrayUtils.inl"