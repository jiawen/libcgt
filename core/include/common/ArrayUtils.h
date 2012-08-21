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

	static bool saveTXT( const std::vector< float >& array, const char* filename );
	static bool saveTXT( const Array2D< float >& array, const char* filename );
	static bool saveTXT( const Array3D< float >& array, const char* filename );

	static bool saveTXT( const Array3D< Vector2f >& array, const char* filename );

	static bool saveTXT( const Array2D< Vector4f >& array, const char* filename );
};
