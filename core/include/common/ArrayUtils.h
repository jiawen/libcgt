#pragma once

#include <vector>
#include "Array2D.h"
#include "Array3D.h"

class ArrayUtils
{
public:

	static bool saveTXT( const std::vector< float >& array, const char* filename );
	static bool saveTXT( const Array2D< float >& array, const char* filename );
	static bool saveTXT( const Array3D< float >& array, const char* filename );
};
