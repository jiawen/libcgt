#ifndef ARRAY_UTILS_H
#define ARRAY_UTILS_H

class ArrayUtils
{
public:

	static bool saveTXT( const Array2D< float2 >& array, const char* filename );
	static bool saveTXT( const Array3D< float2 >& array, const char* filename );
};

#endif ARRAY_UTILS_H
