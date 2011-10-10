#ifndef ARRAY_UTIL_H
#define ARRAY_UTIL_H

#include "ArrayWithLength.h"

class ArrayUtils
{
public:

	static ArrayWithLength< float > createFloatArray( int length, float fillValue );
	static void dumpFloatArrayToFileText( ArrayWithLength< float > afData, const char* filename );
};

#endif // ARRAY_UTIL_H
