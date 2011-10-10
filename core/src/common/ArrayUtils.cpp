#include "common/ArrayUtils.h"

#include <cstdlib>

//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
ArrayWithLength< float > ArrayUtils::createFloatArray( int length, float fillValue )
{
	float* arr = new float[ length ];
	for( int i = 0; i < length; ++i )
	{
		arr[ i ] = fillValue;
	}
	
	return ArrayWithLength< float >( arr, length );
}

// static
void ArrayUtils::dumpFloatArrayToFileText( ArrayWithLength< float > afData, const char* filename )
{
	FILE* fp = fopen( filename, "w" );

	for( int i = 0; i < afData.length(); ++i )
	{
		fprintf( fp, "{%d}: <%f>\n", i, afData[i] );
	}
	fclose( fp );
}
