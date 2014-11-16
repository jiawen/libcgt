#include "common/ArrayUtils.h"

#include <cstdlib>


//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
bool ArrayUtils::saveTXT( const std::vector< float >& array, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	int retVal;
	int n = static_cast< int >( array.size() );

	retVal = fprintf( fp, "Size: %d\n", n );
	if( retVal < 0 )
	{
		return false;
	}

	retVal = fprintf( fp, "Format: float\n" );
	if( retVal < 0 )
	{
		return false;
	}

	for( int i = 0; i < n; ++i )
	{
		fprintf( fp, "[%d]: %f\n", i, array[ i ] );
	}
	
	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( const std::vector< int >& array, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	int retVal;
	int n = static_cast< int >( array.size() );

	retVal = fprintf( fp, "Size: %d\n", n );
	if( retVal < 0 )
	{
		return false;
	}

	retVal = fprintf( fp, "Format: int\n" );
	if( retVal < 0 )
	{
		return false;
	}

	for( int i = 0; i < n; ++i )
	{
		fprintf( fp, "[%d]: %d\n", i, array[ i ] );
	}

	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( const std::vector< Vector2f >& array, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	int retVal;
	int n = static_cast< int >( array.size() );

	retVal = fprintf( fp, "Size: %d\n", n );
	if( retVal < 0 )
	{
		return false;
	}

	retVal = fprintf( fp, "Format: float2\n" );
	if( retVal < 0 )
	{
		return false;
	}

	for( int i = 0; i < n; ++i )
	{
		Vector2f v = array[i];
		fprintf( fp, "[%d]: %f %f\n", i, v.x, v.y );
	}

	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( const std::vector< Vector3f >& array, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	int retVal;
	int n = static_cast< int >( array.size() );

	retVal = fprintf( fp, "Size: %d\n", n );
	if( retVal < 0 )
	{
		return false;
	}

	retVal = fprintf( fp, "Format: float3\n" );
	if( retVal < 0 )
	{
		return false;
	}

	for( int i = 0; i < n; ++i )
	{
		Vector3f v = array[i];
		fprintf( fp, "[%d]: %f %f %f\n", i, v.x, v.y, v.z );
	}

	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( const std::vector< Vector4f >& array, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	int retVal;
	int n = static_cast< int >( array.size() );

	retVal = fprintf( fp, "Size: %d\n", n );
	if( retVal < 0 )
	{
		return false;
	}

	retVal = fprintf( fp, "Format: float4\n" );
	if( retVal < 0 )
	{
		return false;
	}

	for( int i = 0; i < n; ++i )
	{
		Vector4f v = array[i];
		fprintf( fp, "[%d]: %f %f %f %f\n", i, v.x, v.y, v.z, v.w );
	}

	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( const Array2DView< ubyte4 >& view, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}
	
	int retVal;
	int w = view.width();
	int h = view.height();

	retVal = fprintf( fp, "Size: %d x %d\n", w, h );
	if( retVal < 0 )
	{
		return false;
	}

	retVal = fprintf( fp, "Format: ubyte4\n" );
	if( retVal < 0 )
	{
		return false;
	}	

	for( int y = 0; y < h; ++y )
	{
		for( int x = 0; x < w; ++x )
		{
			int index = y * w + x;
			ubyte4 v = view( x, y );
			fprintf( fp, "[%d] (%d %d): %d %d %d %d\n", index, x, y, v.x, v.y, v.z, v.w );
		}
	}
	
	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( const Array2D< float >& array, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}
	
	int retVal;
	
	retVal = fprintf( fp, "Size: %d x %d\n", array.width(), array.height() );
	if( retVal < 0 )
	{
		return false;
	}

	retVal = fprintf( fp, "Format: float\n" );
	if( retVal < 0 )
	{
		return false;
	}

	int w = array.width();
	int h = array.height();

	for( int y = 0; y < h; ++y )
	{
		for( int x = 0; x < w; ++x )
		{
			int index = y * w + x;
            float v = array[ { x, y } ];
			fprintf( fp, "[%d] (%d %d): %f\n", index, x, y, v );
		}
	}
	
	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( const Array3D< float >& array, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	int retVal;

	retVal = fprintf( fp, "Size: %d x %d x %d\n", array.width(), array.height(), array.depth() );
	if( retVal < 0 )
	{
		return false;
	}

	retVal = fprintf( fp, "Format: float\n" );
	if( retVal < 0 )
	{
		return false;
	}

	int w = array.width();
	int h = array.height();
	int d = array.depth();

	for( int z = 0; z < d; ++z )
	{
		for( int y = 0; y < h; ++y )
		{
			for( int x = 0; x < w; ++x )
			{
				int index = z * w * h + y * w + x;
				float v = array( x, y, z );
				fprintf( fp, "[%d] (%d %d %d): %f\n", index, x, y, z, v );
			}
		}
	}
	
	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( const Array3D< Vector2f >& array, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	int retVal;

	retVal = fprintf( fp, "Size: %d x %d x %d\n", array.width(), array.height(), array.depth() );
	if( retVal < 0 )
	{
		return false;
	}

	retVal = fprintf( fp, "Format: float2\n" );
	if( retVal < 0 )
	{
		return false;
	}

	int w = array.width();
	int h = array.height();
	int d = array.depth();

	for( int z = 0; z < d; ++z )
	{
		for( int y = 0; y < h; ++y )
		{
			for( int x = 0; x < w; ++x )
			{
				int index = z * w * h + y * w + x;
				Vector2f v = array( x, y, z );
				fprintf( fp, "[%d] (%d %d %d): %f %f\n", index, x, y, z, v.x, v.y );
			}
		}
	}

	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( const Array2D< Vector4f >& array, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	int retVal;

	retVal = fprintf( fp, "Size: %d x %d\n", array.width(), array.height() );
	if( retVal < 0 )
	{
		return false;
	}

	retVal = fprintf( fp, "Format: float4\n" );
	if( retVal < 0 )
	{
		return false;
	}

	int w = array.width();
	int h = array.height();

	for( int y = 0; y < h; ++y )
	{
		for( int x = 0; x < w; ++x )
		{
			int index = y * w + x;
            Vector4f v = array[ { x, y } ];
			fprintf( fp, "[%d] (%d %d): %f %f %f %f\n", index, x, y, v.x, v.y, v.z, v.w );
		}
	}

	retVal = fclose( fp );
	return( retVal == 0 );
}
