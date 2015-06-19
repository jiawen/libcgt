#include "common/ArrayUtils.h"

#include <cstdlib>


//////////////////////////////////////////////////////////////////////////
// Public
//////////////////////////////////////////////////////////////////////////

// static
bool ArrayUtils::saveTXT( Array1DView< const int16_t > view, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	int retVal;
	int n = static_cast< int >( view.width() );

	retVal = fprintf( fp, "Size: %d\n", n );
	if( retVal < 0 )
	{
		return false;
	}

	retVal = fprintf( fp, "Format: int16_t\n" );
	if( retVal < 0 )
	{
		return false;
	}

	for( int i = 0; i < n; ++i )
	{
		fprintf( fp, "[%d]: %d\n", i, view[ i ] );
	}

	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( Array1DView< const int32_t > view, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	int retVal;
	int n = static_cast< int >( view.width() );

	retVal = fprintf( fp, "Size: %d\n", n );
	if( retVal < 0 )
	{
		return false;
	}

	retVal = fprintf( fp, "Format: int32_t\n" );
	if( retVal < 0 )
	{
		return false;
	}

	for( int i = 0; i < n; ++i )
	{
		fprintf( fp, "[%d]: %d\n", i, view[ i ] );
	}

	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( Array1DView< const float >& view, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	int retVal;
	int n = static_cast< int >( view.width() );

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
		float v = view[i];
		fprintf( fp, "[%d]: %f %f\n", i, v );
	}

	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( Array1DView< const Vector2f >& view, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	int retVal;
	int n = static_cast< int >( view.width() );

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
		Vector2f v = view[i];
		fprintf( fp, "[%d]: %f %f\n", i, v.x, v.y );
	}

	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( Array1DView< const Vector3f >& view, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	int retVal;
	int n = static_cast< int >( view.width() );

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
		Vector3f v = view[i];
		fprintf( fp, "[%d]: %f %f %f\n", i, v.x, v.y, v.z );
	}

	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( Array1DView< const Vector4f >& view, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	int retVal;
	int n = static_cast< int >( view.width() );

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
		Vector4f v = view[i];
		fprintf( fp, "[%d]: %f %f %f %f\n", i, v.x, v.y, v.z, v.w );
	}

	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( Array2DView< const uint8_t > view, const char* filename )
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

	retVal = fprintf( fp, "Format: uint8_t\n" );
	if( retVal < 0 )
	{
		return false;
	}	

	for( int y = 0; y < h; ++y )
	{
		for( int x = 0; x < w; ++x )
		{
			int index = y * w + x;
            uint8_t v = view[ { x, y } ];
			fprintf( fp, "[%d] (%d %d): %d\n", index, x, y, v );
		}
	}
	
	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( Array2DView< const uint8x4 > view, const char* filename )
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

	retVal = fprintf( fp, "Format: uint8x4\n" );
	if( retVal < 0 )
	{
		return false;
	}	

	for( int y = 0; y < h; ++y )
	{
		for( int x = 0; x < w; ++x )
		{
			int index = y * w + x;
            uint8x4 v = view[ { x, y } ];
			fprintf( fp, "[%d] (%d %d): %d %d %d %d\n", index, x, y, v.x, v.y, v.z, v.w );
		}
	}
	
	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( Array2DView< const int16_t > view, const char* filename )
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

	retVal = fprintf( fp, "Format: int16_t\n" );
	if( retVal < 0 )
	{
		return false;
	}	

	for( int y = 0; y < h; ++y )
	{
		for( int x = 0; x < w; ++x )
		{
			int index = y * w + x;
            int16_t v = view[ { x, y } ];
			fprintf( fp, "[%d] (%d %d): %d\n", index, x, y, v );
		}
	}
	
	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( Array2DView< const float > view, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}
	
	int retVal;
	
	retVal = fprintf( fp, "Size: %d x %d\n", view.width(), view.height() );
	if( retVal < 0 )
	{
		return false;
	}

	retVal = fprintf( fp, "Format: float\n" );
	if( retVal < 0 )
	{
		return false;
	}

	int w = view.width();
	int h = view.height();

	for( int y = 0; y < h; ++y )
	{
		for( int x = 0; x < w; ++x )
		{
			int index = y * w + x;
            float v = view[ { x, y } ];
			fprintf( fp, "[%d] (%d %d): %f\n", index, x, y, v );
		}
	}
	
	retVal = fclose( fp );
	return( retVal == 0 );
}


// static
bool ArrayUtils::saveTXT( Array2DView< const Vector4f > view, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	int retVal;

	retVal = fprintf( fp, "Size: %d x %d\n", view.width(), view.height() );
	if( retVal < 0 )
	{
		return false;
	}

	retVal = fprintf( fp, "Format: float4\n" );
	if( retVal < 0 )
	{
		return false;
	}

	int w = view.width();
	int h = view.height();

	for( int y = 0; y < h; ++y )
	{
		for( int x = 0; x < w; ++x )
		{
			int index = y * w + x;
            Vector4f v = view[ { x, y } ];
			fprintf( fp, "[%d] (%d %d): %f %f %f %f\n", index, x, y, v.x, v.y, v.z, v.w );
		}
	}

	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( Array3DView< const float > view, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	int retVal;

	retVal = fprintf( fp, "Size: %d x %d x %d\n", view.width(), view.height(), view.depth() );
	if( retVal < 0 )
	{
		return false;
	}

	retVal = fprintf( fp, "Format: float\n" );
	if( retVal < 0 )
	{
		return false;
	}

	int w = view.width();
	int h = view.height();
	int d = view.depth();

	for( int z = 0; z < d; ++z )
	{
		for( int y = 0; y < h; ++y )
		{
			for( int x = 0; x < w; ++x )
			{
				int index = z * w * h + y * w + x;
                float v = view[ { x, y, z } ];
				fprintf( fp, "[%d] (%d %d %d): %f\n", index, x, y, z, v );
			}
		}
	}
	
	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
bool ArrayUtils::saveTXT( Array3DView< const Vector2f > view, const char* filename )
{
	FILE* fp = fopen( filename, "w" );
	if( fp == NULL )
	{
		return false;
	}

	int retVal;

	retVal = fprintf( fp, "Size: %d x %d x %d\n", view.width(), view.height(), view.depth() );
	if( retVal < 0 )
	{
		return false;
	}

	retVal = fprintf( fp, "Format: float2\n" );
	if( retVal < 0 )
	{
		return false;
	}

	int w = view.width();
	int h = view.height();
	int d = view.depth();

	for( int z = 0; z < d; ++z )
	{
		for( int y = 0; y < h; ++y )
		{
			for( int x = 0; x < w; ++x )
			{
				int index = z * w * h + y * w + x;
                Vector2f v = view[ { x, y, z } ];
				fprintf( fp, "[%d] (%d %d %d): %f %f\n", index, x, y, z, v.x, v.y );
			}
		}
	}

	retVal = fclose( fp );
	return( retVal == 0 );
}

// static
template<>
bool ArrayUtils::fill( Array2DView< uint8_t > view, const uint8_t& value )
{
    if( view.isNull( ) )
    {
        return false;
    }

    if( view.packed( ) )
    {
        memset( view.pointer( ), value, view.numElements( ) );
        return true;
    }

    if( view.elementsArePacked( ) )
    {
        // Fill w bytes at a time.
        int w = view.width( );
        for( int y = 0; y < view.height( ); ++y )
        {
            memset( view.rowPointer( y ), value, w );
        }
        return true;
    }

    // Nothing is packed, iterate.
    int ne = view.numElements( );
    for( int k = 0; k < ne; ++k )
    {
        view[ k ] = value;
    }
    return true;
}
