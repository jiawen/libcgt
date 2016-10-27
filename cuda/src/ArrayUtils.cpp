#include "ArrayUtils.h"

#include <common/Array1D.h>
#include <common/Array2D.h>
#include <common/Array3D.h>
#include <common/ArrayUtils.h>

// TODO: Array2D<float2> --> Array2DView<Vector2f>

namespace libcgt { namespace cuda { namespace arrayutils {

bool saveTXT( Array1DView< const int3 > array, const char* filename )
{
    FILE* fp = fopen( filename, "w" );
    if( fp == nullptr )
    {
        return false;
    }

    fprintf( fp, "Size: %zu\n", array.size() );
    fprintf( fp, "Format: int3\n" );

    int length = static_cast< int >( array.size() );
    for( int i = 0; i < length; ++i )
    {
        int3 v = array[i];
        fprintf( fp, "[%d]: %d, %d, %d\n", i, v.x, v.y, v.z );
    }

    fclose( fp );

    return true;
}

bool saveTXT( Array2DView< const float2 > array, const char* filename )
{
    FILE* fp = fopen( filename, "w" );
    if( fp == NULL )
    {
        return false;
    }

    fprintf( fp, "Size: %d x %d\n", array.width(), array.height() );
    fprintf( fp, "Format: float2\n" );

    int w = array.width();
    int h = array.height();

    for( int y = 0; y < h; ++y )
    {
        for( int x = 0; x < w; ++x )
        {
            int index = y * w + x;
            float2 v = array[ { x, y } ];
            fprintf( fp, "[%d] (%d %d): %f, %f\n", index, x, y, v.x, v.y );
        }
    }
    fclose( fp );

    return true;
}

bool saveTXT( Array2DView< const float4 > array, const char* filename )
{
    FILE* fp = fopen( filename, "w" );
    if( fp == NULL )
    {
        return false;
    }

    fprintf( fp, "Size: %d x %d\n", array.width(), array.height() );
    fprintf( fp, "Format: float4\n" );

    int w = array.width();
    int h = array.height();

    for( int y = 0; y < h; ++y )
    {
        for( int x = 0; x < w; ++x )
        {
            int index = y * array.width() + x;
            float4 v = array[ { x, y } ];
            fprintf( fp, "[%d] (%d %d): %f, %f, %f, %f\n", index, x, y, v.x, v.y, v.z, v.w );
        }
    }
    fclose( fp );

    return true;
}

bool saveTXT( Array2DView< const uchar4 > array, const char* filename )
{
    FILE* fp = fopen( filename, "w" );
    if( fp == NULL )
    {
        return false;
    }

    fprintf( fp, "Size: %d x %d\n", array.width(), array.height() );
    fprintf( fp, "Format: ubyte4\n" );

    int w = array.width();
    int h = array.height();

    for( int y = 0; y < h; ++y )
    {
        for( int x = 0; x < w; ++x )
        {
            int index = y * array.width() + x;
            uchar4 v = array[ { x, y } ];
            fprintf( fp, "[%d] (%d %d): %d, %d, %d, %d\n", index, x, y, v.x, v.y, v.z, v.w );
        }
    }
    fclose( fp );

    return true;
}

bool saveTXT( Array3DView< const ushort2 > array, const char* filename )
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

    retVal = fprintf( fp, "Format: ushort2\n" );
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
                ushort2 v = array[ { x, y, z } ];
                fprintf( fp, "[%d] (%d %d %d): %d %d\n", index, x, y, z, v.x, v.y );
            }
        }
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( Array3DView< const int2 > array, const char* filename )
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

    retVal = fprintf( fp, "Format: int2\n" );
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
                int2 v = array[ { x, y, z } ];
                fprintf( fp, "[%d] (%d %d %d): %d %d\n", index, x, y, z, v.x, v.y );
            }
        }
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( Array3DView< const int3 > array, const char* filename )
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

    retVal = fprintf( fp, "Format: int2\n" );
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
                int3 v = array[ { x, y, z } ];
                fprintf( fp, "[%d] (%d %d %d): %d %d %d\n", index, x, y, z, v.x, v.y, v.z );
            }
        }
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( Array3DView< const int4 > array, const char* filename )
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

    retVal = fprintf( fp, "Format: int4\n" );
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
                int4 v = array[ { x, y, z } ];
                fprintf( fp, "[%d] (%d %d %d): %d %d %d %d\n",
					index, x, y, z,
					v.x, v.y, v.z, v.w );
            }
        }
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( const DeviceArray1D< int3 >& array, const char* filename )
{
    Array1D< int3 > h_array( array.length() );
    array.copyToHost( h_array.writeView() );
    return saveTXT( h_array, filename );
}

bool saveTXT( const DeviceArray2D< float >& array, const char* filename )
{
    Array2D< float > h_array( array.size() );
    array.copyToHost( h_array.writeView() );
    return ArrayUtils::saveTXT( h_array, filename );
}

bool saveTXT( const DeviceArray2D< float2 >& array, const char* filename )
{
    Array2D< float2 > h_array( array.size() );
    array.copyToHost( h_array.writeView() );
    return saveTXT( h_array, filename );
}

bool saveTXT( const DeviceArray2D< float4 >& array, const char* filename )
{
    Array2D< float4 > h_array( array.size() );
    array.copyToHost( h_array.writeView() );
    return saveTXT( h_array, filename );
}

bool saveTXT( const DeviceArray2D< uchar4 >& array, const char* filename )
{
    Array2D< uchar4 > h_array( array.size() );
    array.copyToHost( h_array.writeView() );
    return saveTXT( h_array, filename );
}

bool saveTXT( const DeviceArray3D< ushort2 >& array, const char* filename )
{
    Array3D< ushort2 > h_array( array.size() );
    array.copyToHost( h_array.writeView() );
    return saveTXT( h_array, filename );
}

bool saveTXT( const DeviceArray3D< int2 >& array, const char* filename )
{
    Array3D< int2 > h_array( array.size() );
    array.copyToHost( h_array.writeView() );
    return saveTXT( h_array, filename );
}

bool saveTXT( const DeviceArray3D< int3 >& array, const char* filename )
{
    Array3D< int3 > h_array( array.size() );
    array.copyToHost( h_array.writeView() );
    return saveTXT( h_array, filename );
}

} } } // arrayutils, cuda, libcgt
