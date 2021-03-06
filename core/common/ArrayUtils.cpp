#include "libcgt/core/common/ArrayUtils.h"

#include <cstdlib>

namespace libcgt { namespace core { namespace arrayutils {

template<>
bool fill( Array2DWriteView< uint8_t > view, const uint8_t& value )
{
    if( view.isNull() )
    {
        return false;
    }

    if( view.packed() )
    {
        memset( view.pointer(), value, view.numElements() );
        return true;
    }

    if( view.elementsArePacked() )
    {
        // Fill w bytes at a time.
        int w = view.width();
        for( int y = 0; y < view.height(); ++y )
        {
            memset( view.rowPointer( y ), value, w );
        }
        return true;
    }

    // Nothing is packed, iterate.
    int ne = view.numElements();
    for( int k = 0; k < ne; ++k )
    {
        view[ k ] = value;
    }
    return true;
}

Array1DReadView< uint8_t > readViewOf( const std::string& s )
{
    return Array1DReadView< uint8_t >( s.data(), s.length() );
}

} } } // namespace arrayutils, core, libcgt

bool saveTXT( Array1DReadView< int16_t > view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
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

    for( size_t i = 0; i < n; ++i )
    {
        int val = view[ i ];
        fprintf( fp, "[%zd]: %d\n", i, val );
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( Array1DReadView< int32_t > view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
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

    for( size_t i = 0; i < n; ++i )
    {
        int v = view[ i ];
        fprintf( fp, "[%zd]: %d\n", i, v );
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( Array1DReadView< Vector3i >& view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
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

    for( size_t i = 0; i < n; ++i )
    {
        Vector3i v = view[i];
        fprintf( fp, "[%zd]: %d %d %d\n", i, v.x, v.y, v.z );
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( Array1DReadView< float >& view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
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

    for( size_t i = 0; i < n; ++i )
    {
        float v = view[ i ];
        fprintf( fp, "[%zd]: %f\n", i, v );
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( Array1DReadView< Vector2f >& view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
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

    for( size_t i = 0; i < n; ++i )
    {
        Vector2f v = view[ i ];
        fprintf( fp, "[%zd]: %f %f\n", i, v.x, v.y );
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( Array1DReadView< Vector3f >& view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
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

    for( size_t i = 0; i < n; ++i )
    {
        Vector3f v = view[i];
        fprintf( fp, "[%zd]: %f %f %f\n", i, v.x, v.y, v.z );
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( Array1DReadView< Vector4f >& view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
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

    for( size_t i = 0; i < n; ++i )
    {
        Vector4f v = view[i];
        fprintf( fp, "[%zd]: %f %f %f %f\n", i, v.x, v.y, v.z, v.w );
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( Array2DReadView< uint8_t > view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
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
            fprintf( fp, "[%d] (%d %d): %u\n", index, x, y, v );
        }
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( Array2DReadView< uint8x4 > view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
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
            fprintf( fp, "[%d] (%d %d): %u %u %u %u\n",
                index, x, y, v.x, v.y, v.z, v.w );
        }
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( Array2DReadView< int16_t > view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
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

bool saveTXT( Array2DReadView< float > view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
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

bool saveTXT( Array2DReadView< Vector2f > view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
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

    retVal = fprintf( fp, "Format: float2\n" );
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
            Vector2f v = view[ { x, y } ];
            fprintf( fp, "[%d] (%d %d): %f %f\n",
                index, x, y, v.x, v.y );
        }
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( Array2DReadView< Vector3f > view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
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

    retVal = fprintf( fp, "Format: float3\n" );
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
            Vector3f v = view[ { x, y } ];
            fprintf( fp, "[%d] (%d %d): %f %f %f\n",
                index, x, y, v.x, v.y, v.z );
        }
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( Array2DReadView< Vector4f > view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
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
            fprintf( fp, "[%d] (%d %d): %f %f %f %f\n",
                index, x, y, v.x, v.y, v.z, v.w );
        }
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( Array3DReadView< uint16x2 > view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
    if( fp == NULL )
    {
        return false;
    }

    int retVal;

    retVal = fprintf( fp, "Size: %d x %d x %d\n",
        view.width(), view.height(), view.depth() );
    if( retVal < 0 )
    {
        return false;
    }

    retVal = fprintf( fp, "Format: ushort2\n" );
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
                uint16x2 v = view[ { x, y, z } ];
                fprintf( fp, "[%d] (%d %d %d): %u %u\n",
                    index, x, y, z, v.x, v.y );
            }
        }
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( Array3DReadView< Vector2i > view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
    if( fp == NULL )
    {
        return false;
    }

    int retVal;

    retVal = fprintf( fp, "Size: %d x %d x %d\n",
        view.width(), view.height(), view.depth() );
    if( retVal < 0 )
    {
        return false;
    }

    retVal = fprintf( fp, "Format: int2\n" );
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
                Vector2i v = view[ { x, y, z } ];
                fprintf( fp, "[%d] (%d %d %d): %d %d\n",
                    index, x, y, z, v.x, v.y );
            }
        }
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( Array3DReadView< Vector3i > view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
    if( fp == NULL )
    {
        return false;
    }

    int retVal;

    retVal = fprintf( fp, "Size: %d x %d x %d\n",
        view.width(), view.height(), view.depth() );
    if( retVal < 0 )
    {
        return false;
    }

    retVal = fprintf( fp, "Format: int3\n" );
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
                Vector3i v = view[ { x, y, z } ];
                fprintf( fp, "[%d] (%d %d %d): %d %d %d\n",
                    index, x, y, z, v.x, v.y, v.z );
            }
        }
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( Array3DReadView< Vector4i > view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
    if( fp == NULL )
    {
        return false;
    }

    int retVal;

    retVal = fprintf( fp, "Size: %d x %d x %d\n",
        view.width(), view.height(), view.depth() );
    if( retVal < 0 )
    {
        return false;
    }

    retVal = fprintf( fp, "Format: int4\n" );
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
                Vector4i v = view[ { x, y, z } ];
                fprintf( fp, "[%d] (%d %d %d): %d %d %d %d\n",
                    index, x, y, z, v.x, v.y, v.z, v.w );
            }
        }
    }

    retVal = fclose( fp );
    return( retVal == 0 );
}

bool saveTXT( Array3DReadView< float > view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
    if( fp == NULL )
    {
        return false;
    }

    int retVal;

    retVal = fprintf( fp, "Size: %d x %d x %d\n",
        view.width(), view.height(), view.depth() );
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

bool saveTXT( Array3DReadView< Vector2f > view,
    const std::string& filename )
{
    FILE* fp = fopen( filename.c_str(), "w" );
    if( fp == NULL )
    {
        return false;
    }

    int retVal;

    retVal = fprintf( fp, "Size: %d x %d x %d\n",
        view.width(), view.height(), view.depth() );
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
