#include "libcgt/core/io/PortablePixelMapIO.h"

#include <cassert>
#include <cstdio>

#include "libcgt/core/imageproc/ColorUtils.h"

using namespace libcgt::core::imageproc;

// static
bool PortablePixelMapIO::writeRGB( const std::string& filename,
    Array2DReadView< uint8x3 > image )
{
    FILE* fp = fopen( filename.c_str(), "wb" );
    if( fp == nullptr )
    {
        return false;
    }

    int nCharsWritten =
        fprintf( fp, "P6\n%d %d\n255\n", image.width(), image.height() );
    if( nCharsWritten < 0 )
    {
        fclose( fp );
        return false;
    }

    if( image.packed() )
    {
        size_t nWritten = fwrite( image.rowPointer( 0 ), sizeof( uint8x3 ),
            image.numElements(), fp );
        if( nWritten != image.numElements() )
        {
            fclose( fp );
            return false;
        }
    }
    else if( image.rowsArePacked() )
    {
        for( int y = 0; y < image.height(); ++y )
        {
            size_t nWritten = fwrite( image.rowPointer( y ), sizeof( uint8x3 ),
                image.width(), fp );
            if( nWritten != image.width() )
            {
                fclose( fp );
                return false;
            }
        }
    }
    else
    {
        for( int y = 0; y < image.height(); ++y )
        {
            for( int x = 0; x < image.width(); ++x )
            {
                size_t nWritten = fwrite( image.elementPointer( { x, y } ),
                    sizeof( uint8x3 ), 1, fp );
                if( nWritten != 1 )
                {
                    fclose( fp );
                    return false;
                }
            }
        }
    }

    fclose( fp );
    return true;
}

// static
bool PortablePixelMapIO::writeRGBText( const std::string& filename,
    Array2DReadView< uint8x3 > image )
{
    FILE* fp = fopen( filename.c_str(), "w" );
    if( fp == nullptr )
    {
        return false;
    }

    int nCharsWritten = 0;

    nCharsWritten =
        fprintf( fp, "P3\n%d %d\n255\n", image.width(), image.height() );
    if( nCharsWritten < 0 )
    {
        fclose( fp );
        return false;
    }

    for( int y = 0; y < image.height(); ++y )
    {
        for( int x = 0; x < image.width(); ++x )
        {
            uint8x3 c = image[ { x, y } ];
            nCharsWritten = fprintf( fp, "%d %d %d ", c.x, c.y, c.z );
            if( nCharsWritten < 0 )
            {
                fclose( fp );
                return false;
            }
        }
    }

    fclose( fp );
    return true;
}
