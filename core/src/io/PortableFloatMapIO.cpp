#include "io/PortableFloatMapIO.h"

// static
PortableFloatMapIO::PFMData PortableFloatMapIO::read( const std::string& filename )
{
    PortableFloatMapIO::PFMData output;
    output.valid = false;

    FILE* fp = fopen( filename.c_str(), "rb" );
    if( fp == nullptr )
    {
        return output;
    }

    char buffer[ 80 ];

    // Read the top level header
    char* readStatus = fgets( buffer, 80, fp );
    if( readStatus == nullptr )
    {
        return output;
    }

    int nComponents = 0;
    if( strncmp( buffer, "Pf", 2 ) == 0 )
    {
        nComponents = 1;
    }
    else if( strncmp( buffer, "PF4", 3 ) == 0 )
    {
        nComponents = 4;
    }
    else if( strncmp( buffer, "PF", 2 ) == 0 )
    {
        nComponents = 3;
    }
    else
    {
        fclose( fp );
        return output;
    }

    // TODO: read comments: lines that begin with #
    int width;
    int height;
    float scale;
    int nMatches = fscanf( fp, "%d %d %f\n", &width, &height, &scale );
    if( nMatches != 3 || width <= 0 || height <= 0 )
    {
        fclose( fp );
        return output;
    }

    float* floatData = new float[ nComponents * width * height ];
    size_t nElementsRead = fread( floatData, nComponents * sizeof( float ), width * height, fp );
    if( nElementsRead != width * height )
    {
        delete[] floatData;
        fclose( fp );
        return output;
    }

    if( nComponents == 1 )
    {
        output.grayscale = Array2D< float >( floatData, { width, height } );
    }
    else if( nComponents == 3 )
    {
        output.rgb = Array2D< Vector3f >( floatData, { width, height } );
    }
    if( nComponents == 4 )
    {
        output.rgba = Array2D< Vector4f >( floatData, { width, height } );
    }

    output.valid = true;
    output.nComponents = nComponents;
    output.scale = scale;
    fclose( fp );
    return output;
}

// static
bool PortableFloatMapIO::write( const std::string& filename, Array2DView< const float > image )
{
    int w = image.width();
    int h = image.height();

    // use "wb" binary mode to ensure that on Windows,
    // newlines in the header are written out as '\n'
    FILE* pFile = fopen( filename.c_str(), "wb" );
    if( pFile == nullptr )
    {
        return false;
    }

    // write header
    int nCharsWritten = fprintf( pFile, "Pf\n%d %d\n-1\n", w, h );
    if( nCharsWritten < 0 )
    {
        fclose( pFile );
        return false;
    }

    if( image.packed() )
    {
        fwrite( image.rowPointer( 0 ), sizeof( float ), image.numElements(), pFile );
    }
    else if( image.elementsArePacked() )
    {
        for( int y = 0; y < h; ++y )
        {
            fwrite( image.rowPointer( y ), sizeof( float ), image.width(), pFile );
        }
    }
    else
    {
        for( int y = 0; y < h; ++y )
        {
            for( int x = 0; x < w; ++x )
            {
                fwrite( image.elementPointer( { x, y } ), sizeof( float ), 1, pFile );
            }
        }
    }

    fclose( pFile );
    return true;
}

// static
bool PortableFloatMapIO::write( const std::string& filename, Array2DView< const Vector3f > image )
{
    int w = image.width();
    int h = image.height();

    // use "wb" binary mode to ensure that on Windows,
    // newlines in the header are written out as '\n'
    FILE* pFile = fopen( filename.c_str(), "wb" );
    if( pFile == nullptr )
    {
        return false;
    }

    // write header
    int nCharsWritten = fprintf( pFile, "PF\n%d %d\n-1\n", w, h );
    if( nCharsWritten < 0 )
    {
        fclose( pFile );
        return false;
    }

    if( image.packed() )
    {
        fwrite( image.rowPointer( 0 ), sizeof( Vector3f ), image.numElements(), pFile );
    }
    else if( image.elementsArePacked() )
    {
        for( int y = 0; y < h; ++y )
        {
            fwrite( image.rowPointer( y ), sizeof( Vector3f ), image.width(), pFile );
        }
    }
    else
    {
        for( int y = 0; y < h; ++y )
        {
            for( int x = 0; x < w; ++x )
            {
                fwrite( image.elementPointer( { x, y } ), sizeof( Vector3f ), 1, pFile );
            }
        }
    }

    fclose( pFile );
    return true;
}

// static
bool PortableFloatMapIO::write( const std::string& filename, Array2DView< const Vector4f > image )
{
    int w = image.width();
    int h = image.height();

    // use "wb" binary mode to ensure that on Windows,
    // newlines in the header are written out as '\n'
    FILE* pFile = fopen( filename.c_str(), "wb" );
    if( pFile == nullptr )
    {
        return false;
    }

    // write header
    int nCharsWritten = fprintf( pFile, "PF4\n%d %d\n-1\n", w, h );
    if( nCharsWritten < 0 )
    {
        fclose( pFile );
        return false;
    }

    // All at once.
    if( image.packed() )
    {
        fwrite( image.rowPointer( 0 ), sizeof( Vector4f ), image.numElements(), pFile );
    }
    // Row by Row.
    else if( image.elementsArePacked() )
    {
        for( int y = 0; y < h; ++y )
        {
            fwrite( image.rowPointer( y ), sizeof( Vector4f ), image.width(), pFile );
        }
    }
    // Element by element.
    else
    {
        for( int y = 0; y < h; ++y )
        {
            for( int x = 0; x < w; ++x )
            {
                fwrite( image.elementPointer( { x, y } ), sizeof( Vector4f ), 1, pFile );
            }
        }
    }

    fclose( pFile );
    return true;
}
