#include "PortableGrayMapIO.h"

// static
PortableGrayMapIO::PGMData PortableGrayMapIO::read( const char* filename )
{
    PortableGrayMapIO::PGMData output;
    output.valid = false;

    FILE* pFile = fopen( filename, "rb" );
    if( pFile == nullptr )
    {
        return output;
    }

    char str[255];

    int width;
    int height;
    int maxVal;

    fscanf( pFile, " %s", str );

    if( ferror( pFile ) || feof( pFile ) ||
        ( strcmp( str, "P5" ) != 0 ) )
    {
        fclose( pFile );
        return output;
    }

    // TODO: parse comments

    int nMatches;

    nMatches = fscanf( pFile, " %d", &width );
    if( ferror( pFile ) || feof( pFile ) ||
        ( nMatches < 1 ) )
    {
        fclose( pFile );
        return output;
    }

    nMatches = fscanf( pFile, " %d", &height );
    if( ferror( pFile ) || feof( pFile ) ||
        ( nMatches < 1 ) )
    {
        fclose( pFile );
        return output;
    }

    nMatches = fscanf( pFile, " %d", &maxVal );
    if( ferror( pFile ) || feof( pFile ) ||
        ( nMatches < 1 ) )
    {
        fclose( pFile );
        return output;
    }

    // There must be exactly one whitespace character after the maxVal specifier.
    int whitespace = getc( pFile );
    if( !isspace( whitespace ) )
    {
        fclose( pFile );
        return output;
    }

    if( maxVal < 0 || maxVal > 65535 )
    {
        fclose( pFile );
        return output;
    }

    output.maxVal = maxVal;
    if( maxVal < 256 )
    {
        output.gray8.resize( { width, height } );
        size_t nElementsRead = fread( output.gray8, 1, width * height, pFile );
        if( nElementsRead == width * height )
        {
            output.valid = true;
        }
    }
    else // maxVal < 65536
    {
        output.gray16.resize( { width, height } );
        size_t nElementsRead = fread( output.gray16, 2, width * height, pFile );
        if( nElementsRead == width * height )
        {
            output.valid = true;
        }
    }

    fclose( pFile );
    return output;
}


// static
bool PortableGrayMapIO::writeBinary( const char* filename,
    Array2DReadView< uint8_t > image )
{
    if( image.width() < 0 || image.height() < 0 )
    {
        return false;
    }

    FILE* pf = fopen( filename, "wb" );

    fprintf( pf, "P5\n%d %d\n255\n", image.width(), image.height() );

    if( image.packed() )
    {
        fwrite( image, 1, image.width() * image.height(), pf );
    }
    else if( image.elementsArePacked() )
    {
        for( int y = 0; y < image.height(); ++y )
        {
            fwrite( image.rowPointer( y ), 1, image.width(), pf );
        }
    }
    else
    {
        for( int y = 0; y < image.height(); ++y )
        {
            for( int x = 0; x < image.width(); ++x )
            {
                fwrite( image.elementPointer( { x, y } ), 1, 1, pf );
            }
        }
    }

    bool succeeded = ( ferror( pf ) != 0 );
    fclose( pf );
    return succeeded;
}

// static
bool PortableGrayMapIO::writeBinary( const char* filename,
    Array2DReadView< uint16_t > image )
{
    if( image.width() < 0 || image.height() < 0 )
    {
        return false;
    }

    FILE* pf = fopen( filename, "wb" );

    fprintf( pf, "P5\n%d %d\n65535\n", image.width(), image.height() );

    if( image.packed() )
    {
        fwrite( image, 2, image.width() * image.height(), pf );
    }
    else if( image.elementsArePacked() )
    {
        for( int y = 0; y < image.height(); ++y )
        {
            fwrite( image.rowPointer( y ), 2, image.width(), pf );
        }
    }
    else
    {
        for( int y = 0; y < image.height(); ++y )
        {
            for( int x = 0; x < image.width(); ++x )
            {
                fwrite( image.elementPointer( { x, y } ), 2, 1, pf );
            }
        }
    }

    bool succeeded = ( ferror( pf ) != 0 );
    fclose( pf );
    return succeeded;
}
