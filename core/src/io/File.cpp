#include "io/File.h"

#include <cstdio>
#include <fstream>

// static
bool File::exists( const char* filename )
{
    FILE* filePointer = fopen( filename, "r" );
    if( filePointer != NULL )
    {
        fclose( filePointer );
        return true;
    }
    return false;
}

// static
size_t File::size( const char* filename )
{
    FILE* filePointer = fopen( filename, "rb" );
    if( filePointer == NULL )
    {
        return 0;
    }

    // get the file size
    int seekResult = fseek( filePointer, 0L, SEEK_END );
    if( seekResult != 0 )
    {
        return 0;
    }

    long fileSize = ftell( filePointer );
    if( fileSize == -1 )
    {
        return 0;
    }
    return fileSize;
}

// static
std::string File::readTextFile( const char* filename )
{
    std::string output;

    size_t fileSize = size( filename );
    if( fileSize > 0 )
    {
        output.resize( fileSize );
        FILE* filePointer = fopen( filename, "r" );
        size_t bytesRead = fread( &( output[0] ), sizeof( char ), fileSize, filePointer );

        if( ferror( filePointer ) != 0 )
        {
            output = std::string();
        }

        fclose( filePointer );
    }
    return output;
}

// static
std::vector< uint8_t > File::readBinaryFile( const char* filename )
{
    std::vector< uint8_t > output;

    size_t fileSize = size( filename );
    if( fileSize > 0 )
    {
        output.resize( fileSize );
        FILE* filePointer = fopen( filename, "rb" );
        size_t bytesRead = fread( output.data(), 1, fileSize, filePointer );
        if( ( bytesRead != fileSize ) ||
            ( ferror( filePointer ) != 0 ) )
        {
            output = std::vector< uint8_t >();
        }

        fclose( filePointer );
    }

    return output;
}

// static
std::vector< std::string > File::readLines( const char* filename )
{
    std::vector< std::string > output;
    std::ifstream infile( filename );

    std::string line;
    while( std::getline( infile, line ) )
    {
        output.push_back( line );
    }

    return output;
}