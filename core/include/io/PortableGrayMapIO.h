#pragma once

#include <common/Array2D.h>

class QString;

class PortableGrayMapIO
{
public:

    // TODO: does not parse comments
    // TODO: use views, uint8_t, uint16_t

	// PGM specifies that maxVal from the file is > 0 and < 65536
	// TODO: reads P5 (binary) only
	// P2 is the text version
	//
	// if maxVal < 256, then the array can be used as is
	// else, then maxVal is guaranteed to be < 65536 and the array is (2 * width x height)
	//   and it should be reinterpreted with:
	//   outputShort = inputUint8_t.reinterpretAs< ushort >()
    // TODO: return a variant
	static bool read( QString filename, Array2D< uint8_t >& output, int& maxVal );
};

#include <QString>

// static
bool PortableGrayMapIO::read( QString filename, Array2D< uint8_t >& output, int& maxVal )
{
	QByteArray cstrFilename = filename.toLocal8Bit();
	FILE* pFile = fopen( cstrFilename.constData(), "rb" );
	if( pFile == nullptr )
	{
		return false;
	}
	
	char str[255];

	int width;
	int height;

	fscanf( pFile, " %s", str );

	if( ferror( pFile ) || feof( pFile ) ||
		( strcmp( str, "P5" ) != 0 ) )
	{
		fclose( pFile );
		return false;
	}

	// TODO: parse comments

	int nMatches;
	
	nMatches = fscanf( pFile, " %d", &width );
	if( ferror( pFile ) || feof( pFile ) ||
		( nMatches < 1 ) )
	{
		fclose( pFile );
		return false;
	}

	nMatches = fscanf( pFile, " %d", &height );
	if( ferror( pFile ) || feof( pFile ) ||
		( nMatches < 1 ) )
	{
		fclose( pFile );
		return false;
	}

	nMatches = fscanf( pFile, " %d", &maxVal );
	if( ferror( pFile ) || feof( pFile ) ||
		( nMatches < 1 ) )
	{
		fclose( pFile );
		return false;
	}

	// there must be exactly one whitespace character after the maxVal specifier
	int whitespace = getc( pFile );
	if( !isspace( whitespace ) )
	{
		fclose( pFile );
		return false;
	}

	if( maxVal < 0 || maxVal > 65535 )
	{
		fclose( pFile );
		return false;
	}
	else if( maxVal < 256 )
	{
		output.resize( width, height );
		fread( output, 1, width * height, pFile );
	}
	else // maxVal < 65536
	{
		output.resize( 2 * width, height );
		fread( output, 2, width * height, pFile );
	}	

	fclose( pFile );
	return true;
}
