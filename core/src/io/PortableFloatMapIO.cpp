#include "io/PortableFloatMapIO.h"

#include <QFile>
#include <QString>
#include <QTextStream>

#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>

// static
PortableFloatMapIO::PFMData PortableFloatMapIO::read( QString filename )
{
	PFMData output;
	output.valid = false;

	QByteArray cstrFilename = filename.toLocal8Bit();
	FILE* file = fopen( cstrFilename.constData(), "rb" );	

	if( file == nullptr )
	{
		return output;
	}

	const int LINE_LENGTH = 80;
	char line[LINE_LENGTH];

	// Read format.
	fgets( line, LINE_LENGTH, file );

	// grayscale
	if( strcmp( line, "Pf\n" ) == 0 )
	{
		output.nComponents = 1;
	}
	// RGB
	else if( strcmp( line, "PF\n" ) == 0 )
	{
		output.nComponents = 3;
	}
	// RGBA
	else if( strcmp( line, "PF4\n" ) == 0 )
	{
		output.nComponents = 4;
	}
	// Invalid
	else
	{		
		return output;
	}

	// Read dimensions.
	fgets( line, LINE_LENGTH, file );
	int width;
	int height;
	sscanf( line, "%d %d", &width, &height );

	if( width <= 0 || height <= 0 )
	{
		return output;
	}
	
	// Read scale.
	fgets( line, LINE_LENGTH, file );
	sscanf( line, "%f", &( output.scale ) );

	// Allocate memory.
	uint8_t* buffer;
	if( output.nComponents == 1 )
	{
		output.grayscale.resize( width, height );
		buffer = reinterpret_cast< uint8_t* >( output.grayscale.pointer() );
	}
	else if( output.nComponents == 3 )
	{
		output.rgb.resize( width, height );
		output.rgb.fill(Vector3f(1,1,1));
		buffer = reinterpret_cast< uint8_t* >( output.rgb.pointer() );
	}
	else
	{
		output.rgba.resize( width, height );
		buffer = reinterpret_cast< uint8_t* >( output.rgba.pointer() );
	}

	// Read buffer.
	size_t nElementsRead = fread( buffer, output.nComponents * sizeof( float ), width * height, file );
	if( nElementsRead == width * height )
	{
		output.valid = true;
	}

	fclose( file );
	return output;
}

// static
bool PortableFloatMapIO::write( QString filename, Array2DView< float > image )
{
	int w = image.width();
	int h = image.height();

	// use "wb" binary mode to ensure that on Windows,
	// newlines in the header are written out as '\n'
	QByteArray cstrFilename = filename.toLocal8Bit();
	FILE* pFile = fopen( cstrFilename.constData(), "wb" );
	if( pFile == nullptr )
	{
		return false;
	}

	// write header
	fprintf( pFile, "Pf\n%d %d\n-1\n", w, h );

	if( image.packed() )
	{
		fwrite( image.rowPointer( 0 ), sizeof( float ), image.width() * image.height(), pFile );
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
				fwrite( &( image( x, y ) ), sizeof( float ), 1, pFile );
			}
		}
	}

	fclose( pFile );
	return true;
}

// static
bool PortableFloatMapIO::write( QString filename, Array2DView< Vector3f > image )
{
	int w = image.width();
	int h = image.height();

	// use "wb" binary mode to ensure that on Windows,
	// newlines in the header are written out as '\n'
	QByteArray cstrFilename = filename.toLocal8Bit();
	FILE* pFile = fopen( cstrFilename.constData(), "wb" );
	if( pFile == nullptr )
	{
		return false;
	}

	// write header
	fprintf( pFile, "PF\n%d %d\n-1\n", w, h );

	if( image.packed() )
	{
		fwrite( image.rowPointer( 0 ), sizeof( Vector3f ), image.width() * image.height(), pFile );
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
				fwrite( &( image( x, y ) ), sizeof( Vector3f ), 1, pFile );
			}
		}
	}

	fclose( pFile );
	return true;
}

// static
bool PortableFloatMapIO::write( QString filename, Array2DView< Vector4f > image )
{
	int w = image.width();
	int h = image.height();

	// use "wb" binary mode to ensure that on Windows,
	// newlines in the header are written out as '\n'
	QByteArray cstrFilename = filename.toLocal8Bit();
	FILE* pFile = fopen( cstrFilename.constData(), "wb" );
	if( pFile == nullptr )
	{
		return false;
	}

	// write header
	fprintf( pFile, "PF4\n%d %d\n-1\n", w, h );

	if( image.packed() )
	{
		fwrite( image.rowPointer( 0 ), sizeof( Vector4f ), image.width() * image.height(), pFile );
	}
	else if( image.elementsArePacked() )
	{
		for( int y = 0; y < h; ++y )
		{
			fwrite( image.rowPointer( y ), sizeof( Vector4f ), image.width(), pFile );
		}
	}
	else
	{
		for( int y = 0; y < h; ++y )
		{
			for( int x = 0; x < w; ++x )
			{				
				fwrite( &( image( x, y ) ), sizeof( Vector4f ), 1, pFile );
			}
		}
	}

	fclose( pFile );
	return true;
}