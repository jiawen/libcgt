#include "io/PortableFloatMapIO.h"

#include <QString>

#include <vecmath/Vector3f.h>
#include <vecmath/Vector4f.h>

// static
bool PortableFloatMapIO::writeGrayscale( QString filename, Array2DView< float > image )
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
bool PortableFloatMapIO::writeRGB( QString filename, Array2DView< Vector3f > image )
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
bool PortableFloatMapIO::writeRGBA( QString filename, Array2DView< Vector4f > image )
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