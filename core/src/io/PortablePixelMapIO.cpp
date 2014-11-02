#include "io/PortablePixelMapIO.h"

#include <cassert>
#include <QFile>
#include <QDataStream>
#include <QTextStream>

#include "imageproc/ColorUtils.h"

using namespace libcgt::core::imageproc::colorutils;

// static
bool PortablePixelMapIO::write( QString filename, Array2DView< uint8x3 > image )
{
	QFile outputFile( filename );

	// try to open the file in write only mode
	if( !( outputFile.open( QIODevice::WriteOnly ) ) )
	{
		return false;
	}

	QTextStream outputTextStream( &outputFile );
	outputTextStream.setCodec( "ISO-8859-1" );
	outputTextStream << "P6\n";
	outputTextStream << image.width() << " " << image.height() << "\n";
	outputTextStream << "255\n";

	outputTextStream.flush();

	QDataStream outputDataStream( &outputFile );

	for( int y = 0; y < image.height(); ++y )
	{
		for( int x = 0; x < image.width(); ++x )
		{
			uint8x3 rgb = image( x, y );
			outputDataStream << rgb.x << rgb.y << rgb.z;			
		}
	}
	
	// TODO: error check
	return true;
}

// static
bool PortablePixelMapIO::write( QString filename, Array2DView< Vector3f > image )								  
{
	QFile outputFile( filename );

	// try to open the file in write only mode
	if( !( outputFile.open( QIODevice::WriteOnly ) ) )
	{
		return false;
	}

	QTextStream outputTextStream( &outputFile );
	outputTextStream.setCodec( "ISO-8859-1" );
	outputTextStream << "P6\n";
	outputTextStream << image.width() << " " << image.height() << "\n";
	outputTextStream << "255\n";

	outputTextStream.flush();

	QDataStream outputDataStream( &outputFile );	

	for( int y = 0; y < image.height(); ++y )
	{
		for( int x = 0; x < image.width(); ++x )
		{
			Vector3f rgb = image( x, y );

            outputDataStream << toUInt8( saturate( rgb.x ) );
            outputDataStream << toUInt8( saturate( rgb.y ) );
            outputDataStream << toUInt8( saturate( rgb.z ) );
		}
	}

	return true;
}
