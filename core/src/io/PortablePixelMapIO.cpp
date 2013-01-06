#include "io/PortablePixelMapIO.h"

#include <cassert>
#include <QFile>
#include <QDataStream>
#include <QTextStream>
#include <color/ColorUtils.h>

// static
bool PortablePixelMapIO::writeRGB( QString filename, Array2DView< ubyte3 > image )
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
			ubyte3 rgb = image( x, y );
			outputDataStream << rgb.x << rgb.y << rgb.z;			
		}
	}
	
	// TODO: error check
	return true;
}

// static
bool PortablePixelMapIO::writeRGB( QString filename, Array2DView< Vector3f > image )								  
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

			outputDataStream << ColorUtils::floatToUnsignedByte( rgb.x );
			outputDataStream << ColorUtils::floatToUnsignedByte( rgb.y );
			outputDataStream << ColorUtils::floatToUnsignedByte( rgb.z );
		}
	}

	return true;
}
