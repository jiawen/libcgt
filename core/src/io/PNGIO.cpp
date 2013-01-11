#include "io/PNGIO.h"

#include <QString>

#include "../external/lodepng.h"

#include "common/ArrayUtils.h"

// static
bool PNGIO::writeRGB( QString filename, Array2DView< ubyte3 > image )
{
	Array2D< ubyte3 > tmpImage;
	ubyte* pSrcPointer;

	if( !image.packed() )
	{
		tmpImage.resize( image.size() );
		ArrayUtils::copy< ubyte3 >( image, tmpImage	);
		pSrcPointer = reinterpret_cast< ubyte* >( tmpImage.rowPointer( 0 ) );

		for( int y = 0; y < 512; ++y )
		{
			for( int x = 0; x < 512; ++x )
			{
				if( image( x, y ).x != tmpImage( x, y ).x ||
					image( x, y ).y != tmpImage( x, y ).y ||
					image( x, y ).z != tmpImage( x, y ).z  )
				{
					printf( "foo!\n" );
				}
			}
		}
	}

	QByteArray cstrFilename = filename.toLocal8Bit();
	unsigned int errVal = lodepng_encode24_file
	(
		cstrFilename.constData(),
		pSrcPointer,
		image.width(),
		image.height()
	);

	bool succeeded = ( errVal == 0 );
	return succeeded;
}

// static
bool PNGIO::writeRGBA( QString filename, Array2DView< ubyte4 > image )
{
	Array2D< ubyte4 > tmpImage;
	ubyte* pSrcPointer;

	if( !image.packed() )
	{
		tmpImage.resize( image.size() );
		ArrayUtils::copy< ubyte4 >( image, tmpImage	);
		pSrcPointer = reinterpret_cast< ubyte* >( tmpImage.rowPointer( 0 ) );
	}
	else
	{
		pSrcPointer = reinterpret_cast< ubyte* >( image.rowPointer( 0 ) );
	}

	QByteArray cstrFilename = filename.toLocal8Bit();
	unsigned int errVal = lodepng_encode32_file
	(
		cstrFilename.constData(),
		pSrcPointer,
		image.width(),
		image.height()
	);

	bool succeeded = ( errVal == 0 );
	return succeeded;
}