#include "io/PNGIO.h"

#include <QString>

#include "../external/lodepng.h"

#include <common/ArrayUtils.h>
#include <math/BitPacking.h>

PNGIO::PNGData PNGIO::read( QString filename )
{
	PNGIO::PNGData output;
	output.valid = false;

	unsigned error;
	std::vector< unsigned char > encoded;
	unsigned int width;
	unsigned int height;

	// TODO: check error
	QByteArray cstrFilename = filename.toLocal8Bit();
	lodepng::load_file( encoded, cstrFilename.constData() );
	if( encoded.size() == 0 )
	{
		return output;
	}

	lodepng::State state;	
	error = lodepng_inspect( &width, &height, &state, encoded.data(), encoded.size() );
	if( error != 0 )
	{
		return output;
	}

	output.bitDepth = state.info_png.color.bitdepth;
	
	// Tell the decoder to decode to the same color type
	// as the PNG (no conversion), unless it's paletted,
	// in which case it's decoded to RGBA.
	state.info_raw.bitdepth = state.info_png.color.bitdepth;
	state.info_raw.colortype =
		( state.info_png.color.colortype == LCT_PALETTE ) ?
		LCT_RGBA :
		state.info_png.color.colortype;
	
	unsigned char* bits;
	error = lodepng_decode( &bits, &width, &height, &state, encoded.data(), encoded.size() );
	if( error != 0 )
	{
		return output;
	}

	switch( state.info_raw.colortype )
	{
	case LCT_GREY:
		output.nComponents = 1;
		if( output.bitDepth == 8 )
		{
			output.grey8 = Array2D< uint8_t >( bits, width, height );
		}
		else
		{			
			output.grey16 = Array2D< uint16_t >( bits, width, height );
		}
		output.valid = true;
		break;

	case LCT_RGB:
		output.nComponents = 3;
		if( output.bitDepth == 8 )
		{
			output.rgb8 = Array2D< ubyte3 >( bits, width, height );
		}
		else
		{
			output.rgb16 = Array2D< ushort3 >( bits, width, height );
		}
		output.valid = true;
		break;

	case LCT_GREY_ALPHA:
		output.nComponents = 2;
		if( output.bitDepth == 8 )
		{
			output.greyalpha8 = Array2D< ubyte2 >( bits, width, height );
		}
		else
		{
			output.greyalpha16 = Array2D< ushort2 >( bits, width, height );
		}
		output.valid = true;
		break;

	case LCT_RGBA:
		output.nComponents = 4;
		if( output.bitDepth == 8 )
		{
			output.rgba8 = Array2D< ubyte4 >( bits, width, height );
		}
		else
		{
			output.rgba16 = Array2D< ushort4 >( bits, width, height );
		}
		output.valid = true;
		break;

	default:
		// should never get here
		// returns output.valid = false;
		break;
	}

	// Swap endianness of 16-bit outputs
	if( output.bitDepth == 16 )
	{
		Array1DView< uint16_t > bitsView( bits, output.nComponents * width * height );
		BitPacking::byteSwap16( bitsView );
	}

	return output;
}

// static
bool PNGIO::writeRGB( QString filename, Array2DView< const uint8x3 > image )
{
	Array2D< uint8x3 > tmpImage;
	const uint8_t* pSrcPointer;

	if( !image.packed() )
	{
		tmpImage.resize( image.size() );
		ArrayUtils::copy< uint8x3 >( image, tmpImage	);
		pSrcPointer = reinterpret_cast< uint8_t* >( tmpImage.rowPointer( 0 ) );
	}
	else
	{
		pSrcPointer = reinterpret_cast< const uint8_t* >( image.rowPointer( 0 ) );
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
bool PNGIO::writeRGBA( QString filename, Array2DView< const uint8x4 > image )
{
	Array2D< uint8x4 > tmpImage;
	const uint8_t* pSrcPointer;

	if( !image.packed() )
	{
		tmpImage.resize( image.size() );
		ArrayUtils::copy< uint8x4 >( image, tmpImage	);
		pSrcPointer = reinterpret_cast< uint8_t* >( tmpImage.rowPointer( 0 ) );
	}
	else
	{
		pSrcPointer = reinterpret_cast< const uint8_t* >( image.rowPointer( 0 ) );
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