#include "io/PNGIO.h"

#include "../third_party/lodepng.h"

#include <common/Array1DView.h>
#include <common/ArrayUtils.h>
#include <math/BitPacking.h>

PNGIO::PNGData PNGIO::read( const std::string& filename )
{
    PNGIO::PNGData output;
    output.valid = false;

    unsigned error;
    std::vector< unsigned char > encoded;
    unsigned int width;
    unsigned int height;

    // TODO: check error
    lodepng::load_file( encoded, filename );
    if( encoded.size() == 0 )
    {
        return output;
    }

    lodepng::State state;
    error = lodepng_inspect( &width, &height, &state,
        encoded.data(), encoded.size() );
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
    error = lodepng_decode( &bits, &width, &height, &state,
        encoded.data(), encoded.size() );
    if( error != 0 )
    {
        return output;
    }

    Vector2i size{ static_cast< int >( width ), static_cast< int >( height ) };
    switch( state.info_raw.colortype )
    {
    case LCT_GREY:
        output.nComponents = 1;
        if( output.bitDepth == 8 )
        {
            output.gray8 = Array2D< uint8_t >( bits, size );
        }
        else
        {
            output.gray16 = Array2D< uint16_t >( bits, size );
        }
        output.valid = true;
        break;

    case LCT_RGB:
        output.nComponents = 3;
        if( output.bitDepth == 8 )
        {
            output.rgb8 = Array2D< uint8x3 >( bits, size );
        }
        else
        {
            output.rgb16 = Array2D< uint16x3 >( bits, size );
        }
        output.valid = true;
        break;

    case LCT_GREY_ALPHA:
        output.nComponents = 2;
        if( output.bitDepth == 8 )
        {
            output.grayalpha8 = Array2D< uint8x2 >( bits, size );
        }
        else
        {
            output.grayalpha16 = Array2D< uint16x2 >( bits, size );
        }
        output.valid = true;
        break;

    case LCT_RGBA:
        output.nComponents = 4;
        if( output.bitDepth == 8 )
        {
            output.rgba8 = Array2D< uint8x4 >( bits, size );
        }
        else
        {
            output.rgba16 = Array2D< uint16x4 >( bits, size );
        }
        output.valid = true;
        break;

    default:
        // Should never get here.
        // Returns output.valid = false.
        break;
    }

    // Swap endianness of 16-bit outputs.
    if( output.bitDepth == 16 )
    {
        Array1DView< uint16_t > bitsView( bits,
            output.nComponents * width * height );
        BitPacking::byteSwap16( bitsView, bitsView );
    }

    return output;
}

// static
bool PNGIO::write( const std::string& filename,
    Array2DView< const uint8_t > image )
{
    Array2D< uint8_t > tmpImage;
    const uint8_t* srcPointer;

    if( !image.packed() )
    {
        tmpImage.resize( image.size() );
        libcgt::core::arrayutils::copy< uint8_t >( image, tmpImage );
        srcPointer = reinterpret_cast< const uint8_t* >( tmpImage.pointer() );
    }
    else
    {
        srcPointer = reinterpret_cast< const uint8_t* >( image.pointer() );
    }

    unsigned int errVal = lodepng::encode(
        filename, srcPointer, image.width(), image.height(), LCT_GREY );
    bool succeeded = ( errVal == 0 );
    return succeeded;
}

// static
bool PNGIO::write( const std::string& filename,
    Array2DView< const uint8x3 > image )
{
    Array2D< uint8x3 > tmpImage;
    const uint8_t* srcPointer;

    if( !image.packed() )
    {
        tmpImage.resize( image.size() );
        libcgt::core::arrayutils::copy< uint8x3 >( image, tmpImage );
        srcPointer = reinterpret_cast< const uint8_t* >( tmpImage.pointer() );
    }
    else
    {
        srcPointer = reinterpret_cast< const uint8_t* >( image.pointer() );
    }

    unsigned int errVal = lodepng::encode(
        filename, srcPointer, image.width(), image.height(), LCT_RGB );
    bool succeeded = ( errVal == 0 );
    return succeeded;
}

// static
bool PNGIO::write( const std::string& filename,
    Array2DView< const uint8x4 > image )
{
    Array2D< uint8x4 > tmpImage;
    const uint8_t* srcPointer;

    if( !image.packed() )
    {
        tmpImage.resize( image.size() );
        libcgt::core::arrayutils::copy< uint8x4 >( image, tmpImage );
        srcPointer = reinterpret_cast< const uint8_t* >( tmpImage.pointer() );
    }
    else
    {
        srcPointer = reinterpret_cast< const uint8_t* >( image.pointer() );
    }

    unsigned int errVal = lodepng::encode(
        filename, srcPointer, image.width(), image.height(), LCT_RGBA );
    bool succeeded = ( errVal == 0 );
    return succeeded;
}

// static
bool PNGIO::write( const std::string& filename,
    Array2DView< const uint16_t > image )
{
    // TODO: use BitPacking::byteSwap16 on the buffer once it supports a
    // destination.
    Array2D< uint16_t > tmpImage( image.size() );
    libcgt::core::arrayutils::map( image, tmpImage.writeView(),
        [&] ( uint16_t z )
        {
            return BitPacking::byteSwap16( z );
        }
    );

    const uint8_t* srcPointer = reinterpret_cast< const uint8_t* >(
        tmpImage.pointer() );
    unsigned int errVal = lodepng::encode(
        filename, srcPointer, image.width(), image.height(), LCT_GREY, 16U );
    bool succeeded = ( errVal == 0 );
    return succeeded;
}
