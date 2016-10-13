#include <gflags/gflags.h>

#include <core/common/ArrayUtils.h>
#include <core/common/BasicTypes.h>
#include <core/imageproc/ColorMap.h>
#include <core/io/File.h>
#include <core/io/NumberedFilenameBuilder.h>
#include <core/io/PNGIO.h>
#include <core/vecmath/Range1i.h>
#include <camera_wrappers/RGBDStream.h>
#include <third_party/pystring.h>

using namespace libcgt::camera_wrappers;
using namespace libcgt::core;
using namespace libcgt::core::arrayutils;

DEFINE_string( output_format, "DEPTH_MM_U16",
    "Output format. Allowed formats: DEPTH_MM_U16, DEPTH_M_F32" );
DEFINE_double( depth_scale, 1.0,
    "(Float output only)\n"
    "Scale factor 'a' to apply. zOut = a * zIn + b. Default: 1.0." );
DEFINE_double( depth_offset, 0.0,
    "(Float output only)\n"
    "Offset 'b' to apply. zOut = a * zIn + b. Default: 0.0." );

int main( int argc, char* argv[] )
{
    gflags::ParseCommandLineFlags( &argc, &argv, true );
    if( argc < 3 )
    {
        fprintf( stderr, "Usage: %s <src_prefix> <output.rgbd>\n", argv[ 0 ] );
        fprintf( stderr, "Will look for <src_prefix><#####>.png and write it"
            " to output.rgbd.\n" );
        return 1;
    }
    printf( "TODO: let the user specify input and output stream format.\n" );
    printf( "TODO: let the user specify the number of zeros.\n" );
    printf( "TODO: let the user specify start and end index.\n" );

    NumberedFilenameBuilder nfb( argv[1], ".png" );

    int i = 0;
    std::string inputFilename = nfb.filenameForNumber( i );

    int nFrames = INT_MAX;
    //nFrames = 10;

    printf( "Reading: %s\n", inputFilename.c_str() );
    auto pngInput = PNGIO::read( inputFilename );
    if( pngInput.bitDepth != 16 || pngInput.nComponents != 1 )
    {
        fprintf( stderr, "PNG depth inputs must be 16-bit grayscale.\n" );
        return 2;
    }

    Vector2i resolution = pngInput.gray16.size();
    printf( "Input resolution: %d x %d\n", resolution.x, resolution.y );

    std::vector< StreamMetadata > outputMetadata;
    Array2D< float > tmpFloatBuffer;

    if( FLAGS_output_format == "DEPTH_MM_U16" )
    {
        printf( "Output format is DEPTH_MM_U16.\n" );
        outputMetadata.push_back( StreamMetadata{ StreamType::DEPTH,
            PixelFormat::DEPTH_MM_U16, resolution } );
    }
    else if( FLAGS_output_format == "DEPTH_M_F32" )
    {
        printf( "Output format is DEPTH_M_F32.\n" );
        outputMetadata.push_back( StreamMetadata{ StreamType::DEPTH,
            PixelFormat::DEPTH_M_F32, resolution } );
        tmpFloatBuffer.resize( resolution );

        printf( "depth scale = %lf\n", FLAGS_depth_scale );
        printf( "depth offset = %lf\n", FLAGS_depth_offset );
    }
    else
    {
        fprintf( stderr, "Invalid output format.\n" );
        return 3;
    }

    return 0;

    RGBDOutputStream outputStream( outputMetadata, argv[ 2 ] );
    if( !outputStream.isValid() )
    {
        fprintf( stderr, "Unable to open output file %s\n", argv[ 2 ] );
        return 4;
    }

    float a = static_cast< float >( FLAGS_depth_scale );
    float b = static_cast< float >( FLAGS_depth_offset );
    Array1DView< const uint8_t > data;

    while( i < nFrames && File::exists( inputFilename.c_str() ) )
    {
        printf( "Reading: %s\n", inputFilename.c_str() );
        pngInput = PNGIO::read( inputFilename );

        // Munge on the data.
        flipYInPlace( pngInput.gray16.writeView() );

        if( FLAGS_output_format == "DEPTH_MM_U16" )
        {
            data = Array1DView< const uint8_t >( pngInput.gray16,
                resolution.x * resolution.y * sizeof( uint16_t ) );
        }
        else if( FLAGS_output_format == "DEPTH_M_F32" )
        {
            map( pngInput.gray16.readView(), tmpFloatBuffer.writeView(),
                [&] ( uint16_t z )
                {
                    float zFloat = static_cast< float >( z );
                    return a * zFloat + b;
                }
            );
            data = Array1DView< const uint8_t >( tmpFloatBuffer,
                resolution.x * resolution.y * sizeof( float ) );
        }

        bool ok = outputStream.write( 0, i, i, data );
        if( !ok )
        {
            fprintf( stderr, "Error writing frame %d.\n", i );
            return 4;
        }

        ++i;
        inputFilename = nfb.filenameForNumber( i );
    }
    bool ok = outputStream.close();
    if( !ok )
    {
        fprintf( stderr, "Error close file.\n" );
        return 5;
    }

    return 0;
}
