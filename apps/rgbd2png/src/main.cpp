#include <third_party/pystring/pystring.h>
#include <gflags/gflags.h>

#include "libcgt/camera_wrappers/RGBDStream.h"
#include "libcgt/core/common/BasicTypes.h"
#include "libcgt/core/imageproc/ColorMap.h"
#include "libcgt/core/io/NumberedFilenameBuilder.h"
#include "libcgt/core/io/PNGIO.h"
#include "libcgt/core/vecmath/Range1i.h"

using libcgt::camera_wrappers::RGBDInputStream;
using libcgt::camera_wrappers::PixelFormat;
using libcgt::core::imageproc::linearRemapToLuminance;

DEFINE_bool( color, true, "Set true to write color stream (if present)." );
DEFINE_bool( depth, true, "Set true to write depth stream (if present)." );
DEFINE_bool( infrared, true,
    "Set true to write infrared stream  (if present)." );

// --> libcgt::core.
#include <iomanip>
#include <sstream>

// T must be an integral type.
template< typename T >
std::string toZeroFilledString( T x, int width )
{
    std::stringstream stream;
    stream << std::setw( width ) << std::setfill( '0' ) << std::internal << x;
    return stream.str();
}



int main( int argc, char* argv[] )
{
    if( argc < 3 )
    {
        printf( "Usage: %s <flags> <src.rgbd> <output_dir>\n", argv[ 0 ] );
        printf( "Files will be saved to <dir>/<src>_<color|depth|infrared>_<frame_index>_<timestamp>.png" );
        return 1;
    }

    // E.g., "recording_00003.rgbd".
    std::string basename = pystring::os::path::basename( argv[ 1 ] );

    std::string root; // E.g., recording_00003.
    std::string ext; // E.g., ".rgbd".
    pystring::os::path::splitext( root, ext, basename );

    // Make a new root for each stream: new dir + root + stream name
    // E.g. "/dst/recording_00003_color"
    std::string colorRoot = pystring::os::path::join( argv[ 2 ],
        root + "_color" );
    std::string depthRoot = pystring::os::path::join( argv[ 2 ],
        root + "_depth" );
    std::string infraredRoot = pystring::os::path::join( argv[ 2 ],
        root + "_infrared" );

    RGBDInputStream inputStream( argv[ 1 ] );
    if( !inputStream.isValid() )
    {
        fprintf( stderr, "Error reading input %s.\n", argv[ 1 ] );
        return 2;
    }

    // Find color stream.
    int colorStream = -1;
    for( int i = 0; i < inputStream.metadata().size(); ++i )
    {
        // TODO: print "found color stream %d, format is ..."
        if( inputStream.metadata()[ i ].type == StreamType::COLOR )
        {
            colorStream = i;
            break;
        }
    }

    // Find depth stream.
    int depthStream = -1;
    for( int i = 0; i < inputStream.metadata().size(); ++i )
    {
        if( inputStream.metadata()[ i ].type == StreamType::DEPTH )
        {
            depthStream = i;
            break;
        }
    }

    // Find infrared stream.
    int infraredStream = -1;
    for( int i = 0; i < inputStream.metadata().size(); ++i )
    {
        if( inputStream.metadata()[ i ].type == StreamType::INFRARED )
        {
            infraredStream = i;
            break;
        }
    }

    if( colorStream == -1 && depthStream == -1 && infraredStream == -1 )
    {
        fprintf( stderr, "Could not find any streams to convert.\n" );
        return 3;
    }

    const int TIMESTAMP_FIELD_WIDTH = 20;
    NumberedFilenameBuilder colorNFB( colorRoot + "_", "" );
    NumberedFilenameBuilder depthNFB( depthRoot + "_", "" );
    NumberedFilenameBuilder infraredNFB( infraredRoot + "_", "" );
    Array2D< uint8_t > tonemappedDepth;
    if( depthStream != -1 )
    {
        tonemappedDepth.resize(
            inputStream.metadata()[ depthStream ].size );
    }

    Array2D< uint8_t > tonemappedInfrared;
    if( infraredStream != -1 )
    {
        tonemappedInfrared.resize(
            inputStream.metadata()[ infraredStream ].size );
    }

    uint32_t streamId;
    int frameIndex;
    int64_t timestamp;
    Array1DReadView< uint8_t > src =
        inputStream.read( streamId, frameIndex, timestamp );
    while( src.notNull() )
    {
        if( streamId == colorStream && FLAGS_color )
        {
            std::string outputFilename = colorNFB.filenameForNumber(
                frameIndex );
            outputFilename += "_" +
                toZeroFilledString( timestamp, TIMESTAMP_FIELD_WIDTH ) +
                ".png";
            Array2DReadView< uint8x3 > src2D( src.pointer(),
                inputStream.metadata()[ colorStream ].size );
            printf( "Writing color frame to %s\n", outputFilename.c_str() );
            PNGIO::write( outputFilename, src2D );
        }

        if( streamId == depthStream && FLAGS_depth )
        {
            std::string outputFilename = depthNFB.filenameForNumber(
                frameIndex );
            outputFilename += "_" +
                toZeroFilledString( timestamp, TIMESTAMP_FIELD_WIDTH ) +
                ".png";

            Array2DReadView< uint16_t > src2D( src.pointer(),
                inputStream.metadata()[ depthStream ].size );

            // TODO: dump depth to the right format
            // TODO: source min max are not well specified...
            Range1i srcRange = Range1i::fromMinMax( 800, 4000 );
            Range1i dstRange = Range1i::fromMinMax( 51, 256 );
            linearRemapToLuminance( src2D, srcRange, dstRange,
                tonemappedDepth );

            printf( "Writing depth frame to %s\n", outputFilename.c_str() );
            PNGIO::write( outputFilename, tonemappedDepth );
        }

        if( streamId == infraredStream && FLAGS_infrared )
        {
            std::string outputFilename = infraredNFB.filenameForNumber(
                frameIndex );
            outputFilename += "_" +
                toZeroFilledString( timestamp, TIMESTAMP_FIELD_WIDTH ) +
                ".png";

            Array2DReadView< uint16_t > src2D( src.pointer(),
                inputStream.metadata()[ infraredStream ].size );

            Range1i srcRange( 1024 );
            Range1i dstRange( 256 );
            linearRemapToLuminance( src2D, srcRange, dstRange,
                tonemappedInfrared );

            printf( "Writing infrared frame to %s\n", outputFilename.c_str() );
            PNGIO::write( outputFilename, tonemappedInfrared );
        }
        src =
            inputStream.read( streamId, frameIndex, timestamp );
    }
    return 0;
}
