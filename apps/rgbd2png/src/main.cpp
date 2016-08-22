#include <core/common/BasicTypes.h>
#include <core/io/NumberedFilenameBuilder.h>
#include <core/io/PNGIO.h>
#include <camera_wrappers/RGBDStream.h>
#include <third_party/pystring.h>

using libcgt::camera_wrappers::RGBDInputStream;
using libcgt::camera_wrappers::PixelFormat;

int main( int argc, char* argv[] )
{
    if( argc < 3 )
    {
        printf( "Usage: %s <src.rgbd> <output_dir>\n", argv[ 0 ] );
        printf( "Files will be saved to <dir>/<src>_<frame_index>_<timestamp>.png" );
        return 1;
    }

    std::string root;
    std::string ext;

    std::string basename = pystring::os::path::basename( argv[ 1 ] );
    pystring::os::path::splitext( root, ext, basename );

    std::string newRoot = pystring::os::path::join( argv[ 2 ], root );

    RGBDInputStream inputStream( argv[ 1 ] );
    if( !inputStream.isValid() )
    {
        printf( "Error reading input %s\n" );
        return 2;
    }

    // TODO(jiawen): other formats.
    // Find color stream.
    int colorStream = -1;
    Vector2i srcSize;
    for( int i = 0; i < inputStream.metadata().size(); ++i )
    {
        if( inputStream.metadata()[ i ].format == PixelFormat::RGB_U888 )
        {
            colorStream = i;
            srcSize = inputStream.metadata()[ i ].size;
            break;
        }
    }

    if( colorStream == -1 )
    {
        printf( "Could not find color stream" );
        return 3;
    }

    // TODO(jiawen): make a to_string() that zero-pads.

    NumberedFilenameBuilder nfb( newRoot + "_", "" );
    int i = 0;
    uint32_t streamId;
    int frameIndex;
    int64_t timestamp;
    Array1DView< const uint8_t > src =
        inputStream.read( streamId, frameIndex, timestamp );
    while( src.notNull() )
    {
        // TODO(jiawen): other formats.
        if( streamId == colorStream )
        {
            std::string outputFilename = nfb.filenameForNumber( frameIndex );
            outputFilename += "_" + std::to_string( timestamp ) + ".png";
            Array2DView< const uint8x3 > src2D( src.pointer(), srcSize );
            printf( "Writing frame %d to %s\n", i, outputFilename.c_str() );
            PNGIO::write( outputFilename, src2D );
            ++i;
        }
        src =
            inputStream.read( streamId, frameIndex, timestamp );
    }
    return 0;
}
