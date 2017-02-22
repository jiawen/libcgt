#include <gflags/gflags.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <third_party/pystring/pystring.h>

#include "libcgt/camera_wrappers/RGBDStream.h"
#include "libcgt/core/cameras/Intrinsics.h"
#include "libcgt/core/common/BasicTypes.h"
#include "libcgt/core/imageproc/ColorMap.h"
#include "libcgt/core/io/NumberedFilenameBuilder.h"
#include "libcgt/core/io/PNGIO.h"
#include "libcgt/core/vecmath/Range1i.h"
#include "libcgt/opencv_interop/ArrayUtils.h"

using libcgt::camera_wrappers::RGBDInputStream;
using libcgt::camera_wrappers::PixelFormat;
using libcgt::core::cameras::Intrinsics;
using libcgt::core::imageproc::linearRemapToLuminance;
using libcgt::opencv_interop::array2DViewAsCvMat;

// TODO: write usage
// rgbdtool --list_streams --input <input.rgbd>
// rgbdtool --dump_frames [--undistort] --input <input.rgbd> --stream <stream> --output <dir>
// rgbdtool --dump_timestamps --input <input.rgbd> --stream <stream> [--output <file>]

DEFINE_bool( list_streams, false,
    "Set this flag to list what streams are present in the file." );
DEFINE_bool( dump_frames, false,
    "Set this flag to write undistorted frames to the given directory." );
DEFINE_bool( dump_timestamps, false,
    "Set this flag to dump \"frame_index, timestamp_ns\" pairs (sans quotes). "
    "If --output is not set, writes to stdout.\n" );

// Options.
DEFINE_bool( undistort, false,
    "Set this flag to write undistorted frames to the given directory." );
DEFINE_string( calibration, "",
    "If using --undistort, this is required and must point to the .yaml "
    "calibration file for that stream." );
DEFINE_uint32( stream, 0,
    "id of stream to process." );

DEFINE_string( input, "", "Input filename (.rgbd)." );
DEFINE_string( output, "",
    "Output file or directory name (depending on mode)." );

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

std::string toString( StreamType type )
{
    switch( type )
    {
    case StreamType::UNKNOWN:
        return "UNKNOWN";
    case StreamType::COLOR:
        return "COLOR";
    case StreamType::DEPTH:
        return "DEPTH";
    case StreamType::INFRARED:
        return "INFRARED";
    default:
        return "UNKNOWN";
    }
}

std::string toString( PixelFormat format )
{
    switch( format )
    {
    case PixelFormat::INVALID:
        return "INVALID";

    case PixelFormat::DEPTH_MM_U16:
        return "DEPTH_MM_U16";
    case PixelFormat::DEPTH_M_F32:
        return "DEPTH_M_F32";
    case PixelFormat::DEPTH_M_F16:
        return "DEPTH_M_F16";

    case PixelFormat::RGBA_U8888:
        return "RGBA_U8888";
    case PixelFormat::RGB_U888:
        return "RGB_U888";
    case PixelFormat::BGRA_U8888:
        return "BGRA_U8888";
    case PixelFormat::BGR_U888:
        return "BGR_U888";

    case PixelFormat::GRAY_U8:
        return "GRAY_U8";
    case PixelFormat::GRAY_U16:
        return "GRAY_U16";
    case PixelFormat::GRAY_U32:
        return "GRAY_U32";

    case PixelFormat::DISPARITY_S8:
        return "DISPARITY_S8";
    case PixelFormat::DISPARITY_S16:
        return "DISPARITY_S16";
    case PixelFormat::DISPARITY_S32:
        return "DISPARITY_S32";
    case PixelFormat::DISPARITY_F16:
        return "DISPARITY_F16";
    case PixelFormat::DISPARITY_F32:
        return "DISPARITY_F32";

    case PixelFormat::DEPTH_UNCALIBRATED_S8:
        return "DEPTH_UNCALIBRATED_S8";
    case PixelFormat::DEPTH_UNCALIBRATED_U8:
        return "DEPTH_UNCALIBRATED_U8";
    case PixelFormat::DEPTH_UNCALIBRATED_S16:
        return "DEPTH_UNCALIBRATED_S16";
    case PixelFormat::DEPTH_UNCALIBRATED_U16:
        return "DEPTH_UNCALIBRATED_U16";
    case PixelFormat::DEPTH_UNCALIBRATED_S32:
        return "DEPTH_UNCALIBRATED_S32";
    case PixelFormat::DEPTH_UNCALIBRATED_U32:
        return "DEPTH_UNCALIBRATED_U32";
    case PixelFormat::DEPTH_UNCALIBRATED_F16:
        return "DEPTH_UNCALIBRATED_F16";
    case PixelFormat::DEPTH_UNCALIBRATED_F32:
        return "DEPTH_UNCALIBRATED_F32";

    default:
        return "UNKNOWN";
    }
}

// TODO: move this into opencv_interop.
// TODO: make opencv_interop capable of rectifying without OpenCV by
// initializing our own undistortion map.
struct CvCameraCalibrationParams
{
    cv::Size imageSize;
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;

    cv::Mat newCameraMatrix;

    cv::Mat map0;
    cv::Mat map1;
};

bool LoadCvCameraCalibrationParams( const std::string& filename,
    CvCameraCalibrationParams& output )
{
    if( filename == "" )
    {
        return false;
    }

    cv::FileStorage fs( filename, cv::FileStorage::READ );
    if( !fs.isOpened() )
    {
        return false;
    }

    fs[ "imageSize" ] >> output.imageSize;
    fs[ "cameraMatrix_cv" ] >> output.cameraMatrix;
    fs[ "distCoeffs" ] >> output.distCoeffs;
    fs[ "newCameraMatrix_cv" ] >> output.newCameraMatrix;

    cv::initUndistortRectifyMap( output.cameraMatrix, output.distCoeffs,
        cv::Mat(), // R = empty aka identity: we are not doing a stereo rectification.
        output.newCameraMatrix, output.imageSize,
        CV_32FC1, output.map0, output.map1 );

    return true;
}

void listStreams( const RGBDInputStream& stream )
{
    const auto& metadata = stream.metadata();
    printf( "# streams: %llu\n", metadata.size() );
    printf( "\n" );
    for( size_t i = 0; i < metadata.size(); ++i )
    {
        const auto& streamMetadata = metadata[i];
        Vector2i res = streamMetadata.size;

        printf( "stream_id: %llu\n", i );
        printf( "type: %s\n", toString( streamMetadata.type ).c_str() );
        printf( "pixel_format: %s\n",
            toString( streamMetadata.format ).c_str() );
        printf( "resolution: %d x %d\n", res.x, res.y );
        printf( "\n" );
    }
}

// TODO: Move all of this into a LensUndistorter class.

void undistort( Array2DReadView< uint8x3 > src,
    const CvCameraCalibrationParams& calib, Array2DWriteView< uint8x3 > dst )
{
    cv::Mat cvSrc = array2DViewAsCvMat( src );
    cv::Mat cvDst = array2DViewAsCvMat( dst );
    cv::remap( cvSrc, cvDst, calib.map0, calib.map1, cv::INTER_LANCZOS4 );
}

bool dumpFrames( RGBDInputStream& stream, uint32_t requestedStreamId )
{
    // Validate parameters.
    if( requestedStreamId >= stream.metadata().size() )
    {
        fprintf( stderr,
            "Requested stream %u exceeds number of streams %llu.\n",
            requestedStreamId, stream.metadata().size() );
        return false;
    }

    const auto& metadata = stream.metadata()[ requestedStreamId ];
    if( metadata.type == StreamType::UNKNOWN )
    {
        fprintf( stderr, "Cannot handle stream type: UNKNOWN.\n" );
        return false;
    }
    else if( metadata.type == StreamType::COLOR )
    {
        if( metadata.format != PixelFormat::RGB_U888 )
        {
            fprintf( stderr, "Cannot handle pixel format: \n",
                toString( metadata.format ) );
            return false;
        }
    }
    else if( metadata.type == StreamType::DEPTH ||
        metadata.type == StreamType::INFRARED )
    {
        fprintf( stderr, "We cannot (yet) handle depth.\n" );
        return false;
    }


    const int TIMESTAMP_FIELD_WIDTH = 20;

    uint32_t streamId;
    int frameIndex;
    int64_t timestamp;
    Array1DReadView< uint8_t > data;

    // E.g., "recording_00003.rgbd".
    std::string inputBasename = pystring::os::path::basename( FLAGS_input );

    std::string inputRoot; // E.g., recording_00003.
    std::string inputExt; // E.g., ".rgbd".
    pystring::os::path::splitext( inputRoot, inputExt, inputBasename );
    NumberedFilenameBuilder nfb(
        pystring::os::path::join( FLAGS_output, inputRoot ) + "_",
        "" /* suffix */ );

    bool ok = true;

    // Load camera calibration.
    CvCameraCalibrationParams calib;
    if( FLAGS_undistort )
    {
        fprintf( stderr, "Reading camera calibration from: \"%s\"...",
            FLAGS_calibration.c_str() );
        ok = LoadCvCameraCalibrationParams( FLAGS_calibration, calib );
        if( ok )
        {
            fprintf( stderr, "ok.\n" );
            if( calib.imageSize.width != metadata.size.x ||
                calib.imageSize.height != metadata.size.y )
            {
                ok = false;
                fprintf( stderr,
                    "Loaded camera calibration parameters, but calibrated "
                    "resolution %d x %d does not match stream resolution "
                    "%d x %d.",
                    calib.imageSize.width, calib.imageSize.height,
                    metadata.size.x, metadata.size.y );
            }
        }
        else
        {
            fprintf( stderr, "FAILED.\n" );
        }
    }

    // Allocate memory in case there needs to be conversions.
    Array2D< uint8_t > outputU8x1( metadata.size );
    Array2D< uint8x3 > outputU8x3( metadata.size );

    data = stream.read( streamId, frameIndex, timestamp );
    while( ok && data.notNull() )
    {
        if( streamId != requestedStreamId )
        {
            data = stream.read( streamId, frameIndex, timestamp );
            continue;
        }

        std::string outputFilename = nfb.filenameForNumber( frameIndex );
        outputFilename += "_" +
            toZeroFilledString( timestamp, TIMESTAMP_FIELD_WIDTH ) +
            ".png";

        // TODO: --output_format
        if( metadata.type == StreamType::COLOR )
        {
            if( metadata.format == PixelFormat::RGB_U888 )
            {
                // Reinterpret the source data from the stream.
                Array2DReadView< uint8x3 > src( data.pointer(),
                    metadata.size );

                // View of what to write.
                Array2DReadView< uint8x3 > outputView;
                if( FLAGS_undistort )
                {
                    // If we have to undistort, then the call undistort(), then
                    // point outputView to the remapped buffer.
                    undistort( src, calib, outputU8x3 );
                    outputView = outputU8x3;
                }
                else
                {
                    // Otherwise, point outputView to src.
                    outputView = src;
                }

                printf( "Writing color frame: "
                    "frameIndex: %d, timestamp: %lld to:\n"
                    "\"%s\" ...",
                    frameIndex, timestamp,
                    outputFilename.c_str() );
                ok = PNGIO::write( outputView, outputFilename );
                if( ok )
                {
                    printf( "done.\n" );
                }
                else
                {
                    printf( "FAILED.\n" );
                }
            }
        }
        data = stream.read( streamId, frameIndex, timestamp );
    }

    return ok;
}

void dumpTimestamps( RGBDInputStream& stream, uint32_t requestedStreamId )
{
    uint32_t streamId;
    int frameIndex;
    int64_t timestamp;
    Array1DReadView< uint8_t > src;

    FILE* f = stdout;
    if( FLAGS_output != "" )
    {
        f = fopen( FLAGS_output.c_str(), "w" );
    }

    src = stream.read( streamId, frameIndex, timestamp );
    while( src.notNull() )
    {
        if( streamId == requestedStreamId )
        {
            fprintf( f, "%u, %llu\n", frameIndex, timestamp );
        }

        src = stream.read( streamId, frameIndex, timestamp );
    }

    if( f != stdout && f != NULL )
    {
        fclose( f );
    }
}

int main( int argc, char* argv[] )
{
    gflags::ParseCommandLineFlags( &argc, &argv, true );

    if( FLAGS_input == "" )
    {
        fprintf( stderr, "Required flag --input must be a file" );
        return 1;
    }

    if( !( FLAGS_list_streams ||
        FLAGS_dump_frames ||
        FLAGS_dump_timestamps ) )
    {
        fprintf( stderr,
            "At least one of --list_streams, --dump_frames, or "
            "--dump_timestamps must be true.\n" );
        return 2;
    }

    // TODO: only one of these can be on at a time.

    RGBDInputStream inputStream( FLAGS_input );
    if( !inputStream.isValid() )
    {
        fprintf( stderr, "Error reading input file %s.\n",
            FLAGS_input.c_str() );
        return 3;
    }

    if( FLAGS_list_streams )
    {
        listStreams( inputStream );
        return 0;
    }

    if( FLAGS_dump_frames )
    {
        // TODO(C++17): check that directory exists.
        if( FLAGS_output == "" )
        {
            fprintf( stderr,
                "When --dump_frames is on, --output is required." );
            return 4;
        }

        bool ok = dumpFrames( inputStream, FLAGS_stream );
        if( ok )
        {
            return 0;
        }
        else
        {
            return 4;
        }
    }

    if( FLAGS_dump_timestamps )
    {
        dumpTimestamps( inputStream, FLAGS_stream );
        return 0;
    }

#if 0
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
            PNGIO::write( tonemappedDepth, outputFilename );
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
            PNGIO::write( tonemappedInfrared, outputFilename );
        }
        src =
            inputStream.read( streamId, frameIndex, timestamp );
    }
    return 0;
#endif
}
