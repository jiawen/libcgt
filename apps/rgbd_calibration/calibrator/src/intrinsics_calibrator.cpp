#include <iostream>
#include <vector>

#include <gflags/gflags.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <camera_wrappers/Kinect1x/KinectCamera.h>
#include <camera_wrappers/OpenNI2/OpenNI2Camera.h>
#include <core/io/NumberedFilenameBuilder.h>
#include <opencv_interop/VecmathUtils.h>
#include <third_party/pystring.h>

#include "common.h"

using libcgt::camera_wrappers::kinect1x::KinectCamera;
using libcgt::core::cameras::Intrinsics;
using libcgt::opencv_interop::toCV;
using libcgt::opencv_interop::fromCV3x3;

const cv::Size boardSize( 4, 11 ); // (cols, rows), TODO(jiawen): gflags

DEFINE_double( feature_spacing, 0.02, // 2 cm.
    "Spacing between two features on the calibration target in physical units "
    "(if you specify meters here, then downstream lengths will also be in "
    "these units).\n\n"
    "For the checkerboard, it is the square size (spacing between corners).\n"
    "For the symmetric circle grid, it is the spacing between circles.\n"
    "For the asymmetric circle grid, it is HALF the spacing between circles "
    "on any row or column." );

DEFINE_string( stream, "none", "color|infrared");
static bool validateStream( const char* flagname, const std::string& value )
{
    if( value != "color" && value != "infrared" )
    {
        fprintf( stderr, "stream must be \"color\" or \"infrared\"\n" );
        return false;
    }
    return true;
}
const bool stream_dummy = gflags::RegisterFlagValidator( &FLAGS_stream,
    &validateStream );

DEFINE_bool( visualize_detection, false,
    "Debug output: visualize each input image where a pattern was detected as "
    "<stream>_detect_<index>.png" );

DEFINE_bool( visualize_undistort, false,
    "Debug output: write out undistorted images as "
    "<stream>_undistort_<index>.png" );

int main( int argc, char* argv[] )
{
    gflags::ParseCommandLineFlags( &argc, &argv, true );

    if( argc < 2 )
    {
        fprintf( stderr, "output directory is required.\n" );
        return 1;
    }

    std::string dir( argv[ 1 ] );

    std::string calibrationType = FLAGS_stream + "_intrinsics";
    std::string filenameRoot = FLAGS_stream + "_";

    // TODO: read default intrinsics by camera and stream type to get initial guess.
    cv::Mat defaultIntrinsics;

    defaultIntrinsics = toCV(
        libcgt::camera_wrappers::openni2::OpenNI2Camera::defaultDepthIntrinsics().asMatrix() );
        // libcgt::camera_wrappers::openni2::OpenNI2Camera::defaultColorIntrinsics().asMatrix() );
        // KinectCamera::colorIntrinsics( Vector2i{ 640, 480 } ).asMatrix() );

    std::vector< cv::Mat > images = readImages( dir, filenameRoot );
    if( images.size() == 0 )
    {
        fprintf( stderr, "Could not read any images.\n" );
        return 1;
    }

    int nImages = static_cast< int >( images.size() );
    cv::Size imageSize( images[ 0 ].cols, images[ 0 ].rows );

    auto featuresPerImage = detectFeatures( images, boardSize );

    std::vector< std::vector< cv::Point2f > > usableFeaturesPerImage;
    for( const auto& v : featuresPerImage )
    {
        if( v.size() > 0 )
        {
            usableFeaturesPerImage.push_back( v );
        }
    }

    size_t nUsableImages = usableFeaturesPerImage.size();
    if( nUsableImages == 0 )
    {
        fprintf( stderr,
            "Did not detect features on input images. Exiting.\n" );
        return 2;
    }

    if( FLAGS_visualize_detection )
    {
        NumberedFilenameBuilder featureDetectNFB( pystring::os::path::join(
            dir, filenameRoot + "detect_" ), ".png" );

        for( int i = 0; i < nImages; ++i )
        {
            if( featuresPerImage[ i ].size() > 0 )
            {
                cv::Mat imWithDetections = images[ i ].clone();
                cv::drawChessboardCorners( imWithDetections, boardSize,
                    featuresPerImage[ i ], true );
                cv::imwrite( featureDetectNFB.filenameForNumber( i ), imWithDetections );
            }
        }
    }


    auto objPoints = generateObjectPoints(
        nUsableImages, boardSize,
        static_cast< float >( FLAGS_feature_spacing ) );

    std::cout << "Calibrating intrinsics...";
    const int flags = CV_CALIB_USE_INTRINSIC_GUESS;
    cv::Mat cameraMatrix = defaultIntrinsics;
    cv::Mat distCoeffs;
    std::vector< cv::Mat > rvecs;
    std::vector< cv::Mat > tvecs;
    std::vector< float > reprojErrs;
    double averageReprojectionError;

    averageReprojectionError =
        cv::calibrateCamera( objPoints, featuresPerImage, imageSize,
        cameraMatrix, distCoeffs, rvecs, tvecs, flags );
    std::cout << "done." << std::endl;

    std::cout << "calibration type: " << calibrationType << std::endl;
    std::cout << "frame count: " << featuresPerImage.size() << std::endl;
    std::cout << "average error (pixels): " << averageReprojectionError << std::endl;
    std::cout << "cameraMatrix: " << std::endl << cameraMatrix << std::endl;
    std::cout << "distCoeffs: " << std::endl << distCoeffs << std::endl;

    std::string outputCalibrationFilename =
        dir + "/" + filenameRoot + "calibration.yaml";
    cv::FileStorage fs( outputCalibrationFilename, cv::FileStorage::WRITE );

    fs << "calibrationType" << calibrationType;
    fs << "frameCount" << static_cast< int >( featuresPerImage.size() );
    fs << "averageErrorPixels" << averageReprojectionError;
    fs << "cameraMatrix" << cameraMatrix;
    fs << "distCoeffs" << distCoeffs;

    fs.release();

    // Intrinsics calibratedIntrinsics = Intrinsics::fromMatrix( fromCV3x3( cameraMatrix ) );

    if( FLAGS_visualize_undistort )
    {
        NumberedFilenameBuilder undistortNFB( pystring::os::path::join(
            dir, filenameRoot + "undistort_" ), ".png" );

        cv::Mat map0;
        cv::Mat map1;
        cv::Mat optimalNewCameraMatrix = cv::getOptimalNewCameraMatrix(
            cameraMatrix, distCoeffs, imageSize, 0, cv::Size() );
        cv::initUndistortRectifyMap( cameraMatrix, distCoeffs, cv::Mat(),
            optimalNewCameraMatrix, imageSize, CV_32FC1, map0, map1 );

        for( int i = 0; i < nImages; ++i )
        {
            cv::Mat undistorted;

            printf( "Undistorting %d of %d\n", i, nImages );
            cv::remap( images[ i ], undistorted, map0, map1, cv::INTER_LANCZOS4 );

            std::string outputFilename = undistortNFB.filenameForNumber( i );
            printf( "Writing %s\n", outputFilename.c_str() );
            cv::imwrite( outputFilename, undistorted );
        }
    }
}
