#include <iostream>
#include <vector>

#include <gflags/gflags.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <camera_wrappers/Kinect1x/KinectCamera.h>
#include <camera_wrappers/OpenNI2/OpenNI2Camera.h>
#include <core/common/ArrayUtils.h>
#include <core/io/NumberedFilenameBuilder.h>
#include <core/io/PortableFloatMapIO.h>
#include <core/math/ArrayOps.h>
#include <opencv_interop/ArrayUtils.h>
#include <opencv_interop/VecmathUtils.h>
#include <third_party/pystring.h>

#include "common.h"

using namespace libcgt::camera_wrappers::kinect1x;
using namespace libcgt::camera_wrappers::openni2;
using namespace libcgt::core::arrayutils;
using namespace libcgt::core::cameras;
using namespace libcgt::opencv_interop;
using namespace pystring;

const cv::Size boardSize( 4, 11 ); // (cols, rows), TODO(jiawen): gflags

// TODO: make this a flag.
// In y-down.
const double depthFromInfraredPixelCoordinateShiftX = -4.5;
const double depthFromInfraredPixelCoordinateShiftY = -3.5;


DEFINE_double( feature_spacing, 0.02, // 2 cm.
    "Spacing between two features on the calibration target in physical units "
    "(if you specify meters here, then downstream lengths will also be in "
    "these units).\n\n"
    "For the checkerboard, it is the square size (spacing between corners).\n"
    "For the symmetric circle grid, it is the spacing between circles.\n"
    "For the asymmetric circle grid, it is HALF the spacing between circles "
    "on any row or column." );

DEFINE_string( dir, "",
    "Working directory. Actual images are read from <dir>/<stream>/"
    "<stream>_<#####>.png" );
static bool validateDir( const char* flagname, const std::string& value )
{
    if( value == "" )
    {
        fprintf( stderr, "dir must be a valid directory.\n" );
        return false;
    }
    return true;
}
const bool dir_dummy = gflags::RegisterFlagValidator( &FLAGS_dir,
    &validateDir );

DEFINE_string( stream, "none", "color|infrared" );
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

// TODO: check that directory exists, create it if it doesn't.
DEFINE_bool( visualize_detection, false,
    "Debug output: visualize each input image where a pattern was detected as "
    "<stream>_detect_<index>.png" );

DEFINE_bool( visualize_undistort, false,
    "Debug output: write out undistorted images as "
    "<stream>_undistort_<index>.png" );

Array2D< Vector2f > undistortMapsAsRG( const cv::Mat_< float >& map0,
    const cv::Mat_< float >& map1 )
{
    Array2DView< const float > map0View =
        cvMatAsArray2DView< const float >( map0 );
    Array2DView< const float > map1View =
        cvMatAsArray2DView< const float >( map1 );

    Array2D< Vector2f > undistortMap( map0View.size() );

    copy( map0View, componentView< float >(
        undistortMap.writeView(), 0 ) );
    copy( map1View, componentView< float >(
        undistortMap.writeView(), sizeof( float ) ) );

    return undistortMap;
}

Array2D< Vector3f > undistortMapsAsRGB( const cv::Mat_< float >& map0,
    const cv::Mat_< float >& map1 )
{
    Array2DView< const float > map0View =
        cvMatAsArray2DView< const float >( map0 );
    Array2DView< const float > map1View =
        cvMatAsArray2DView< const float >( map1 );

    Array2D< Vector3f > undistortMap( map0View.size() );

    copy( map0View, componentView< float >(
        undistortMap.writeView(), 0 ) );
    copy( map1View, componentView< float >(
        undistortMap.writeView(), sizeof( float ) ) );

    return undistortMap;
}

cv::Mat flipPrincipalPointY( const cv::Mat& cameraMatrix, cv::Size imageSize )
{
    cv::Mat output = cameraMatrix.clone();
    output.at< double >( 1, 2 ) =
        imageSize.height - cameraMatrix.at< double >( 1, 2 );
    return output;
}

int main( int argc, char* argv[] )
{
    gflags::ParseCommandLineFlags( &argc, &argv, true );

    const std::string calibrationType = FLAGS_stream + "_intrinsics";
    const std::string inputImageDir = os::path::join(
        FLAGS_dir, FLAGS_stream + "_images" );
    const std::string inputFilenameRoot = FLAGS_stream + "_";

    const std::string debugImageDir = os::path::join(
        FLAGS_dir, FLAGS_stream + "_debug" );

    // TODO: read default intrinsics by camera and stream type to get initial
    // guess.
    cv::Mat defaultIntrinsics;

    if( FLAGS_stream == "color" )
    {
        defaultIntrinsics = toCVMatrix3x3(
            OpenNI2Camera::defaultColorIntrinsics().asMatrix() );
    }
    else
    {
        defaultIntrinsics = toCVMatrix3x3(
            OpenNI2Camera::defaultDepthIntrinsics().asMatrix() );
    }

    std::vector< cv::Mat > images =
        readImages( inputImageDir, inputFilenameRoot );
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
            "Did not detect features on any input image. Exiting.\n" );
        return 2;
    }
    printf( "Found %zu images with features.\n", nUsableImages );

    if( FLAGS_visualize_detection )
    {
        NumberedFilenameBuilder featureDetectNFB( os::path::join(
            debugImageDir, inputFilenameRoot + "detect_" ), ".png" );

        for( int i = 0; i < nImages; ++i )
        {
            if( featuresPerImage[ i ].size() > 0 )
            {
                cv::Mat imWithDetections = images[ i ].clone();
                cv::drawChessboardCorners( imWithDetections, boardSize,
                    featuresPerImage[ i ], true );
                std::string outputFilename =
                    featureDetectNFB.filenameForNumber( i );
                printf( "Writing: %s\n", outputFilename.c_str() );
                cv::imwrite( outputFilename, imWithDetections );
            }
        }
    }

    auto objPoints = generateObjectPoints(
        nUsableImages, boardSize,
        static_cast< float >( FLAGS_feature_spacing ) );

    std::cout << "Calibrating intrinsics...";
    const int flags = CV_CALIB_USE_INTRINSIC_GUESS;
    cv::Mat cameraMatrix = defaultIntrinsics.clone();
    cv::Mat distCoeffs;
    std::vector< cv::Mat > rvecs;
    std::vector< cv::Mat > tvecs;
    std::vector< float > reprojErrs;
    double averageReprojectionError;

    averageReprojectionError =
        cv::calibrateCamera( objPoints, usableFeaturesPerImage, imageSize,
            cameraMatrix, distCoeffs, rvecs, tvecs, flags );
    std::cout << "done.\n";

    // NOTE: all this does is non-uniformly zoom the camera given the
    // estimated distortion coefficients. It changes the frustum planes,
    // not necessarily symmetrically, and then re-numbers the pixel
    // coordinates respectively (i.e., change the focal length and
    // principal point given scaling parameters).
    // alpha = 0: crop so that all pixels in the undistorted are valid.
    // alpha = 1: leave a black border but retain all source pixels.
    const float alpha = 0.0f;
    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(
        cameraMatrix, distCoeffs, imageSize, alpha );

    // Flip the principal point on the GL versions of the camera matrix.
    cv::Mat cameraMatrix_gl = flipPrincipalPointY( cameraMatrix, imageSize );
    cv::Mat newCameraMatrix_gl =
        flipPrincipalPointY( newCameraMatrix, imageSize );

    std::cout << "calibration type: " << calibrationType << "\n";
    std::cout << "resolution: " << imageSize.width << ", " <<
        imageSize.height << "\n";
    std::cout << "usable frame count: " << nUsableImages << "\n";
    std::cout << "average error (pixels): " <<
        averageReprojectionError << "\n";
    std::cout << "cameraMatrix_cv (y-down): " << "\n" << cameraMatrix << "\n";
    std::cout << "cameraMatrix_gl (y-up): " << "\n" << cameraMatrix_gl << "\n";
    std::cout << "distCoeffs: " << "\n" << distCoeffs << "\n";
    std::cout << "alpha: " << alpha << "\n";
    std::cout << "newCameraMatrix_cv (y-down): " << "\n" << newCameraMatrix <<
        "\n";
    std::cout << "newCameraMatrix_gl (y-up): " << "\n" << newCameraMatrix_gl <<
        "\n";

    std::string outputCalibrationFilename = os::path::join(
        FLAGS_dir, inputFilenameRoot + "calibration.yaml" );

    printf( "Writing to: %s\n", outputCalibrationFilename.c_str() );
    cv::FileStorage fs( outputCalibrationFilename, cv::FileStorage::WRITE );

    fs << "calibrationType" << calibrationType;
    fs << "imageSize" << imageSize;
    fs << "usableFrameCount" << static_cast< int >( nUsableImages );
    fs << "averageErrorPixels" << averageReprojectionError;
    fs << "cameraMatrix_cv" << cameraMatrix;
    fs << "cameraMatrix_gl" << cameraMatrix_gl;
    fs << "distCoeffs" << distCoeffs;
    fs << "alpha" << alpha;
    fs << "newCameraMatrix_cv" << newCameraMatrix;
    fs << "newCameraMatrix_gl" << newCameraMatrix_gl;

    fs.release();

    // TODO: collect everything into a IntrinsicCalibrationResults object.

    // NOTE: you obviously get a different undistort map depending on what
    // you pass in for newCameraMatrix. The documentation says that for a
    // monocular camera, you can pass in cameraMatrix itself.
    // Experimentally, this does *not* correspond to alpha = 0 or 1!.
    cv::Mat map0;
    cv::Mat map1;
    cv::initUndistortRectifyMap( cameraMatrix, distCoeffs, cv::Mat(),
        newCameraMatrix, imageSize, CV_32FC1, map0, map1 );

    // NOTE: even though the distortion coefficients are independent of
    // resolution and which way y points on the image plane (because they're a
    // function of angles), the undistortion maps are.
    Array2D< Vector2f > undistortMap = undistortMapsAsRG( map0, map1 );
    std::string undistortMapFilename = os::path::join(
        FLAGS_dir, inputFilenameRoot + "undistort_map_cv.pfm2" );
    std::string undistortMapGLFilename = os::path::join(
        FLAGS_dir, inputFilenameRoot + "undistort_map_gl.pfm2" );
    std::cout << "Writing undistortion map to: " <<
        undistortMapFilename << "\n";
    PortableFloatMapIO::write( undistortMapFilename, undistortMap );
    std::cout << "Writing undistortion map to: " <<
        undistortMapGLFilename << "\n";
    PortableFloatMapIO::write( undistortMapGLFilename,
        flipY( undistortMap.readView() ) );

    // In infrared mode, also output depth intrinsics and undistortion maps.
    if( FLAGS_stream == "infrared" )
    {
        std::string calibrationType( "depth_intrinsics" );
        std::string inputFilenameRoot( "depth_" );

        // Construct depth intrinsics from infrared intrinsics.
        cv::Mat depthCameraMatrix = cameraMatrix.clone();
        depthCameraMatrix.at< double >( 0, 2 ) +=
            depthFromInfraredPixelCoordinateShiftX;
        depthCameraMatrix.at< double >( 1, 2 ) +=
            depthFromInfraredPixelCoordinateShiftY;

        std::cout << "cameraMatrix = " << cameraMatrix << "\n";
        std::cout << "depthCameraMatrix = " << depthCameraMatrix << "\n";

        cv::Mat newDepthCameraMatrix = cv::getOptimalNewCameraMatrix(
            depthCameraMatrix, distCoeffs, imageSize, alpha );

        cv::Mat depthCameraMatrix_gl =
            flipPrincipalPointY( depthCameraMatrix, imageSize );
        cv::Mat newDepthCameraMatrix_gl =
            flipPrincipalPointY( newDepthCameraMatrix, imageSize );

        cv::Mat depthUndistortMap0;
        cv::Mat depthUndistortMap1;
        cv::initUndistortRectifyMap( depthCameraMatrix, distCoeffs, cv::Mat(),
            newDepthCameraMatrix, imageSize, CV_32FC1,
            depthUndistortMap0, depthUndistortMap1 );

        std::string outputCalibrationFilename = os::path::join(
            FLAGS_dir, inputFilenameRoot + "calibration.yaml" );

        printf( "Writing to: %s\n", outputCalibrationFilename.c_str() );
        cv::FileStorage fs( outputCalibrationFilename,
            cv::FileStorage::WRITE );

        fs << "calibrationType" << calibrationType;
        fs << "imageSize" << imageSize;
        fs << "usableFrameCount" << static_cast< int >( nUsableImages );
        fs << "averageErrorPixels" << averageReprojectionError;
        fs << "cameraMatrix_cv" << depthCameraMatrix;
        fs << "cameraMatrix_gl" << depthCameraMatrix_gl;
        fs << "distCoeffs" << distCoeffs;
        fs << "alpha" << alpha;
        fs << "newCameraMatrix_cv" << newDepthCameraMatrix;
        fs << "newCameraMatrix_gl" << newDepthCameraMatrix_gl;

        fs.release();

        Array2D< Vector2f > depthUndistortMap =
            undistortMapsAsRG( depthUndistortMap0, depthUndistortMap1 );
        std::string depthUndistortMapFilename = os::path::join(
            FLAGS_dir, inputFilenameRoot + "undistort_map_cv.pfm2" );
        std::string depthUndistortMapGLFilename = os::path::join(
            FLAGS_dir, inputFilenameRoot + "undistort_map_gl.pfm2" );
        std::cout << "Writing depth undistortion map to: " <<
            depthUndistortMapFilename << "\n";
        PortableFloatMapIO::write(
            depthUndistortMapFilename, depthUndistortMap );
        std::cout << "Writing depth undistortion map to: " <<
            depthUndistortMapGLFilename << "\n";
        PortableFloatMapIO::write( depthUndistortMapGLFilename,
            flipY( depthUndistortMap.readView() ) );
    }


    if( FLAGS_visualize_undistort )
    {
        NumberedFilenameBuilder undistortNFB( os::path::join(
            debugImageDir, inputFilenameRoot + "undistort_" ), ".png" );

        for( int i = 0; i < nImages; ++i )
        {
            cv::Mat undistorted;

            printf( "Undistorting %d of %d\n", i, nImages );
            cv::remap( images[ i ], undistorted, map0, map1,
                cv::INTER_LANCZOS4 );

            std::string outputFilename = undistortNFB.filenameForNumber( i );
            printf( "Writing: %s\n", outputFilename.c_str() );
            cv::imwrite( outputFilename, undistorted );
        }
    }
}
