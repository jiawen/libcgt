#include <iostream>
#include <vector>

#include <gflags/gflags.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <core/vecmath/EuclideanTransform.h>
#include <core/vecmath/Matrix3f.h>
#include <core/io/NumberedFilenameBuilder.h>
#include <opencv_interop/VecmathUtils.h>
#include <third_party/pystring.h>

#include "common.h"

const cv::Size boardSize( 4, 11 ); // (cols, rows), TODO: make this a flag.

DEFINE_string( dir, "",
    "Working directory.\n"
    "Expects intrinsic calibration to be under"
    " <dir>/color_calibration.yaml and <dir>/infrared_calibration.yaml"
    "Expects infrared_<#####>.png and color_<#####>.png to be under"
    " <dir>/infrared_color_stereo_images\n" );
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

DEFINE_double( feature_spacing, 0.02, // 2 cm.
    "Spacing between two features on the calibration target in physical units "
    "(if you specify meters here, then downstream lengths will also be in "
    "these units).\n\n"
    "For the checkerboard, it is the square size (spacing between corners).\n"
    "For the symmetric circle grid, it is the spacing between circles.\n"
    "For the asymmetric circle grid, it is HALF the spacing between circles "
    "on any row or column." );

struct IntrinsicCalibrationData
{
    cv::Mat intrinsicsMatrix;
    cv::Mat distCoeffs;
};

IntrinsicCalibrationData readIntrinsicsFromFile( const std::string& filename )
{
    cv::FileStorage fs( filename, cv::FileStorage::READ );
    IntrinsicCalibrationData output;
    fs[ "cameraMatrix" ] >> output.intrinsicsMatrix;
    fs[ "distCoeffs" ] >> output.distCoeffs;
    return output;
}

int main( int argc, char* argv[] )
{
    gflags::ParseCommandLineFlags( &argc, &argv, true );

    std::string stereoImageDir = pystring::os::path::join(
        FLAGS_dir, "infrared_color_stereo_images" );

    // IR is on the left.
    std::vector< cv::Mat > infraredImages =
        readImages( stereoImageDir, "infrared_" );
    std::vector< cv::Mat > colorImages =
        readImages( stereoImageDir, "color_" );

    printf( "Loaded %zu infrared images.\n", infraredImages.size() );
    printf( "Loaded %zu color images.\n", colorImages.size() );
    if( infraredImages.size() != colorImages.size() )
    {
        printf( "Number of images not equal, aborting.\n" );
        return 1;
    }
    int nImages = static_cast< int >( infraredImages.size() );
    cv::Size imageSize( infraredImages[ 0 ].cols, infraredImages[ 0 ].rows );
    std::cout << "infrared image size: " << imageSize << std::endl;

    auto infraredIntrinsics = readIntrinsicsFromFile( pystring::os::path::join(
        FLAGS_dir, "infrared_calibration.yaml" ) );
    auto colorIntrinsics = readIntrinsicsFromFile( pystring::os::path::join(
        FLAGS_dir, "color_calibration.yaml" ) );

    std::cout << std::endl << std::endl;

    std::cout << "infrared intrinsics camera matrix:\n" <<
        infraredIntrinsics.intrinsicsMatrix << "\n";
    std::cout << "infrared distortion coefficients:\n" <<
        infraredIntrinsics.distCoeffs << "\n";
    std::cout << "color intrinsics camera matrix: " << std::endl <<
        colorIntrinsics.intrinsicsMatrix << std::endl;
    std::cout << "color distortion coefficients: " << std::endl <<
        colorIntrinsics.distCoeffs << std::endl;

    std::cout << std::endl << std::endl;

    auto stereoFeatures =
        detectStereoFeatures( infraredImages, colorImages, boardSize );

    auto objPoints =
        generateObjectPoints( stereoFeatures.left.size(),
        boardSize, static_cast< float >( FLAGS_feature_spacing ) );

    // These are the defaults.
    cv::TermCriteria criteria( cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
        30, 1e-6 );
    int flags = cv::CALIB_FIX_INTRINSIC; // Shouldn't need anything else.

    cv::Mat colorFromInfrared_R;
    cv::Mat colorFromInfrared_T;
    cv::Mat E;
    cv::Mat F;
    double reprojectionError;

    printf( "Stereo calibrating %d image pairs...", nImages );
    reprojectionError = cv::stereoCalibrate(
        objPoints,
        stereoFeatures.left, stereoFeatures.right,
        infraredIntrinsics.intrinsicsMatrix, infraredIntrinsics.distCoeffs,
        colorIntrinsics.intrinsicsMatrix, colorIntrinsics.distCoeffs,
        imageSize, // Only used to initialize intrinsics, shouldn't matter.
        colorFromInfrared_R, colorFromInfrared_T,
        E, F,
        flags,
        criteria
    );
    printf( "done.\n" );

    std::cout << "\n\n";
    std::cout << "reprojection error:\n" << reprojectionError << "\n";
    std::cout << "colorFromInfrared_R:\n" << colorFromInfrared_R << "\n\n";
    std::cout << "colorFromInfrared_T:\n" << colorFromInfrared_T << "\n\n";
    std::cout << "colorFromInfrared_T (mm):\n" << 1000 * colorFromInfrared_T <<
        "\n\n";
    std::cout << "E:\n" << E << "\n\n";
    std::cout << "F:\n" << F << "\n\n";

    Matrix3f r = libcgt::opencv_interop::fromCV3x3( colorFromInfrared_R );
    Vector3f t = libcgt::opencv_interop::fromCV3x1( colorFromInfrared_T );
    libcgt::core::vecmath::EuclideanTransform e( r, t );
    libcgt::core::vecmath::EuclideanTransform eInv =
        libcgt::core::vecmath::inverse( e );

    std::cout << "eInv.r = " << std::endl << eInv.rotation.toString() << std::endl;
    std::cout << "eInv.t = " << std::endl << eInv.translation.toString() << std::endl;

    //cv::Mat infraredFromColor_R = libcgt::opencv_interop::toCV( eInv.rotation );
    //cv::Mat infraredFromColor_R = libcgt::opencv_interop::toCV( eInv.rotation );
}
