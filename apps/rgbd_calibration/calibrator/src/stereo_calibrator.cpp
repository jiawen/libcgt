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

using namespace pystring;
using namespace libcgt::core::vecmath;
using namespace libcgt::opencv_interop;

const cv::Size boardSize( 4, 11 ); // (cols, rows), TODO: make this a flag.

// TODO: make this a flag.
// In y-down.
const double depthFromInfraredPixelCoordinateShiftX = -4.5;
const double depthFromInfraredPixelCoordinateShiftY = -3.5;

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
    cv::Mat intrinsicsMatrix_cv;
    cv::Mat intrinsicsMatrix_gl;
    cv::Mat distCoeffs;
    double alpha;
    cv::Mat newIntrinsicsMatrix_cv;
    cv::Mat newIntrinsicsMatrix_gl;
};

IntrinsicCalibrationData readIntrinsicsFromFile( const std::string& filename )
{
    // TODO: abort when it fails to read something.
    cv::FileStorage fs( filename, cv::FileStorage::READ );
    IntrinsicCalibrationData output;
    fs[ "cameraMatrix_cv" ] >> output.intrinsicsMatrix_cv;
    fs[ "cameraMatrix_gl" ] >> output.intrinsicsMatrix_gl;
    fs[ "distCoeffs" ] >> output.distCoeffs;
    fs[ "alpha" ] >> output.alpha;
    fs[ "newCameraMatrix_cv" ] >> output.newIntrinsicsMatrix_cv;
    fs[ "newCameraMatrix_gl" ] >> output.newIntrinsicsMatrix_gl;
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
    printf( "Loaded %d image pairs...", nImages );
    cv::Size infraredImageSize( infraredImages[ 0 ].cols,
        infraredImages[ 0 ].rows );
    cv::Size depthImageSize = infraredImageSize;
    cv::Size colorImageSize( colorImages[ 0 ].cols, colorImages[ 0 ].rows );
    std::cout << "infrared image size: " << infraredImageSize << "\n";
    std::cout << "depth image size: " << depthImageSize << "\n";
    std::cout << "color image size: " << colorImageSize << "\n";
    std::cout << "\n\n";

    auto infraredIntrinsics = readIntrinsicsFromFile( os::path::join(
        FLAGS_dir, "infrared_calibration.yaml" ) );
    auto colorIntrinsics = readIntrinsicsFromFile( os::path::join(
        FLAGS_dir, "color_calibration.yaml" ) );
    auto depthIntrinsics = readIntrinsicsFromFile( os::path::join(
        FLAGS_dir, "depth_calibration.yaml" ) );

    // TODO: print the whole struct by overloading ostream operator <<;
    std::cout << "infrared intrinsics camera matrix (cv):\n" <<
        infraredIntrinsics.intrinsicsMatrix_cv << "\n";
    std::cout << "infrared distortion coefficients:\n" <<
        infraredIntrinsics.distCoeffs << "\n";
    std::cout << "color intrinsics camera matrix (cv):\n" <<
        colorIntrinsics.intrinsicsMatrix_cv << "\n";
    std::cout << "color distortion coefficients:\n" <<
        colorIntrinsics.distCoeffs << "\n";

    std::cout << "\n\n";

    StereoFeatures stereoFeatures =
        detectStereoFeatures( infraredImages, colorImages, boardSize );
    int nUsableImages = static_cast< int >( stereoFeatures.left.size() );
    printf( "Found %d image pairs with usable features.\n" );

    auto objPoints =
        generateObjectPoints( stereoFeatures.left.size(),
        boardSize, static_cast< float >( FLAGS_feature_spacing ) );

    // These are the defaults.
    cv::TermCriteria criteria( cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
        30, 1e-6 );
    int flags = cv::CALIB_FIX_INTRINSIC; // Shouldn't need anything else.

    // Note: this ordering is correct.
    // Physically, the infrared camera is on the left, the color is on the
    // right. stereoCalibrate returns [R|t] as the matrix right_from_left.
    cv::Mat colorFromInfrared_cv_R;
    cv::Mat colorFromInfrared_cv_T;
    cv::Mat E;
    cv::Mat F;
    double averageReprojectionError;

    averageReprojectionError = cv::stereoCalibrate(
        objPoints,
        stereoFeatures.left, stereoFeatures.right,
        infraredIntrinsics.intrinsicsMatrix_cv, infraredIntrinsics.distCoeffs,
        colorIntrinsics.intrinsicsMatrix_cv, colorIntrinsics.distCoeffs,
        infraredImageSize, // Only used to initialize intrinsics, so ignored.
        colorFromInfrared_cv_R, colorFromInfrared_cv_T,
        E, F,
        flags,
        criteria
    );
    printf( "done.\n\n" );

    // Convert to libcgt to get an efficient inverse because OpenCV's
    // invertAffineTransform() only works on 2x3 matrices.
    EuclideanTransform colorFromInfrared_cv(
        fromCV3x3( colorFromInfrared_cv_R ),
        fromCV3x1( colorFromInfrared_cv_T )
    );
    EuclideanTransform infraredFromColor_cv = inverse( colorFromInfrared_cv );

    // Rotate into OpenGL coordinates.
    EuclideanTransform colorFromInfrared_gl = glFromCV( colorFromInfrared_cv );
    EuclideanTransform infraredFromColor_gl = glFromCV( infraredFromColor_cv );

    // Convert back to OpenCV to write out.
    cv::Mat colorFromInfrared_gl_R =
        toCVMatrix3x3( colorFromInfrared_gl.rotation );
    cv::Mat colorFromInfrared_gl_T =
        toCVMatrix3x1( colorFromInfrared_gl.translation );
    cv::Mat infraredFromColor_cv_R =
        toCVMatrix3x3( infraredFromColor_cv.rotation );
    cv::Mat infraredFromColor_cv_T =
        toCVMatrix3x1( infraredFromColor_cv.translation );
    cv::Mat infraredFromColor_gl_R =
        toCVMatrix3x3( infraredFromColor_gl.rotation );
    cv::Mat infraredFromColor_gl_T =
        toCVMatrix3x1( infraredFromColor_gl.translation );

    std::cout << "reprojection error:\n" << averageReprojectionError << "\n\n";
    std::cout << "essentialMatrix:\n" << E << "\n\n";
    std::cout << "fundamentalMatrix:\n" << F << "\n\n";
    std::cout << "colorFromInfrared_cv_R:\n" << colorFromInfrared_cv_R << "\n\n";
    std::cout << "colorFromInfrared_cv_T:\n" << colorFromInfrared_cv_T << "\n\n";
    std::cout << "infraredFromColor_cv_R:\n" << infraredFromColor_cv_R << "\n\n";
    std::cout << "infraredFromColor_cv_T:\n" << infraredFromColor_cv_T << "\n\n";
    std::cout << "colorFromInfrared_gl_R:\n" << colorFromInfrared_gl_R << "\n\n";
    std::cout << "colorFromInfrared_gl_T:\n" << colorFromInfrared_gl_T << "\n\n";
    std::cout << "infraredFromColor_gl_R:\n" << infraredFromColor_gl_R << "\n\n";
    std::cout << "infraredFromColor_gl_T:\n" << infraredFromColor_gl_T << "\n\n";

    // TODO: error check output.
    std::string outputFilename = os::path::join(
        FLAGS_dir, "stereo_calibration.yaml" );
    std::cout << "Writing to: " << outputFilename << "\n";
    cv::FileStorage fs( outputFilename, cv::FileStorage::WRITE );

    const std::string calibrationType( "rgbd_intrinsics+extrinsics" );
    fs << "calibrationType" << calibrationType;
    fs << "infraredImageSize" << infraredImageSize;
    fs << "colorImageSize" << colorImageSize;
    fs << "depthImageSize" << depthImageSize;
    fs << "usableFrameCount" << static_cast< int >( nUsableImages );
    fs << "averageErrorPixels" << averageReprojectionError;

    // Intrinsics.
    fs << "colorCameraMatrix_cv" << colorIntrinsics.intrinsicsMatrix_cv;
    fs << "colorCameraMatrix_gl" << colorIntrinsics.intrinsicsMatrix_gl;
    fs << "colorDistCoeffs" << colorIntrinsics.distCoeffs;
    fs << "colorAlpha" << colorIntrinsics.alpha;
    fs << "colorNewCameraMatrix_cv" << colorIntrinsics.newIntrinsicsMatrix_cv;
    fs << "colorNewCameraMatrix_gl" << colorIntrinsics.newIntrinsicsMatrix_gl;
    fs << "infraredCameraMatrix_cv" << infraredIntrinsics.intrinsicsMatrix_cv;
    fs << "infraredCameraMatrix_gl" << infraredIntrinsics.intrinsicsMatrix_gl;
    fs << "infraredDistCoeffs" << infraredIntrinsics.distCoeffs;
    fs << "infraredAlpha" << infraredIntrinsics.alpha;
    fs << "infraredNewCameraMatrix_cv" <<
        infraredIntrinsics.newIntrinsicsMatrix_cv;
    fs << "infraredNewCameraMatrix_gl" <<
        infraredIntrinsics.newIntrinsicsMatrix_gl;
    fs << "depthCameraMatrix_cv" << depthIntrinsics.intrinsicsMatrix_cv;
    fs << "depthCameraMatrix_gl" << depthIntrinsics.intrinsicsMatrix_gl;
    fs << "depthDistCoeffs" << depthIntrinsics.distCoeffs;
    fs << "depthAlpha" << depthIntrinsics.alpha;
    fs << "depthNewCameraMatrix_cv" << depthIntrinsics.newIntrinsicsMatrix_cv;
    fs << "depthNewCameraMatrix_gl" << depthIntrinsics.newIntrinsicsMatrix_gl;
    fs << "depthFromInfraredPixelCoordinateShiftX" <<
        depthFromInfraredPixelCoordinateShiftX;
    fs << "depthFromInfraredPixelCoordinateShiftY" <<
        depthFromInfraredPixelCoordinateShiftY;

    // Extrinsics.
    fs << "colorFromInfrared_cv_R" << colorFromInfrared_cv_R;
    fs << "colorFromInfrared_cv_T" << colorFromInfrared_cv_T;
    fs << "infraredFromColor_cv_R" << infraredFromColor_cv_R;
    fs << "infraredFromColor_cv_T" << infraredFromColor_cv_T;
    fs << "colorFromInfrared_gl_R" << colorFromInfrared_gl_R;
    fs << "colorFromInfrared_gl_T" << colorFromInfrared_gl_T;
    fs << "infraredFromColor_gl_R" << infraredFromColor_gl_R;
    fs << "infraredFromColor_gl_T" << infraredFromColor_gl_T;

    // color <--> depth extrinsics are the same as color <--> infrared.
    fs << "colorFromDepth_cv_R" << colorFromInfrared_cv_R;
    fs << "colorFromDepth_cv_T" << colorFromInfrared_cv_T;
    fs << "depthFromColor_cv_R" << infraredFromColor_cv_R;
    fs << "depthFromColor_cv_T" << infraredFromColor_cv_T;
    fs << "colorFromDepth_gl_R" << colorFromInfrared_gl_R;
    fs << "colorFromDepth_gl_T" << colorFromInfrared_gl_T;
    fs << "depthFromColor_gl_R" << infraredFromColor_gl_R;
    fs << "depthFromColor_gl_T" << infraredFromColor_gl_T;
}
