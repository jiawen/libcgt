#pragma once

#include <opencv2/calib3d.hpp>

#include <common/Array2D.h>
#include <cameras/Intrinsics.h>
#include <vecmath/Vector2f.h>
#include <imageproc/Sampling.h>

namespace libcgt { namespace opencv_interop {

// Construct a 3x3 OpenCV-style camera matrix given individual intrinsic
// parameters. The y-axis points down, pixel centers have integer coordinates.
cv::Mat_< double > makeCameraMatrix( double focalLength,
    double principalPointX, double principalPointY,
    double skew = 0 );

// Given an OpenCV-style camera matrix (intrinsics), flip the y axis such that
// y points up.
//
// shiftHalfPixel: OpenCV have integer coordinates at pixel centers. Set to
// true to shift them to use OpenGL conventions, with integer coordinates at
// pixel corners.
cv::Mat_< double > cameraMatrixCVToGL( const cv::Mat_< double >& cameraMatrix,
    cv::Size imageSize, bool shiftHalfPixel = true );

// Directly convert an OpenCV-style camera matrix (intrinsics) into a libcgt
// intrinsics. Does not flip the y axis or shift the coordinates. For that,
// first call cameraMatrixCVToGL().
libcgt::core::cameras::Intrinsics cameraMatrixToIntrinsics(
    const cv::Mat_< double >& cameraMatrix );

// Given the calibration data from OpenCV, construct the undistort map as an
// Array2D< Vector2f >, in either OpenCV or OpenGL convention.
//
// NOTE: you get a different undistort map depending on what you pass in for
// newCameraMatrix. The documentation says that for a monocular camera, you can
// pass in the original cameraMatrix itself. Experimentally, it seems that this
// does *not* correspond to a setting of alpha = 0 or 1
// for newCameraMatrix = getOptimalNewCameraMatrix( alpha ).
//
// NOTE: even though the distortion coefficients are independent of
// resolution and which way y points on the image plane (because they're a
// function of angles), the undistortion maps are.
//
// shiftHalfPixel: OpenCV have integer coordinates at pixel centers. Set to
// true to shift them to use OpenGL conventions, with integer coordinates at
// pixel corners.
//
// flipY: OpenCV has the y axis pointing down. To use directly as an OpenGL
// texture, it needs to be flipped in the Y direction.
//
// normalizeCoordinates: OpenGL and DirectX use [0,1]^2 for texture
// coordinates. CUDA can use either. If true, the output undistortion map will
// be in the unit square. Otherwise, it will be in [0, imageSize].
Array2D< Vector2f > undistortRectifyMap( cv::InputArray cameraMatrix,
    cv::InputArray distCoeffs,
    cv::InputArray R, cv::InputArray newCameraMatrix,
    cv::Size imageSize, bool shiftHalfPixel = true, bool flipY = true,
    bool normalizeCoordinates = true );

template< typename T >
bool remap( Array2DReadView< T > srcDistorted, Array2DReadView< Vector2f > map,
    Array2DWriteView< T > dstUndistorted )
{
    if( srcDistorted.size() != map.size() ||
        srcDistorted.size() != dstUndistorted.size() )
    {
        return false;
    }

    for( int y = 0; y < dstUndistorted.height(); ++y )
    {
        for( int x = 0; x < dstUndistorted.width(); ++x )
        {
            Vector2i xy{ x, y };
            Vector2f uv = map[ xy ];
            dstUndistorted[ xy ] = libcgt::core::imageproc::bilinearSample(
                srcDistorted, uv );
        }
    }

    return true;
}

} } // opencv_interop, libcgt
