#include "Calib3d.h"

#include <opencv2/imgproc.hpp>

#include <common/ArrayUtils.h>
#include <common/ForND.h>
#include "ArrayUtils.h"

using namespace libcgt::core;
using namespace libcgt::core::arrayutils;
using namespace libcgt::core::cameras;
using libcgt::opencv_interop::cvMatAsArray2DView;

namespace
{

Array2D< Vector2f > undistortMapsAsRG( const cv::Mat_< float >& map0,
    const cv::Mat_< float >& map1,
    bool shiftHalfPixel, bool flipY, bool normalizeCoordinates )
{
    Array2DView< const float > map0View =
        cvMatAsArray2DView< const float >( map0 );
    Array2DView< const float > map1View =
        cvMatAsArray2DView< const float >( map1 );

    Array2D< Vector2f > undistortMap( map0View.size() );

    Vector2f normalization{ 1.0f };
    if( normalizeCoordinates )
    {
        normalization = 1.0f / Vector2f( map0View.size() );
    }

    if( flipY )
    {
        Array2DView < Vector2f > dst = ::flipY( undistortMap.writeView() );
        if( shiftHalfPixel )
        {
            for2D( dst.size(), [&] ( const Vector2i& xy )
            {
                dst[ xy ] = normalization * Vector2f
                {
                    map0View[ xy ] + 0.5f,
                    dst.height() - ( map1View[ xy ] + 0.5f )
                };
            } );
        }
        else
        {
            for2D( dst.size(), [&] ( const Vector2i& xy )
            {
                dst[ xy ] = normalization * Vector2f
                {
                    map0View[ xy ],
                    dst.height() - map1View[ xy ]
                };
            } );
        }
    }
    else
    {
        if( shiftHalfPixel )
        {
            for2D( undistortMap.size(), [&] ( const Vector2i& xy )
            {
                undistortMap[ xy ] = normalization * Vector2f
                {
                    map0View[ xy ] + 0.5f,
                    map1View[ xy ] + 0.5f
                };
            } );
        }
        else
        {
            auto dstX = componentView< float >( undistortMap.writeView(), 0 );
            auto dstY = componentView< float >( undistortMap.writeView(),
                sizeof( float ) );
            copy( map0View, dstX );
            copy( map1View, dstY );

            if( normalizeCoordinates )
            {
                for2D( undistortMap.size(), [&] ( const Vector2i& xy )
                {
                    undistortMap[ xy ] = normalization * undistortMap[ xy ];
                } );
            }
        }
    }

    return undistortMap;
}

}

namespace libcgt { namespace opencv_interop {

cv::Mat_< double > makeCameraMatrix( double focalLength,
    double principalPointX, double principalPointY, double skew )
{
    cv::Mat_< double > cameraMatrix = cv::Mat_<double>::zeros( 3, 3 );
    cameraMatrix( 0, 0 ) = focalLength;
    cameraMatrix( 0, 1 ) = skew;
    cameraMatrix( 0, 2 ) = principalPointX;
    cameraMatrix( 1, 1 ) = focalLength;
    cameraMatrix( 1, 2 ) = principalPointY;
    cameraMatrix( 2, 2 ) = 1;
    return cameraMatrix;
}

cv::Mat_< double > cameraMatrixCVToGL( const cv::Mat_< double >& cameraMatrix,
    cv::Size imageSize, bool shiftHalfPixel )
{
    double shift = shiftHalfPixel ? 0.5f : 0.0f;
    cv::Mat output = cameraMatrix.clone();
    output.at< double >( 0, 2 ) =
        cameraMatrix.at< double >( 0, 2 ) + shiftHalfPixel;
    output.at< double >( 1, 2 ) = imageSize.height -
        ( cameraMatrix.at< double >( 1, 2 ) + shiftHalfPixel );
    return output;
}

Intrinsics cameraMatrixToIntrinsics( const cv::Mat_< double >& cameraMatrix )
{
    Intrinsics intrinsics
    {
        {
            static_cast< float >( cameraMatrix( 0, 0 ) ),
            static_cast< float >( cameraMatrix( 1, 1 ) )
        },
        {
            static_cast< float >( cameraMatrix( 0, 2 ) ),
            static_cast< float >( cameraMatrix( 1, 2 ) )
        }
    };

    return intrinsics;
}

Array2D< Vector2f > undistortRectifyMap( cv::InputArray cameraMatrix,
    cv::InputArray distCoeffs,
    cv::InputArray R, cv::InputArray newCameraMatrix,
    cv::Size imageSize,
    bool shiftHalfPixel, bool flipY, bool normalizeCoordinates )
{
    cv::Mat map0;
    cv::Mat map1;
    cv::initUndistortRectifyMap( cameraMatrix, distCoeffs, R,
        newCameraMatrix, imageSize, CV_32FC1, map0, map1 );
    return undistortMapsAsRG( map0, map1,
        shiftHalfPixel, flipY, normalizeCoordinates );
}

} } // opencv_interop, libcgt
