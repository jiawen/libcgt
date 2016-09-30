#include "VecmathUtils.h"

#include <cameras/CameraUtils.h>
#include <vecmath/Matrix3f.h>
#include <vecmath/Vector2f.h>
#include <vecmath/Vector3f.h>

namespace libcgt { namespace opencv_interop {

cv::Point2f toCVPoint( const Vector2f& v )
{
    return cv::Point2f( v.x, v.y );
}

cv::Point3f toCVPoint( const Vector3f& v )
{
    return cv::Point3f( v.x, v.y, v.z );
}

cv::Mat_< float > toCVMatrix3x1( const Vector3f& v )
{
    cv::Mat_< float > output( 3, 1 );
    output.at< float >( 0, 0 ) = v.x;
    output.at< float >( 1, 0 ) = v.y;
    output.at< float >( 2, 0 ) = v.z;
    return output;
}

cv::Mat_< float > toCVMatrix3x3( const Matrix3f& a )
{
    cv::Mat_< float > output( 3, 3 );
    for( int i = 0; i < 3; ++i )
    {
        for( int j = 0; j < 3; ++j )
        {
            output( i, j ) = a( i, j );
        }
    }
    return output;
}

cv::Mat_< float > toCVMatrix4x4( const Matrix4f& a )
{
    cv::Mat_< float > output( 4, 4 );
    for( int i = 0; i < 4; ++i )
    {
        for( int j = 0; j < 4; ++j )
        {
            output( i, j ) = a( i, j );
        }
    }
    return output;
}

// Grab 3 elements from row i, starting at column j0.
// a( i, j0 : j0 + 3 )
Vector3f fromCV1x3( const cv::Mat_< float >& a, int i, int j0 )
{
    Vector3f output;
    for( int j = 0; j < 3; ++j )
    {
        output[ j ] = a( i, j0 + j );
    }
    return output;
}

// Grab 3 elements from column j, starting at row i0.
// a( i0 : i0 + 3, j )
Vector3f fromCV3x1( const cv::Mat_< float >& a, int i0, int j )
{
    Vector3f output;
    for( int i = 0; i < 3; ++i )
    {
        output[ i ] = a( i0 + i, j );
    }
    return output;
}

Matrix3f fromCV3x3( const cv::Mat_< float >& a, int i0, int j0 )
{
    Matrix3f output;
    for( int i = 0; i < 3; ++i )
    {
        for( int j = 0; j < 3; ++j )
        {
            output( i, j ) = a( i0 + i, j0 + j );
        }
    }
    return output;
}

Matrix4f fromCV4x4( const cv::Mat_< float >& a, int i0, int j0 )
{
    Matrix4f output;
    for( int i = 0; i < 4; ++i )
    {
        for( int j = 0; j < 4; ++j )
        {
            output( i, j ) = a( i0 + i, j0 + j );
        }
    }
    return output;
}

} } // opencv_interop, libcgt
