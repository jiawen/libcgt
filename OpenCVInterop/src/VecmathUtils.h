#pragma once

#include <opencv2/core.hpp>

class Matrix3f;
class Matrix4f;
class Vector2f;
class Vector3f;

namespace libcgt { namespace opencv_interop { namespace vecmathutils {

cv::Point2f toCV( const Vector2f& v );

cv::Point3f toCV( const Vector3f& v );

cv::Mat_< float > toCV( const Matrix3f& a );

cv::Mat_< float > toCV( const Matrix4f& a );

// Grab 3 elements from row i, starting at column j0.
// a( i, j0 : j0 + 3 )
Vector3f fromCV1x3( const cv::Mat_< float >& a, int i = 0, int j0 = 0 );

// Grab 3 elements from column j, starting at row i0.
// a( i0 : i0 + 3, j )
Vector3f fromCV3x1( const cv::Mat_< float >& a, int i0 = 0, int j = 0 );

Matrix3f fromCV3x3( const cv::Mat_< float >& a, int i0 = 0, int j0 = 0 );

Matrix4f fromCV4x4( const cv::Mat_< float >& a, int i0 = 0, int j0 = 0 );

} } } // vecmath, opencv_interop, libcgt
