#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <common/Array2DView.h>

class Matrix3f;
class Vector2f;
class Vector3f;

namespace libcgt
{
namespace opencv_interop
{
namespace vecmath
{
    cv::Point2f toOpenCVPoint2f( const Vector2f& v );

    cv::Point3f toOpenCVPoint3f( const Vector3f& v );

    cv::Mat_< float > toOpenCV3x3( const Matrix3f& a );

    Matrix3f fromOpenCV3x3( const cv::Mat_< float >& a, int i0 = 0, int j0 = 0 );

    // Grab 3 elements from row i, starting at column j0.
    // a( i, j0 : j0 + 3 )
    Vector3f fromOpenCV1x3( const cv::Mat_< float >& a, int i = 0, int j0 = 0 );

    // Grab 3 elements from column j, starting at row i0.
    // a( i0 : i0 + 3, j )
    Vector3f fromOpenCV3x1( const cv::Mat_< float >& a, int i0 = 0, int j = 0 );

    // TODO(jiawen): Move to another file.
    // Interpret a an Array2DView over the given type S.
    // a is normally indexed as (row, col), but stored row major.
    // The returned Array2DView will be indexed as (x, y) (aka, col, row),
    // with the same storage.
    template< typename S >
    Array2DView< S > cvMatAsArray2DView( const cv::Mat& a )
    {
        // stride is (step[1], step[0]) because cv::Mat is indexed
        // as (row, col) but stored row major.
        Vector2i size{ a.cols, a.rows };
        Vector2i stride
        {
            static_cast< int >( a.step[1] ),
            static_cast< int >( a.step[0] )
        };
        return Array2DView< S >( a.data, size, stride );
    }

    // TODO(jiawen): Move to another file.
    // Interpret a an Array2DView over the given type S.
    // a is normally indexed as (row, col), but stored row major.
    // The returned Array2DView will be indexed as (x, y) (aka, col, row),
    // with the same storage.
    template< typename S, typename T >
    Array2DView< S > cvMatAsArray2DView( const cv::Mat_< T >& a )
    {
        // stride is (step[1], step[0]) because cv::Mat is indexed
        // as (row, col) but stored row major.
        Vector2i size{ a.cols, a.rows };
        Vector2i stride
        {
            static_cast< int >( a.step[1] ),
            static_cast< int >( a.step[0] )
        };
        return Array2DView< S >( a.data, size, stride );
    }
} // vecmath
} // opencv_interop
} // libcgt
