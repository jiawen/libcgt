#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <common/Array2DView.h>
#include <common/BasicTypes.h>

namespace libcgt { namespace opencv_interop { namespace arrayutils {

// Interpret "view" as a 3-channel const cv::Mat.
const cv::Mat array2DViewAsCvMat( Array2DView< const uint8x3 > view );

// Interpret "view" as a 3-channel cv::Mat.
cv::Mat array2DViewAsCvMat( Array2DView< uint8x3 > view );

// Interpret "a" as an Array2DView over the given type S.
// "a" is normally indexed as (row, col), but stored row major.
// The returned Array2DView will be indexed as (x, y) (aka, col, row),
// with the same storage.
template< typename S >
Array2DView< S > cvMatAsArray2DView( const cv::Mat& a );

// Interpret "a" as an Array2DView over the given type S.
// "a" is normally indexed as (row, col), but stored row major.
// The returned Array2DView will be indexed as (x, y) (aka, col, row),
// with the same storage.
template< typename S, typename T >
Array2DView< S > cvMatAsArray2DView( const cv::Mat_< T >& a );

} } } // arrayutils, opencv_interop, libcgt

#include "ArrayUtils.inl"
