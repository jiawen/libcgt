#pragma once

#include <cstdint>
#include <opencv2/core.hpp>

#include "libcgt/core/common/ArrayView.h"
#include "libcgt/core/common/BasicTypes.h"

namespace libcgt { namespace opencv_interop {

// Interpret "view" as a const cv::Mat.
// Returns an empty cv::Mat if view's elements are not packed.
const cv::Mat array2DViewAsCvMat( Array2DReadView< uint8_t > view );

// Interpret "view" as a cv::Mat.
// Returns an empty cv::Mat if view's elements are not packed.
cv::Mat array2DViewAsCvMat( Array2DWriteView< uint8_t > view );

// Interpret "view" as const cv::Mat.
// Returns an empty cv::Mat if view's elements are not packed.
const cv::Mat array2DViewAsCvMat( Array2DReadView< uint8x3 > view );

// Interpret "view" as cv::Mat.
// Returns an empty cv::Mat if view's elements are not packed.
cv::Mat array2DViewAsCvMat( Array2DWriteView< uint8x3 > view );

// Interpret "a" as an Array2DWriteView over the given type S.
// "a" is normally indexed as (row, col), but stored row major.
// The returned view will be indexed as (x, y) (aka, col, row),
// with the same storage.
template< typename S >
Array2DWriteView< S > cvMatAsArray2DView( const cv::Mat& a );

// Interpret "a" as an Array2DWriteView over the given type S.
// "a" is normally indexed as (row, col), but stored row major.
// The returned view will be indexed as (x, y) (aka, col, row),
// with the same storage.
template< typename S, typename T >
Array2DWriteView< S > cvMatAsArray2DView( const cv::Mat_< T >& a );

} } // opencv_interop, libcgt

#include "libcgt/opencv_interop/ArrayUtils.inl"
