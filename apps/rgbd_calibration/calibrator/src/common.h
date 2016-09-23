#pragma once

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

enum class PatternType
{
    CHESSBOARD,
    CIRCLES_GRID,
    ASYMMETRIC_CIRCLES_GRID
};

// patternSize: number of corners or circles.
// squareSize: chessboard: the side length (in world units, e.g., meters) of
//   each square. CIRCLES_GRID: the spacing between circles.
std::vector< cv::Point3f > calcBoardCornerPositions(
    cv::Size patternSize, float squareSize,
    PatternType patternType = PatternType::CHESSBOARD );

std::vector< cv::Mat > readImages( const std::string& dir,
    const std::string& filenamePrefix );

// Outer vector is over images.
// Inner vector is over points.
std::vector< std::vector< cv::Point2f > > detectFeatures(
    const std::vector< cv::Mat >& images,
    cv::Size patternSize );

// HACK: clean this up.
struct StereoFeatures
{
    std::vector< std::vector< cv::Point2f > > left;
    std::vector< std::vector< cv::Point2f > > right;
};

StereoFeatures detectStereoFeatures(
    const std::vector< cv::Mat >& leftImages,
    const std::vector< cv::Mat >& rightImages,
    cv::Size patternSize );

std::vector< std::vector< cv::Point3f > > generateObjectPoints(
    size_t nImages, cv::Size boardSize, float squareSize );
