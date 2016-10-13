#include "common.h"

#include <opencv2/highgui.hpp>

#include <core/io/File.h>
#include <core/io/NumberedFilenameBuilder.h>
#include <third_party/pystring.h>

std::vector< cv::Mat > readImages( const std::string& dir,
    const std::string& filenamePrefix )
{
    std::vector< cv::Mat > output;

    std::string fullPrefix = pystring::os::path::join( dir, filenamePrefix );
    std::string suffix( ".png" );

    NumberedFilenameBuilder nfb( fullPrefix, suffix );
    int i = 0;
    while( true && i < 10 )
    {
        std::string filename = nfb.filenameForNumber( i );
        printf( "Looking for image: %s...\n", filename.c_str() );
        if( !File::exists( filename.c_str() ) )
        {
            printf( "File does not exist, "
                " assuming end of collection reached.\n" );
            break;
        }

        printf( "Loading image: %s\n", filename.c_str() );
        output.push_back( cv::imread( filename ) );
        ++i;
    }
    return output;
}

std::vector< std::vector< cv::Point2f > > detectFeatures(
    const std::vector< cv::Mat >& images,
    cv::Size patternSize )
{
    std::vector< std::vector< cv::Point2f > > detectedFeaturesPerImage;

    printf( "Detecting features...\n" );
    for( int i = 0; i < static_cast< int >( images.size() ); ++i )
    {
        printf( "Image %d of %zu: ", i, images.size() );
        std::vector< cv::Point2f > foundPoints;
        bool found = cv::findCirclesGrid( images[ i ], patternSize,
            foundPoints, cv::CALIB_CB_ASYMMETRIC_GRID );
        if( found )
        {
            printf( "found %zu points.\n", foundPoints.size() );
            detectedFeaturesPerImage.push_back( foundPoints );
        }
        else
        {
            printf( "FAILED: found %zu points.\n", foundPoints.size() );
            detectedFeaturesPerImage.emplace_back();
        }
    }
    printf( "\n" );

    return detectedFeaturesPerImage;
}

StereoFeatures detectStereoFeatures(
    const std::vector< cv::Mat >& leftImages,
    const std::vector< cv::Mat >& rightImages,
    cv::Size patternSize )
{
    StereoFeatures output = {};

    if( leftImages.size() != rightImages.size() )
    {
        return output;
    }

    int nImages = static_cast< int >( leftImages.size() );

    printf( "Looking for feature points...\n" );
    for( int i = 0; i < nImages; ++i )
    {
        printf( "image %d of %d\n", i, nImages );

        std::vector< cv::Point2f > leftPoints;
        std::vector< cv::Point2f > rightPoints;

        bool leftFound = cv::findCirclesGrid( leftImages[ i ], patternSize,
            leftPoints, cv::CALIB_CB_ASYMMETRIC_GRID );
        bool rightFound = cv::findCirclesGrid( rightImages[ i ], patternSize,
            rightPoints, cv::CALIB_CB_ASYMMETRIC_GRID );

        if( !leftFound )
        {
            printf( "Failed to detect the pattern on the left image.\n" );
        }
        if( !rightFound )
        {
            printf( "Failed to detect the pattern on the right image.\n" );
        }

        if( leftFound && rightFound )
        {
            printf( "Found the pattern on both images.\n" );
            output.left.push_back( leftPoints );
            output.right.push_back( rightPoints );
        }
        else
        {
            printf( "FAILED, dropping image.\n" );
        }
    }

    return output;
}

// patternSize: number of corners or circles.
// featureSpacing: chessboard: the side length (in world units, e.g., meters) of
//   each square. CIRCLES_GRID: the spacing between circles.
std::vector< cv::Point3f > calcBoardCornerPositions(
    cv::Size patternSize, float featureSpacing,
    PatternType patternType )
{
    std::vector< cv::Point3f > corners;

    switch( patternType )
    {
    case PatternType::CHESSBOARD:
    case PatternType::CIRCLES_GRID:
        for( int i = 0; i < patternSize.height; ++i )
        {
            for( int j = 0; j < patternSize.width; ++j )
            {
                corners.push_back
                (
                    cv::Point3f
                    (
                        j * featureSpacing,
                        i * featureSpacing,
                        0
                    )
                );
            }
        }
        break;

    case PatternType::ASYMMETRIC_CIRCLES_GRID:
        for( int i = 0; i < patternSize.height; i++ )
        {
            for( int j = 0; j < patternSize.width; j++ )
            {
                corners.push_back
                (
                    cv::Point3f
                    (
                        ( 2 * j + i % 2 ) * featureSpacing,
                        i * featureSpacing,
                        0
                    )
                );
            }
        }
        break;
    default:
        break;
    }

    return corners;
}

std::vector< std::vector< cv::Point3f > > generateObjectPoints(
    size_t nImages, cv::Size boardSize, float featureSpacing )
{
    std::vector< std::vector< cv::Point3f > > objPoints;
    objPoints.reserve( nImages );
    objPoints.push_back( calcBoardCornerPositions( boardSize, featureSpacing,
        PatternType::ASYMMETRIC_CIRCLES_GRID ) );
    objPoints.resize( nImages, objPoints[ 0 ] );
    return objPoints;
}
